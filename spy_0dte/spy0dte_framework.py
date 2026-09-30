"""
SPY 0DTE regime-switching framework -- Phase 2 implementation.

Pipeline
--------
    bars  --prepare_bars-->  FeatureBuilder  -->  RegimeClassifier  -->  WalkForwardModel
                                                                            |
                                                                            v
                                                                    EntryExitEngine
                                                           (event-driven: on_bar / run)

Input contract
--------------
A DataFrame of SPY regular-session 1-minute bars:
    index     DatetimeIndex of each bar's START time. tz-aware (converted to
              America/New_York) or naive (assumed to already be New York time).
    required  open, high, low, close, volume
    optional  iv     ATM 0DTE implied volatility, annualized decimal (0.14 = 14%),
                     sampled at that bar. Without it an IV proxy is used and the
                     IV-dependent regime condition (VRP) is switched off.
              vix, vix9d   index levels sampled at that bar.

Look-ahead policy (enforced, see assert_no_lookahead)
------------------------------------------------------
1. A value stamped on bar t depends only on bars <= t.
2. Every rolling statistic used for normalization is shifted so it excludes the
   bar (or session) it normalizes. Nothing is ever back-filled.
3. Session length comes from the exchange calendar (config), never from how
   many bars happen to exist in the data.
4. Signals are formed at bar t's close and filled at bar t+1's open.
5. Models are trained only on sessions that end before the prediction window
   starts (plus an embargo); this is asserted, not assumed.
6. Labels (which look forward by definition) live in TripleBarrierLabeler and
   are never part of the feature matrix.

Units: time in minutes of the regular session; option prices per share
(multiply by 100 for one contract); P&L reported in USD per contract.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.isotonic import IsotonicRegression

try:  # LightGBM is preferred; scikit-learn's histogram GBM is a drop-in fallback.
    import lightgbm as lgb
    import warnings as _warnings
    # LightGBM 4.7's classifier maps eval_X/eval_y back onto its own deprecated
    # eval_set internally and warns about it; nothing to fix on our side.
    _warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")
    HAVE_LGBM = True
except Exception:  # pragma: no cover - depends on the environment
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_LGBM = False

log = logging.getLogger("spy0dte")

NY = "America/New_York"
SESSION_OPEN_MIN = 9 * 60 + 30          # 09:30
REGULAR_SESSION_BARS = 390              # 09:30-16:00
HALF_DAY_SESSION_BARS = 210             # 09:30-13:00
TRADING_DAYS = 252
MIN_PER_YEAR = REGULAR_SESSION_BARS * TRADING_DAYS   # trading-minute year
HOLDOUT_START = "2025-09-22"   # locked: the last 12 months of the downloaded history
ANNUALIZE_1M = math.sqrt(MIN_PER_YEAR)               # 1-min stdev -> annualized
PHASE_EDGES = (0, 30, 120, 240, 330, 10_000)          # open / AM / lunch / PM / close

CLASS_DOWN, CLASS_CHOP, CLASS_UP = 0, 1, 2
CLASS_NAMES = {CLASS_DOWN: "down", CLASS_CHOP: "chop", CLASS_UP: "up"}

BREAKOUT, CHOP, NEUTRAL = "Breakout", "Chop", "Neutral"


# =============================================================================
# 0. Bar preparation
# =============================================================================

# NYSE early closes (13:00 ET), from the published exchange calendar. Extend
# each year. Consolidated-tape vendors keep emitting after-hours bars until
# 16:00+ on these days, so without this list 13:00-16:00 after-hours trading
# would be treated as regular session. They cannot be detected from the data:
# bar counts miss them (2021-11-26 had 383 bars between 09:30 and 16:00), and
# post-13:00 volume overlaps normal days, so this published list is the source.
NYSE_EARLY_CLOSES = frozenset(date.fromisoformat(d) for d in (
    "2021-11-26",
    "2022-11-25",
    "2023-07-03", "2023-11-24",
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
))


@dataclass
class CalendarConfig:
    """Exchange-calendar knowledge. Half days (13:00 close) are published in
    advance, so knowing them is not look-ahead -- inferring them from where the
    data happens to end would be."""
    half_days: frozenset = NYSE_EARLY_CLOSES


def prepare_bars(raw: pd.DataFrame, calendar: Optional[CalendarConfig] = None) -> pd.DataFrame:
    """Validate, sort and annotate raw 1-min bars with session bookkeeping.

    Adds:
        session      tz-aware midnight of the bar's trading day
        tau          minutes since 09:30 of the bar's start (0-based)
        session_len  scheduled session length in minutes (calendar, not data)
        mins_left    minutes to the close measured at the bar's CLOSE
    """
    calendar = calendar or CalendarConfig()
    df = raw.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"bars are missing required columns: {sorted(missing)}")

    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize(NY) if idx.tz is None else idx.tz_convert(NY)
    df.index = idx
    df = df[~df.index.duplicated(keep="first")].sort_index()

    minute = df.index.hour * 60 + df.index.minute
    df = df[(minute >= SESSION_OPEN_MIN) & (minute < 16 * 60)].copy()
    minute = df.index.hour * 60 + df.index.minute

    df["session"] = df.index.normalize()
    df["tau"] = (minute - SESSION_OPEN_MIN).astype(int)
    is_half = np.array([d.date() in calendar.half_days for d in df["session"]], dtype=bool)
    df["session_len"] = np.where(is_half, HALF_DAY_SESSION_BARS, REGULAR_SESSION_BARS)
    df = df[df["tau"] < df["session_len"]].copy()           # drop bars after a half-day close
    df["mins_left"] = (df["session_len"] - df["tau"] - 1).astype(int)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[["open", "high", "low", "close"]].le(0).any(axis=1) | df[["open", "high", "low", "close"]].isna().any(axis=1)
    if bad.any():
        log.warning("dropping %d bars with missing or non-positive prices", int(bad.sum()))
        df = df[~bad].copy()
    return df


# =============================================================================
# 1. Indicator primitives (all causal)
# =============================================================================

def wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing == EMA with alpha = 1/n."""
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"],
                      (df["high"] - prev_close).abs(),
                      (df["low"] - prev_close).abs()], axis=1).max(axis=1)


def seasonal_median(x: pd.Series, tau: pd.Series, lookback: int, min_periods: int) -> pd.Series:
    """Median of x at the same minute-of-day over the previous `lookback`
    sessions. The shift(1) inside each minute's group excludes today, so the
    normalizer for 09:45 today uses only earlier days' 09:45 values."""
    return x.groupby(tau.to_numpy()).transform(
        lambda s: s.rolling(lookback, min_periods=min_periods).median().shift(1))


def rolling_z(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    """z-score against a trailing window that excludes the current bar."""
    mu = x.rolling(window, min_periods=min_periods).mean().shift(1)
    sd = x.rolling(window, min_periods=min_periods).std().shift(1)
    return (x - mu) / sd.where(sd > 0)


def rolling_pct_rank(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Percentile of the current value within the trailing window (inclusive
    of the current bar, exclusive of anything after it)."""
    return x.rolling(window, min_periods=min_periods).rank(pct=True)


# =============================================================================
# 2. FeatureBuilder
# =============================================================================

@dataclass
class FeatureConfig:
    atr_n: int = 14
    adx_n: int = 14
    rsi_n: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_n: int = 20
    rv_n: int = 30                   # bars for realized vol in f7
    seasonal_sessions: int = 20      # m_tau lookback
    seasonal_min_sessions: int = 10
    z_sessions: int = 5              # rolling z window for f1/f4
    rank_sessions: int = 5           # BBW percentile window
    gap_sessions: int = 20           # daily-vol lookback for the gap feature
    iv_proxy_sessions: int = 1       # proxy IV = trailing realized vol over N sessions ...
    iv_proxy_mult: float = 1.15      # ... times a typical implied/realized ratio
    # Ablation switch (audit item 6): False = ATR/ADX/RSI/MACD run continuously
    # across sessions with the overnight move captured by `gap`; True = every
    # indicator restarts at 09:30. Neither is assumed better -- compare them.
    session_reset_indicators: bool = False
    # When a real `iv` column is supplied, bars where it is missing become NaN
    # (not ready) instead of silently borrowing the realized-vol proxy. Mixing a
    # market IV with a synthetic one bar-by-bar makes f7 and every option price
    # inconsistent. The proxy is used only when no IV column exists at all.
    fill_missing_iv_with_proxy: bool = False
    clip: dict = field(default_factory=lambda: {
        "f1_vwap": (-6, 6), "f4_macd": (-6, 6), "f5_atr_seas": (0, 10), "f7_rv_iv": (-3, 3), "gap": (-8, 8)})


class FeatureBuilder:
    """Turns prepared 1-min bars into the stationarized feature set.

    Model features
        f1_vwap      z( (C - VWAP) / ATR14 )                trend: location vs VWAP
        f2_adx       ADX14 / 100                            trend: strength, direction-free
        f3_rsi       (RSI14 - 50) / 50                      momentum
        f4_macd      z( (MACD - signal) / ATR14 )           momentum: acceleration
        f5_atr_seas  ATR14 / median ATR14 at this minute    volatility vs time-of-day norm
        f6_bbw_rank  percentile of Bollinger bandwidth      volatility: squeeze
        f7_rv_iv     ln( RV30 / IV )                        realized vs implied
        gap          ln(session open / prior close) / daily sigma
        tau, mins_left, phase                               time of day

    NaN policy: warm-up values stay NaN (never back-filled); +/-inf becomes NaN;
    features are clipped to fixed bounds (no data-dependent clipping); a boolean
    `ready` column marks bars where every model feature is present.
    """

    MODEL_FEATURES = ["f1_vwap", "f2_adx", "f3_rsi", "f4_macd", "f5_atr_seas",
                      "f6_bbw_rank", "f7_rv_iv", "gap", "tau", "mins_left", "phase"]

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.cfg = config or FeatureConfig()

    def transform(self, bars: pd.DataFrame) -> pd.DataFrame:
        """`bars` must come from prepare_bars. Returns a new frame with the
        original columns plus features and support columns."""
        c = self.cfg
        df = bars.copy()
        H, L, C, O, V = df["high"], df["low"], df["close"], df["open"], df["volume"]
        sess = df["session"]
        day = REGULAR_SESSION_BARS

        # --- indicator primitives: continuous, or restarted every session -------
        first_bar = df["tau"].eq(0)
        prev_c_sess = C.groupby(sess).shift(1)                  # NaN on each session's first bar
        if c.session_reset_indicators:
            W = lambda x, n: x.groupby(sess).transform(lambda s: s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean())
            E = lambda x, span: x.groupby(sess).transform(lambda s: s.ewm(span=span, adjust=False).mean())
            D = lambda x: x.groupby(sess).diff()
            tr = pd.concat([H - L, (H - prev_c_sess).abs(), (L - prev_c_sess).abs()], axis=1).max(axis=1)
            bar_no = df.groupby("session").cumcount()
        else:
            W = wilder
            E = lambda x, span: x.ewm(span=span, adjust=False).mean()
            D = lambda x: x.diff()
            tr = true_range(df)
            bar_no = pd.Series(np.arange(len(df)), index=df.index)

        df["atr14"] = W(tr, c.atr_n)

        tp = (H + L + C) / 3.0
        cum_pv = (tp * V).groupby(sess).cumsum()
        cum_v = V.groupby(sess).cumsum()
        df["vwap"] = cum_pv / cum_v.where(cum_v > 0)          # resets every session

        # Intraday 1-min returns only (audit item 5). Using C.shift(1) put the
        # overnight move (prior 15:59 close -> today's 09:30 close) into the
        # first return of the day, so rv30 double-counted the gap for ~30 bars
        # while `gap` also carries it. The first bar uses open -> close, the
        # same convention the regime classifier already used.
        logret = np.log(C / prev_c_sess).where(~first_bar, np.log(C / O))
        df["rv30"] = logret.rolling(c.rv_n, min_periods=c.rv_n).std() * ANNUALIZE_1M

        # Implied vol: the real column when supplied (gaps stay NaN unless
        # fill_missing_iv_with_proxy), otherwise a proxy flagged on every bar.
        if "iv" in df.columns and df["iv"].notna().any():
            df["iv_is_proxy"] = df["iv"].isna() & c.fill_missing_iv_with_proxy
            df["iv_used"] = (df["iv"].where(df["iv"].notna(), self._iv_proxy(logret))
                             if c.fill_missing_iv_with_proxy else df["iv"])
        else:
            df["iv_is_proxy"] = True
            df["iv_used"] = self._iv_proxy(logret)

        # --- Trend -------------------------------------------------------------
        raw_f1 = (C - df["vwap"]) / df["atr14"]
        df["f1_vwap"] = rolling_z(raw_f1, c.z_sessions * day, c.z_sessions * day // 2)

        up, dn = D(H), -D(L)
        plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
        atr_w = W(tr, c.adx_n)
        pdi = 100 * W(plus_dm, c.adx_n) / atr_w
        mdi = 100 * W(minus_dm, c.adx_n) / atr_w
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi).where((pdi + mdi) > 0)
        df["f2_adx"] = W(dx, c.adx_n) / 100.0

        # --- Momentum ----------------------------------------------------------
        delta = D(C)
        avg_gain = W(delta.clip(lower=0), c.rsi_n)
        avg_loss = W(-delta.clip(upper=0), c.rsi_n)
        rsi = 100 - 100 / (1 + avg_gain / avg_loss.where(avg_loss > 0))
        rsi = rsi.where(avg_loss > 0, 100.0).where(avg_gain.notna())   # all-gain window -> 100
        df["f3_rsi"] = (rsi - 50) / 50

        macd = E(C, c.macd_fast) - E(C, c.macd_slow)
        hist = macd - E(macd, c.macd_signal)
        warm = bar_no >= c.macd_slow + c.macd_signal
        raw_f4 = (hist / df["atr14"]).where(warm)
        df["f4_macd"] = rolling_z(raw_f4, c.z_sessions * day, c.z_sessions * day // 2)

        # --- Volatility --------------------------------------------------------
        df["f5_atr_seas"] = df["atr14"] / seasonal_median(
            df["atr14"], df["tau"], c.seasonal_sessions, c.seasonal_min_sessions)

        sma = C.rolling(c.bb_n, min_periods=c.bb_n).mean()
        sd = C.rolling(c.bb_n, min_periods=c.bb_n).std()
        bbw = 4 * sd / sma
        df["f6_bbw_rank"] = rolling_pct_rank(bbw, c.rank_sessions * day, c.rank_sessions * day // 2)

        df["f7_rv_iv"] = np.log(df["rv30"] / df["iv_used"])

        # --- Overnight gap (known at the open, constant through the session) ---
        df["gap"] = self._gap_feature(df)

        # --- Time of day -------------------------------------------------------
        df["phase"] = pd.cut(df["tau"], bins=list(PHASE_EDGES), right=False, labels=False).astype(float)

        # --- Hygiene -----------------------------------------------------------
        feats = self.MODEL_FEATURES
        df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
        for col, (lo, hi) in c.clip.items():
            df[col] = df[col].clip(lo, hi)
        df["ready"] = df[feats].notna().all(axis=1) & df["iv_used"].notna() & (df["iv_used"] > 0)
        return df

    # ------------------------------------------------------------------------
    def _iv_proxy(self, logret: pd.Series) -> pd.Series:
        n = self.cfg.iv_proxy_sessions * REGULAR_SESSION_BARS
        return logret.rolling(n, min_periods=n // 2).std() * ANNUALIZE_1M * self.cfg.iv_proxy_mult

    def _gap_feature(self, df: pd.DataFrame) -> pd.Series:
        """ln(today's open / prior close) / sigma of daily returns over prior sessions."""
        daily = df.groupby("session").agg(first_open=("open", "first"), last_close=("close", "last"))
        prev_close = daily["last_close"].shift(1)
        daily_ret = np.log(daily["last_close"] / prev_close)
        sigma = daily_ret.rolling(self.cfg.gap_sessions, min_periods=5).std().shift(1)
        gap = np.log(daily["first_open"] / prev_close) / sigma.where(sigma > 0)
        return df["session"].map(gap)


# =============================================================================
# 3. RegimeClassifier
# =============================================================================

@dataclass
class RegimeConfig:
    checkpoints: tuple = (30, 120)   # decide at 10:00, re-check at 11:30 (minutes after open)
    opening_range_bars: int = 30
    value_area_pct: float = 0.70
    value_area_bin: float = 0.05     # $ price bucket for the volume profile
    seasonal_sessions: int = 20
    seasonal_min_sessions: int = 10
    ratr_breakout: float = 1.25
    ratr_chop: float = 0.90
    vrp_breakout: float = 1.00
    vrp_chop: float = 0.80
    ts_breakout: float = 1.00
    ts_chop: float = 0.95
    vix_rise_breakout: float = 0.05
    vix_chop_low: float = 13.0
    vix_chop_high: float = 25.0
    xvwap_breakout_rate: float = 2.0  # max VWAP crosses per 30 bars
    xvwap_chop_rate: float = 4.0      # min VWAP crosses per 30 bars


class RegimeClassifier:
    """Labels every bar Breakout, Chop or Neutral.

    The regime is decided at each checkpoint from data up to and including the
    checkpoint bar's close, then held for the rest of the session until the next
    checkpoint. Bars before the first checkpoint are Neutral. `Neutral` means
    neither rule set was fully met -- the no-trade state.

    R_atr and VRP are both seasonally normalized (divided by their median at the
    same minute over prior sessions), so 1.0 means "an ordinary day so far".

    Breakout (buy gamma) requires ALL of:
        open outside prior-day value area AND opening range fully outside it
        R_atr >= ratr_breakout
        VRP   >= vrp_breakout                          (only with real IV)
        TS >= ts_breakout  OR  VIX up >= vix_rise since open   (only with VIX data)
        VWAP cross rate <= xvwap_breakout_rate
    Chop (sell theta) requires ALL of:
        open inside prior-day value area
        R_atr <= ratr_chop
        VRP   <= vrp_chop                              (only with real IV)
        TS < ts_chop AND vix_chop_low <= VIX <= vix_chop_high   (only with VIX data)
        VWAP cross rate >= xvwap_chop_rate

    A condition whose input column is absent from the data is dropped (logged
    once). A condition whose input is present but NaN on a bar is False.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.cfg = config or RegimeConfig()
        self.active_conditions: dict = {}

    def classify(self, feats: pd.DataFrame) -> pd.DataFrame:
        """`feats` from FeatureBuilder.transform. Returns a frame (same index)
        with the regime label plus every input, for diagnostics."""
        c = self.cfg
        df = feats
        out = pd.DataFrame(index=df.index)
        sess, tau = df["session"], df["tau"]

        # --- prior-day volume profile (value area), carried to the next session
        va = self._value_area_by_session(df)
        prior = va.shift(1)
        out["val"] = sess.map(prior["val"])
        out["vah"] = sess.map(prior["vah"])
        out["poc"] = sess.map(prior["poc"])

        # --- opening range: only defined once the last OR bar has closed -------
        or_mask = tau < c.opening_range_bars
        or_hi = df["high"].where(or_mask).groupby(sess).cummax()
        or_lo = df["low"].where(or_mask).groupby(sess).cummin()
        or_hi = or_hi.groupby(sess).ffill()
        or_lo = or_lo.groupby(sess).ffill()
        complete = tau >= c.opening_range_bars - 1
        out["or_high"] = or_hi.where(complete)
        out["or_low"] = or_lo.where(complete)
        out["session_open"] = df["open"].groupby(sess).transform("first")   # known at 09:30

        # --- ATR expansion since the open vs the same window on prior days ----
        tr_s = true_range(df)
        first_bar = tau.eq(0)
        tr_s = tr_s.where(~first_bar, df["high"] - df["low"])            # no overnight gap in bar 1
        mean_tr = tr_s.groupby(sess).cumsum() / (tau + 1)
        out["ratr"] = mean_tr / seasonal_median(mean_tr, tau, c.seasonal_sessions, c.seasonal_min_sessions)

        # --- realized vol since open vs implied (VRP) -------------------------
        r = np.log(df["close"] / df["close"].shift(1))
        r = r.where(~first_bar, np.log(df["close"] / df["open"]))
        n = tau + 1
        s1 = r.groupby(sess).cumsum()
        s2 = (r * r).groupby(sess).cumsum()
        var = (s2 - s1 * s1 / n) / (n - 1).where(n > 1)
        out["rv_open"] = np.sqrt(var.clip(lower=0)) * ANNUALIZE_1M
        has_real_iv = "iv" in df.columns and not bool(df["iv_is_proxy"].all())
        # Raw realized/implied is biased upward early in the session: the first
        # 30 minutes are structurally more volatile than the day's average, so
        # RV(09:30-10:00) / full-day IV sits well above 1.0 on an ordinary day
        # and a "VRP <= 0.8" rule can essentially never fire. Normalize by the
        # typical ratio at the same minute on prior sessions, so 1.0 means an
        # ordinary day and the thresholds read as "faster/slower than usual,
        # relative to what options priced".
        if has_real_iv:
            out["vrp_raw"] = out["rv_open"] / df["iv"]
            out["vrp"] = out["vrp_raw"] / seasonal_median(out["vrp_raw"], tau, c.seasonal_sessions,
                                                          c.seasonal_min_sessions)
        else:
            out["vrp_raw"] = np.nan
            out["vrp"] = np.nan

        # --- VIX term structure and intraday change ---------------------------
        has_vix = "vix" in df.columns and df["vix"].notna().any()
        has_vix9d = has_vix and "vix9d" in df.columns and df["vix9d"].notna().any()
        out["ts"] = (df["vix9d"] / df["vix"]) if has_vix9d else np.nan
        if has_vix:
            vix_open = df["vix"].groupby(sess).transform("first")
            out["vix"] = df["vix"]
            out["vix_chg"] = df["vix"] / vix_open - 1
        else:
            out["vix"] = np.nan
            out["vix_chg"] = np.nan

        # --- VWAP crossing rate ------------------------------------------------
        side = np.sign(df["close"] - df["vwap"])
        prev_side = side.groupby(sess).shift(1)
        crossed = (side != prev_side) & (side != 0) & (prev_side != 0) & prev_side.notna()
        out["vwap_crosses"] = crossed.astype(int).groupby(sess).cumsum()
        out["xvwap_rate"] = out["vwap_crosses"] / ((tau + 1) / 30.0)

        # --- rule sets ---------------------------------------------------------
        above = (out["session_open"] > out["vah"]) & (out["or_low"] > out["vah"])
        below = (out["session_open"] < out["val"]) & (out["or_high"] < out["val"])
        inside = (out["session_open"] >= out["val"]) & (out["session_open"] <= out["vah"])

        breakout = {
            "open_outside_value": above | below,
            "atr_expansion": out["ratr"] >= c.ratr_breakout,
            "vrp_rich_realized": out["vrp"] >= c.vrp_breakout,
            "vol_stress": (out["ts"] >= c.ts_breakout) | (out["vix_chg"] >= c.vix_rise_breakout),
            "few_vwap_crosses": out["xvwap_rate"] <= c.xvwap_breakout_rate,
        }
        chop = {
            "open_inside_value": inside,
            "atr_contraction": out["ratr"] <= c.ratr_chop,
            "vrp_rich_implied": out["vrp"] <= c.vrp_chop,
            "calm_vol": (out["ts"] < c.ts_chop) & out["vix"].between(c.vix_chop_low, c.vix_chop_high),
            "many_vwap_crosses": out["xvwap_rate"] >= c.xvwap_chop_rate,
        }
        availability = {
            "vrp_rich_realized": has_real_iv, "vrp_rich_implied": has_real_iv,
            "vol_stress": has_vix, "calm_vol": has_vix9d,
        }
        for name, ok in availability.items():
            if not ok and name not in self.active_conditions:
                log.warning("regime condition '%s' disabled: input data not supplied", name)
        breakout = {k: v for k, v in breakout.items() if availability.get(k, True)}
        chop = {k: v for k, v in chop.items() if availability.get(k, True)}
        self.active_conditions = {**{k: True for k in breakout}, **{k: True for k in chop},
                                  **{k: False for k, ok in availability.items() if not ok}}

        is_breakout = pd.concat(breakout, axis=1).fillna(False).astype(bool).all(axis=1)
        is_chop = pd.concat(chop, axis=1).fillna(False).astype(bool).all(axis=1)
        for k, v in {**breakout, **chop}.items():
            out[f"cond_{k}"] = v.fillna(False).astype(bool)

        # --- latch decisions at checkpoint bars --------------------------------
        decision_bar = tau.isin([cp - 1 for cp in c.checkpoints])
        decision = pd.Series(np.where(is_breakout, BREAKOUT, np.where(is_chop, CHOP, NEUTRAL)),
                             index=df.index, dtype=object).where(decision_bar)
        out["regime"] = decision.groupby(sess).ffill().fillna(NEUTRAL)
        out["regime_decided_at"] = pd.Series(np.where(decision_bar, tau, np.nan),
                                             index=df.index).groupby(sess).ffill()
        return out

    def _value_area_by_session(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per session: POC and the value area holding value_area_pct of volume,
        from a typical-price volume profile. Used only for the NEXT session."""
        c = self.cfg
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        rows = {}
        for s, idx in df.groupby("session").indices.items():
            prices, vols = tp.to_numpy()[idx], df["volume"].to_numpy()[idx]
            ok = np.isfinite(prices) & np.isfinite(vols) & (vols > 0)
            if ok.sum() < 5:
                rows[s] = (np.nan, np.nan, np.nan)
                continue
            b = np.round(prices[ok] / c.value_area_bin).astype(np.int64)
            prof = pd.Series(vols[ok]).groupby(b).sum().sort_index()
            levels, v = prof.index.to_numpy(), prof.to_numpy()
            poc_i = int(np.argmax(v))
            lo_i = hi_i = poc_i
            total, target, acc = v.sum(), v.sum() * c.value_area_pct, v[poc_i]
            while acc < target and (lo_i > 0 or hi_i < len(v) - 1):
                below_v = v[lo_i - 1] if lo_i > 0 else -1
                above_v = v[hi_i + 1] if hi_i < len(v) - 1 else -1
                if above_v >= below_v:
                    hi_i += 1
                    acc += v[hi_i]
                else:
                    lo_i -= 1
                    acc += v[lo_i]
            rows[s] = (levels[lo_i] * c.value_area_bin, levels[hi_i] * c.value_area_bin,
                       levels[poc_i] * c.value_area_bin)
        return pd.DataFrame.from_dict(rows, orient="index", columns=["val", "vah", "poc"]).sort_index()


def regime_validation_report(feats: pd.DataFrame, regimes: pd.DataFrame, checkpoint: int = 30) -> pd.DataFrame:
    """EVALUATION ONLY (uses the full day, i.e. the future relative to the
    decision). Compares the regime decided at `checkpoint` with the day's
    realized efficiency ratio ER = |close - open| / (high - low)."""
    day = feats.groupby("session").agg(o=("open", "first"), h=("high", "max"),
                                       l=("low", "min"), c=("close", "last"))
    er = (day["c"] - day["o"]).abs() / (day["h"] - day["l"])
    realized = pd.Series(np.where(er >= 0.6, "trend day", np.where(er <= 0.3, "chop day", "mixed")), index=day.index)
    at_cp = regimes.loc[feats["tau"] == checkpoint - 1, "regime"]
    at_cp.index = feats.loc[feats["tau"] == checkpoint - 1, "session"].to_numpy()
    both = pd.DataFrame({"called": at_cp, "realized": realized}).dropna()
    return pd.crosstab(both["called"], both["realized"], margins=True)


# =============================================================================
# 4. Labels (the ONLY forward-looking component; never a feature)
# =============================================================================

@dataclass
class LabelConfig:
    """horizon=None means "until the end-of-day exit" (15:45, or 15 minutes
    before a half-day close) -- the label for trades held to that exit."""
    horizon: Optional[int] = 30              # minutes, or None = until the EOD exit
    barrier_k: float = 0.75                  # barrier = k x implied move over the horizon
    eod_exit_minutes_before_close: int = 15


DIRECTION_LABEL = LabelConfig(horizon=30, barrier_k=0.75)
# A 0.30-delta short strike sits ~N^-1(0.70) = 0.524 sigma*sqrt(T) from spot, so
# "stayed inside +/-0.524 x implied move until 15:45" is the spread's own question.
SPREAD_LABEL = LabelConfig(horizon=None, barrier_k=0.524)


def label_horizon(feats: pd.DataFrame, cfg: LabelConfig) -> np.ndarray:
    """Bars a label (and the matching trade) can look forward from bar t."""
    mins_left = feats["mins_left"].to_numpy()
    to_eod = np.maximum(mins_left - cfg.eod_exit_minutes_before_close, 0)
    return to_eod if cfg.horizon is None else np.minimum(cfg.horizon, to_eod)


class TripleBarrierLabeler:
    """0 = down barrier first, 1 = neither (chop), 2 = up barrier first.

    Measured from the EXECUTABLE entry -- a signal at bar t fills at bar t+1's
    open -- over bars t+1 .. t+H_eff (audit item 1). Measuring from close[t]
    credited the model with overnight-of-a-minute moves the trade could not
    capture.
        b_t   = k * IV_t * open[t+1] * sqrt(H_eff / MIN_PER_YEAR)
        H_eff = min(horizon, minutes until the end-of-day exit)
    A bar where both barriers print inside the same minute is left unlabeled
    (the order inside a 1-min bar is unknowable). The last bar of a session has
    no same-session fill and is unlabeled.
    """

    def __init__(self, config: Optional[LabelConfig] = None):
        self.cfg = config or DIRECTION_LABEL

    def label(self, feats: pd.DataFrame) -> pd.Series:
        k = self.cfg.barrier_k
        hi, lo = (feats[c].to_numpy(dtype=float) for c in ("high", "low"))
        sess = feats["session"].to_numpy()
        n = len(feats)
        nxt_open = feats["open"].shift(-1).to_numpy(dtype=float)
        same_sess = np.r_[sess[1:] == sess[:-1], False]
        entry = np.where(same_sess, nxt_open, np.nan)
        h_eff = label_horizon(feats, self.cfg)
        b = k * feats["iv_used"].to_numpy(dtype=float) * entry * np.sqrt(np.maximum(h_eff, 0) / MIN_PER_YEAR)
        up_lvl, dn_lvl = entry + b, entry - b

        y = np.full(n, np.nan)
        decided = np.zeros(n, dtype=bool)
        rows = np.arange(n)
        for j in range(1, int(np.nanmax(h_eff)) + 1 if n else 1):
            fwd = np.minimum(rows + j, n - 1)
            valid = (rows + j < n) & (j <= h_eff) & (sess[fwd] == sess) & ~decided
            hit_up = valid & (hi[fwd] >= up_lvl)
            hit_dn = valid & (lo[fwd] <= dn_lvl)
            tie = hit_up & hit_dn
            y[hit_up & ~tie] = CLASS_UP
            y[hit_dn & ~tie] = CLASS_DOWN
            decided |= hit_up | hit_dn          # ties are decided but stay NaN
        chop = ~decided & (h_eff >= 1) & np.isfinite(b) & (b > 0)
        y[chop] = CLASS_CHOP
        name = "label_eod" if self.cfg.horizon is None else f"label_h{self.cfg.horizon}"
        return pd.Series(y, index=feats.index, name=name)


# =============================================================================
# 5. Collinearity tools (fitted per training fold only)
# =============================================================================

def vif_table(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor per column (rows with any NaN dropped)."""
    Z = X.dropna().to_numpy(dtype=float)
    out = {}
    for i, col in enumerate(X.columns):
        y = Z[:, i]
        others = np.delete(Z, i, axis=1)
        A = np.column_stack([np.ones(len(y)), others])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0.0
        out[col] = 1.0 / max(1e-12, 1 - r2)
    return pd.Series(out, name="VIF")


class Residualizer:
    """For each (target, driver) pair, replaces target with target - beta*driver
    when target's VIF exceeds the threshold ON THE TRAINING FOLD."""

    def __init__(self, pairs, vif_threshold: float = 5.0):
        self.pairs, self.vif_threshold, self.betas = tuple(pairs), vif_threshold, {}

    def fit(self, X: pd.DataFrame) -> "Residualizer":
        self.betas = {}
        vif = vif_table(X)
        for target, driver in self.pairs:
            if target in X and driver in X and vif.get(target, 0) > self.vif_threshold:
                pair = X[[target, driver]].dropna()
                var = pair[driver].var()
                if var > 0:
                    self.betas[(target, driver)] = pair[target].cov(pair[driver]) / var
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for (target, driver), beta in self.betas.items():
            X[target] = X[target] - beta * X[driver]
        return X


# =============================================================================
# 6. Walk-forward model, baselines, calibration, holdout gate
# =============================================================================

VARIANTS = ("lgb_raw", "lgb_isotonic", "lgb_temperature", "base_rate", "logistic")


@dataclass
class ModelConfig:
    label: LabelConfig = field(default_factory=lambda: DIRECTION_LABEL)
    sample_step: Optional[int] = None     # training stride; default = horizon (60 for EOD labels)
    min_train_sessions: int = 120
    embargo_sessions: int = 1
    # Training sessions are split in TIME ORDER into fit / early-stop / calibrate.
    early_stop_frac: float = 0.15
    calib_frac: float = 0.15
    early_stopping_rounds: int = 50
    min_calib_per_class: int = 100        # below this, calibration falls back to raw (audit item 13)
    calibration: str = "lgb_temperature"  # which variant feeds the engine: lgb_raw | lgb_isotonic | lgb_temperature
    retrain_freq: str = "M"
    residualize_pairs: tuple = (("f4_macd", "f3_rsi"),)
    vif_threshold: float = 5.0
    # Final holdout (audit item 10): sessions on/after holdout_start are never
    # trained on or predicted unless allow_holdout=True, which only
    # run_holdout.py sets (and it logs every look).
    holdout_start: Optional[str] = None
    allow_holdout: bool = False
    lgb_params: dict = field(default_factory=lambda: dict(
        objective="multiclass", num_class=3, num_leaves=31, min_child_samples=500,
        colsample_bytree=0.7, learning_rate=0.03, n_estimators=2000, verbose=-1))


def _softmax_log(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-9, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Single temperature T minimizing multiclass NLL on the calibration fold.
    Unlike one-vs-rest isotonic + renormalization, this keeps the three
    probabilities a coherent distribution (audit item 14)."""
    from scipy.optimize import minimize_scalar
    nll = lambda T: -np.mean(np.log(np.clip(_softmax_log(p, T)[np.arange(len(y)), y], 1e-12, 1)))
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


class WalkForwardModel:
    """Rolling, strictly out-of-sample class probabilities, plus baselines.

    For each prediction period P (a month by default):
        train sessions  = every session before P, minus `embargo_sessions`,
                          and never a session on/after holdout_start
        fit | stop | cal = time-ordered split of those sessions
        predict          = every ready bar in P
    Output of fit_predict: the chosen calibrated variant as <prefix>down/chop/up.
    All variants (VARIANTS) are kept in self.variants for evaluate_forecasts().
    """

    def __init__(self, features: Optional[list] = None, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig()
        self.features = list(features or FeatureBuilder.MODEL_FEATURES)
        self.labeler = TripleBarrierLabeler(self.cfg.label)
        self.fold_log: list = []
        self.variants: dict = {}
        self.labels: Optional[pd.Series] = None
        if self.cfg.calibration not in ("lgb_raw", "lgb_isotonic", "lgb_temperature"):
            raise ValueError(f"calibration must be lgb_raw, lgb_isotonic or lgb_temperature, not {self.cfg.calibration!r}")

    def step(self) -> int:
        c = self.cfg
        return c.sample_step or (c.label.horizon if c.label.horizon else 60)

    def fit_predict(self, feats: pd.DataFrame, prefix: str = "p_") -> pd.DataFrame:
        cfg, step = self.cfg, self.step()
        y = self.labeler.label(feats)
        self.labels = y
        if y.name in self.features:
            raise AssertionError("label column leaked into the feature list")

        sessions = pd.Index(sorted(feats["session"].unique()))
        hold_ts = pd.Timestamp(cfg.holdout_start, tz=NY) if cfg.holdout_start else None
        period_of = pd.Series(sessions.tz_localize(None).to_period(cfg.retrain_freq), index=sessions)
        cols = [f"{prefix}down", f"{prefix}chop", f"{prefix}up"]
        store = {v: np.full((len(feats), 3), np.nan) for v in VARIANTS}
        fold_col = np.full(len(feats), np.nan)

        for fold, (period, test_sessions) in enumerate(period_of.groupby(period_of).groups.items()):
            first_test = test_sessions.min()
            if hold_ts is not None and not cfg.allow_holdout and first_test >= hold_ts:
                continue                                           # holdout is locked
            prior = sessions[sessions < first_test]
            if hold_ts is not None and not cfg.allow_holdout:
                prior = prior[prior < hold_ts]
            train_sessions = prior[: max(0, len(prior) - cfg.embargo_sessions)]
            if len(train_sessions) < cfg.min_train_sessions:
                continue

            n = len(train_sessions)
            n_cal, n_es = max(1, int(n * cfg.calib_frac)), max(1, int(n * cfg.early_stop_frac))
            fit_s = train_sessions[: n - n_cal - n_es]
            es_s = train_sessions[n - n_cal - n_es: n - n_cal]
            cal_s = train_sessions[n - n_cal:]
            usable = feats["ready"] & y.notna() & (feats["tau"] % step == 0)
            rows = {k: feats["session"].isin(v) & usable for k, v in (("fit", fit_s), ("es", es_s), ("cal", cal_s))}
            test_rows = feats["session"].isin(test_sessions) & feats["ready"]
            if hold_ts is not None and not cfg.allow_holdout:
                test_rows &= feats["session"] < hold_ts
            if rows["fit"].sum() < 200 or rows["cal"].sum() < 50 or test_rows.sum() == 0:
                continue
            train_mask = rows["fit"] | rows["es"] | rows["cal"]
            assert feats.index[train_mask].max() < feats.index[test_rows].min(), \
                "look-ahead: training data overlaps the prediction window"

            res = Residualizer(cfg.residualize_pairs, cfg.vif_threshold).fit(feats.loc[rows["fit"], self.features])
            X = {k: res.transform(feats.loc[m, self.features]) for k, m in rows.items()}
            Xt = res.transform(feats.loc[test_rows, self.features])
            Y = {k: y[m].astype(int).to_numpy() for k, m in rows.items()}
            if len(np.unique(Y["fit"])) < 3:
                continue

            model, n_trees = self._fit_model(X["fit"], Y["fit"], X["es"], Y["es"])
            p_cal, p_test = self._proba(model, X["cal"]), self._proba(model, Xt)
            idx = np.flatnonzero(test_rows.to_numpy())
            store["lgb_raw"][idx] = p_test

            per_class = np.bincount(Y["cal"], minlength=3)
            calib_ok = per_class.min() >= cfg.min_calib_per_class
            if calib_ok:
                store["lgb_isotonic"][idx] = self._isotonic(p_cal, Y["cal"], p_test)
                T = fit_temperature(p_cal, Y["cal"])
                store["lgb_temperature"][idx] = _softmax_log(p_test, T)
            else:
                T = float("nan")
                store["lgb_isotonic"][idx] = p_test          # too few per class: leave uncalibrated
                store["lgb_temperature"][idx] = p_test

            # Baselines, fitted on the same training sessions only.
            train_y = np.concatenate([Y["fit"], Y["es"], Y["cal"]])
            train_phase = pd.concat([feats.loc[rows[k], "phase"] for k in ("fit", "es", "cal")]).to_numpy()
            store["base_rate"][idx] = self._base_rate(train_phase, train_y, feats.loc[test_rows, "phase"].to_numpy())
            store["logistic"][idx] = self._logistic(pd.concat([X["fit"], X["es"]]),
                                                    np.concatenate([Y["fit"], Y["es"]]), Xt)
            fold_col[idx] = fold
            self.fold_log.append(dict(period=str(period), fit_sessions=len(fit_s), es_sessions=len(es_s),
                                      cal_sessions=len(cal_s), trees=n_trees, temperature=T,
                                      calib_per_class=per_class.tolist(), calibrated=bool(calib_ok),
                                      test_rows=int(test_rows.sum()), residualized=list(res.betas)))

        self.variants = {v: pd.DataFrame(store[v], index=feats.index, columns=cols) for v in VARIANTS}
        out = self.variants[cfg.calibration].copy()
        out[f"{prefix}fold"] = fold_col
        return out

    # ------------------------------------------------------------------------
    def _fit_model(self, Xf, yf, Xe, ye):
        """LightGBM with early stopping on the time-ordered early-stop block
        (audit item 11). Falls back to fixed-size sklearn HGB without LightGBM."""
        if HAVE_LGBM:
            model = lgb.LGBMClassifier(**self.cfg.lgb_params)
            if len(ye) >= 50 and len(np.unique(ye)) == 3:
                stop = [lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)]
                try:        # LightGBM >= 4.7 names the validation set eval_X / eval_y
                    model.fit(Xf, yf, eval_X=[Xe], eval_y=[ye], eval_metric="multi_logloss", callbacks=stop)
                except TypeError:
                    model.fit(Xf, yf, eval_set=[(Xe, ye)], eval_metric="multi_logloss", callbacks=stop)
                return model, int(model.best_iteration_ or self.cfg.lgb_params["n_estimators"])
            model.set_params(n_estimators=400)
            model.fit(Xf, yf)
            return model, 400
        p = self.cfg.lgb_params
        model = HistGradientBoostingClassifier(max_leaf_nodes=p.get("num_leaves", 31),
                                               min_samples_leaf=p.get("min_child_samples", 500),
                                               learning_rate=p.get("learning_rate", 0.03), max_iter=400)
        model.fit(Xf, yf)
        return model, 400

    @staticmethod
    def _proba(model, X) -> np.ndarray:
        raw = model.predict_proba(X)
        full = np.zeros((len(X), 3))
        for j, cls in enumerate(model.classes_):
            full[:, int(cls)] = raw[:, j]
        return full

    @staticmethod
    def _isotonic(p_cal, y_cal, p_test) -> np.ndarray:
        """One-vs-rest isotonic, bounded to [0.01, 0.99], then renormalized.
        Kept for comparison; renormalization moves each class off its own
        calibrated value, so temperature scaling is the default."""
        q = np.column_stack([IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
                             .fit(p_cal[:, k], (y_cal == k).astype(float)).predict(p_test[:, k]) for k in range(3)])
        s = q.sum(axis=1, keepdims=True)
        return q / np.where(s > 0, s, 1.0)

    @staticmethod
    def _base_rate(train_phase, train_y, test_phase) -> np.ndarray:
        overall = np.bincount(train_y, minlength=3) / len(train_y)
        table = {}
        for ph in np.unique(train_phase):
            yy = train_y[train_phase == ph]
            table[ph] = np.bincount(yy, minlength=3) / len(yy)
        return np.array([table.get(ph, overall) for ph in test_phase])

    @staticmethod
    def _logistic(X, y, Xt) -> np.ndarray:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        m.fit(X, y)
        raw = m.predict_proba(Xt)
        full = np.zeros((len(Xt), 3))
        for j, cls in enumerate(m.classes_):
            full[:, int(cls)] = raw[:, j]
        return full


def evaluate_forecasts(feats: pd.DataFrame, labels: pd.Series, variants: dict, step: int) -> pd.DataFrame:
    """Out-of-sample scorecard on non-overlapping bars (tau % step == 0).
    For each variant: log-loss, Brier, magnitude AUC (chop vs moved) and
    direction AUC (up vs down, given a move). Paired, session-clustered t-stats
    compare every LightGBM variant with the base-rate and logistic baselines."""
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    mask = (feats["tau"] % step == 0) & labels.notna()
    for v in variants.values():
        mask &= v.iloc[:, 0].notna()
    y = labels[mask].astype(int).to_numpy()
    sess = feats.loc[mask, "session"].to_numpy()
    Y = np.eye(3)[y]
    rows, loss = [], {}
    for name, df in variants.items():
        P = df.loc[mask].iloc[:, :3].to_numpy()
        ll = -np.log(np.clip((P * Y).sum(1), 1e-9, 1))
        loss[name] = ll
        moved = y != CLASS_CHOP
        dir_p = P[moved, 2] / np.clip(P[moved, 2] + P[moved, 0], 1e-9, None)
        rows.append(dict(variant=name, n=len(y), log_loss=ll.mean(), brier=((P - Y) ** 2).sum(1).mean(),
                         auc_magnitude=roc_auc_score(y == CLASS_CHOP, P[:, 1]) if len(np.unique(y == CLASS_CHOP)) > 1 else np.nan,
                         auc_direction=roc_auc_score(y[moved] == CLASS_UP, dir_p) if len(np.unique(y[moved])) > 1 else np.nan))
    out = pd.DataFrame(rows).set_index("variant")
    for base in ("base_rate", "logistic"):
        col = []
        for name in out.index:
            if name == base or base not in loss:
                col.append(np.nan)
                continue
            d = pd.Series(loss[name] - loss[base]).groupby(sess).mean()
            col.append(stats.ttest_1samp(d, 0).statistic)
        out[f"t_vs_{base}"] = col          # negative = better than that baseline
    return out


def calibration_by_phase(feats: pd.DataFrame, labels: pd.Series, probs: pd.DataFrame, step: int,
                         n_bins: int = 10) -> pd.DataFrame:
    """Expected calibration error per class and session phase (audit item 13)."""
    mask = (feats["tau"] % step == 0) & labels.notna() & probs.iloc[:, 0].notna()
    y = labels[mask].astype(int).to_numpy()
    P = probs.loc[mask].iloc[:, :3].to_numpy()
    ph = feats.loc[mask, "phase"].to_numpy()
    rows = []
    for phase in np.unique(ph):
        m = ph == phase
        row = {"phase": int(phase), "n": int(m.sum())}
        for k, nm in CLASS_NAMES.items():
            p, hit = P[m, k], (y[m] == k).astype(float)
            bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
            ece = sum(abs(p[bins == b].mean() - hit[bins == b].mean()) * (bins == b).mean()
                      for b in range(n_bins) if (bins == b).any())
            row[f"ece_{nm}"] = ece
        rows.append(row)
    return pd.DataFrame(rows).set_index("phase")


# =============================================================================
# 7. Option pricing: Black-Scholes (fallback) and real traded prices
# =============================================================================

MIN_T_MINUTES = 0.5   # floor so a contract at the bell still has a finite price


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _t_years(minutes_left: float) -> float:
    return max(minutes_left, MIN_T_MINUTES) / MIN_PER_YEAR


def bs_price(S: float, K: float, minutes_left: float, sigma: float, right: str, r: float = 0.0) -> float:
    T = _t_years(minutes_left)
    if not (sigma > 0 and np.isfinite(sigma)):
        return max(0.0, S - K) if right == "C" else max(0.0, K - S)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    disc = math.exp(-r * T)
    if right == "C":
        return S * _ncdf(d1) - K * disc * _ncdf(d2)
    return K * disc * _ncdf(-d2) - S * _ncdf(-d1)


def bs_delta(S: float, K: float, minutes_left: float, sigma: float, right: str, r: float = 0.0) -> float:
    T = _t_years(minutes_left)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1) if right == "C" else _ncdf(d1) - 1.0


def bs_theta_per_minute(S: float, K: float, minutes_left: float, sigma: float, right: str, r: float = 0.0) -> float:
    """Price change per trading minute from time decay alone (negative for a long option)."""
    T = _t_years(minutes_left)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    decay = -S * _npdf(d1) * sigma / (2.0 * math.sqrt(T))
    carry = -r * K * math.exp(-r * T) * (_ncdf(d2) if right == "C" else -_ncdf(-d2))
    return (decay + carry) / MIN_PER_YEAR


def implied_vol(price: float, S: float, K: float, minutes_left: float, right: str, r: float = 0.0) -> float:
    """Bisection IV on THIS module's trading-minute clock. Used to value
    scenarios consistently with an observed option price (audit item 4: a
    vendor IV computed on another clock would not reproduce the quote)."""
    intrinsic = max(0.0, S - K) if right == "C" else max(0.0, K - S)
    if not np.isfinite(price) or price <= intrinsic + 1e-6:
        return float("nan")
    lo, hi = 1e-3, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, minutes_left, mid, right, r) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def strike_for_delta(S: float, minutes_left: float, sigma: float, abs_delta: float, right: str,
                     increment: float = 1.0, r: float = 0.0) -> float:
    """Listed strike closest to the target |delta| under Black-Scholes."""
    T = _t_years(minutes_left)
    d1 = float(ndtri(abs_delta if right == "C" else 1.0 - abs_delta))
    k_exact = S * math.exp(-d1 * sigma * math.sqrt(T) + (r + 0.5 * sigma * sigma) * T)
    return round(k_exact / increment) * increment


def occ_symbol(root: str, session: pd.Timestamp, right: str, strike: float) -> str:
    return f"{root}{session:%y%m%d}{right}{int(round(strike * 1000)):08d}"


@dataclass
class QuoteModel:
    """Bid/ask around a mid price (audit item 18). Without historical quotes
    the width is MODELED: max(min_width, width_pct x mid), widened late in the
    session. Fills cross `lam` of the full width from the mid:
        buy  = mid + lam * width        sell = mid - lam * width
    lam 0.0 = midpoint, 0.25 = good execution, 0.5 = pay the full half-spread."""
    width_pct: float = 0.03
    min_width: float = 0.01
    late_minutes: int = 30
    late_mult: float = 1.5
    lam: float = 0.25

    def width(self, mid: float, mins_left: float) -> float:
        w = max(self.min_width, self.width_pct * max(mid, 0.0))
        return w * (self.late_mult if mins_left <= self.late_minutes else 1.0)

    def buy(self, mid: float, mins_left: float) -> float:
        return mid + self.lam * self.width(mid, mins_left)

    def sell(self, mid: float, mins_left: float) -> float:
        return max(0.0, mid - self.lam * self.width(mid, mins_left))


def _combine_sources(sources) -> str:
    """One label for a multi-leg price: the weakest leg wins."""
    s = set(sources)
    for label in ("missing", "model", "last_known"):
        if label in s:
            return label
    return "real"


class BSPricer:
    """Theoretical mid from Black-Scholes at the bar's IV. PLUMBING ONLY: it
    prices options with the same assumptions the strategy uses, so profits it
    shows are not evidence of a tradable edge (audit item 3)."""
    name = "black_scholes"
    real = False

    def __init__(self, r: float = 0.0):
        self.r = r

    def quote(self, ts, session, right, strike, S, mins_left, iv, conservative: bool = False):
        """(price, source, staleness_seconds)."""
        return bs_price(S, strike, mins_left, iv, right, self.r), "model", 0.0

    def mid(self, *a, **k) -> Optional[float]:
        return self.quote(*a, **k)[0]

    def listed(self, session, right, strike) -> bool:
        return True


class BarPricer:
    """Real traded prices from historical 1-minute OPTION bars.

    `bars` maps OCC symbol -> DataFrame indexed by bar-start time (NY) with
    at least vwap/close/volume. mid() returns that minute's VWAP; with
    conservative=True it returns min(vwap, close), used when the exact time
    inside the minute is unknown (a stop or target touched intrabar, audit
    item 17). A minute with no trade falls back to the last print within
    `max_stale_minutes`; beyond that the price is unknown (None).
    Trade prints sit between bid and ask, so VWAP approximates the mid; the
    bid/ask width is still modeled by QuoteModel -- historical quotes would
    replace both."""
    name = "option_bars"
    real = True

    def __init__(self, bars: dict, root: str = "SPY", max_stale_minutes: int = 5, min_volume: int = 1):
        self.bars, self.root = bars, root
        self.max_stale = pd.Timedelta(minutes=max_stale_minutes)
        self.min_volume = min_volume

    def listed(self, session, right, strike) -> bool:
        return occ_symbol(self.root, session, right, strike) in self.bars

    def quote(self, ts, session, right, strike, S, mins_left, iv, conservative: bool = False):
        """(price, "real", staleness_seconds), or (None, "missing", nan) when no
        print exists within max_stale_minutes. Never falls back to a model."""
        df = self.bars.get(occ_symbol(self.root, session, right, strike))
        if df is None or df.empty:
            return None, "missing", float("nan")
        i = df.index.searchsorted(ts, side="right") - 1
        if i < 0 or ts - df.index[i] > self.max_stale or df.index[i].normalize() != ts.normalize():
            return None, "missing", float("nan")
        row = df.iloc[i]
        px = row["vwap"] if pd.notna(row.get("vwap")) else row["close"]
        px = float(min(px, row["close"]) if conservative else px)
        return px, "real", (ts - df.index[i]).total_seconds()

    def mid(self, *a, **k) -> Optional[float]:
        return self.quote(*a, **k)[0]


# =============================================================================
# 8. EntryExitEngine: EV decisions on the label-aligned trade
# =============================================================================

@dataclass
class EngineConfig:
    # --- decision -----------------------------------------------------------
    decision_rule: str = "ev"                # "ev" (Phase 1 design) or "threshold" (legacy)
    ev_buffer_usd: float = 5.0               # trade only if after-cost EV per contract exceeds this
    long_conf_thresh: float = 0.55           # threshold rule only
    chop_dir_thresh: float = 0.40            # threshold rule only
    chop_min_p_chop: Optional[float] = 0.45  # threshold rule only
    long_entry_window: tuple = (30, 330)     # minutes after open: 10:00-15:00
    spread_entry_window: tuple = (30, 240)   # 10:00-13:30
    max_trades_per_day: int = 2
    # --- structures ---------------------------------------------------------
    long_delta: float = 0.50
    short_delta: float = 0.30
    wing_delta: float = 0.15
    spread_min_width: float = 2.0            # audit item 16: bound risk per spread
    spread_max_width: float = 5.0
    strike_increment: float = 1.0
    contracts: int = 1
    # --- exits --------------------------------------------------------------
    # "barrier": long options exit exactly where the direction label resolves
    #   (entry +/- b, or H minutes), so the trade IS the bet the model learned.
    # "atr": legacy ATR stop + fixed time exit (not label-aligned).
    long_exit: str = "barrier"
    atr_stop_mult: float = 1.5
    long_max_hold_bars: int = 60
    eod_exit_minutes_before_close: int = 15  # spreads (and everything) flat at 15:45
    spread_loss_stop_mult: Optional[float] = None
    # --- costs --------------------------------------------------------------
    commission_per_contract: float = 0.65
    quote: QuoteModel = field(default_factory=QuoteModel)
    risk_free: float = 0.0
    # --- execution timing / price provenance (review 2, items 1 and 3) -----
    # The signal bar picks the structure; the fill bar re-selects strikes from
    # the fill-bar open, re-prices them, and recomputes EV. Trade only if EV
    # still clears the buffer.
    recheck_ev_at_fill: bool = True
    # No price at exit: "invalid" = trade is flagged, P&L excluded (a worst-case
    # value is reported alongside); "fallback" = legacy last-known -> model chain.
    missing_exit_price: str = "invalid"


@dataclass
class Leg:
    right: str        # "C" or "P"
    strike: float
    qty: int          # +1 long, -1 short


@dataclass
class Position:
    kind: str
    legs: list
    entry_time: pd.Timestamp
    entry_underlying: float
    fills: list                    # per-leg entry fill price per share
    session: pd.Timestamp
    audit: dict
    take_profit: Optional[float] = None
    stop_level: Optional[float] = None
    max_hold: Optional[int] = None
    bars_held: int = 0

    @property
    def entry_value(self) -> float:
        """Signed per-share value paid (+) or received (-)."""
        return sum(lg.qty * f for lg, f in zip(self.legs, self.fills))


class EntryExitEngine:
    """Rule-based execution on top of regime + calibrated probabilities.

    Timing (no look-ahead): a decision is made at bar t's CLOSE using data up
    to t; the chosen contracts fill at bar t+1's OPEN. Exits are checked
    against each bar's own open/high/low/close in time order. Nothing carries
    overnight.

    Decision rule "ev" (Phase 1): for every structure the regime allows,
        long option   EV = sum_k P_k * exit_value_k - entry_cost - commissions
                      scenarios k = up / down / chop from the DIRECTION model,
                      exit at the label's barriers (S0 +/- b) or after H minutes
        credit spread EV = credit_received - sum_k P_k * buyback_k - commissions
                      scenarios from the SPREAD model (inside / breached until 15:45)
    and trade the best one only if EV > ev_buffer_usd. Scenario prices use
    Black-Scholes at an IV implied from the entry price when a real pricer is
    attached (so scenario values stay consistent with the market), else the
    bar's IV.

    Regime gating: Breakout allows long options, Chop allows credit spreads,
    Neutral allows nothing.

    Every trade carries an audit record: decision bar, regime, both models'
    probabilities, EV and its scenario values, selected contracts and their
    model deltas, per-leg mids / modeled widths / fills at entry and exit.

    Integration
        Event-driven (Backtrader-style):  engine.on_bar(ts, row) per closed bar
        Batch:                            trades, bars = engine.run(frame)
        Vectorized overlay:               engine.signal_frame(frame.index)
    """

    REQUIRED = ["open", "high", "low", "close", "session", "tau", "session_len", "atr14",
                "iv_used", "ready", "regime", "p_down", "p_chop", "p_up"]
    SPREAD_PROBS = ["s_down", "s_chop", "s_up"]

    def __init__(self, config: Optional[EngineConfig] = None, pricer=None,
                 direction_label: Optional[LabelConfig] = None, spread_label: Optional[LabelConfig] = None):
        self.cfg = config or EngineConfig()
        self.pricer = pricer or BSPricer(self.cfg.risk_free)
        self.q = self.cfg.quote
        self.dir_label = direction_label or DIRECTION_LABEL
        self.spread_label = spread_label or SPREAD_LABEL
        self.reset()

    def reset(self) -> None:
        self.position: Optional[Position] = None
        self.pending: Optional[dict] = None
        self.trades: list = []
        self.bar_log: list = []
        self.stats = dict(eligible=0, abstain_ev=0, no_quote=0, fill_no_quote=0, fill_abstain_ev=0,
                          no_credit=0, traded=0, exit_missing=0)
        self.realized_usd = 0.0
        self._session = None
        self._trades_today = 0
        self._last_mid: dict = {}

    # ----------------------------------------------------------------- driver
    def run(self, frame: pd.DataFrame):
        need = self.REQUIRED + (self.SPREAD_PROBS if self.cfg.decision_rule == "ev" else [])
        missing = [c for c in need if c not in frame.columns]
        if missing:
            raise ValueError(f"engine frame is missing columns: {missing}")
        self.reset()
        for ts, row in zip(frame.index, frame[need].itertuples(index=False, name="Bar")):
            self.on_bar(ts, row)
        if self.position is not None:
            last = frame.iloc[-1]
            self._close(frame.index[-1], float(last["close"]),
                        int(last["session_len"] - last["tau"] - 1), float(last["iv_used"]), "end_of_data")
        bars = pd.DataFrame(self.bar_log).set_index("time") if self.bar_log else pd.DataFrame()
        return pd.DataFrame(self.trades), bars

    def on_bar(self, ts: pd.Timestamp, bar) -> None:
        cfg = self.cfg
        get = (lambda k: getattr(bar, k)) if not isinstance(bar, (dict, pd.Series)) else (lambda k: bar[k])
        session, tau, session_len = get("session"), int(get("tau")), int(get("session_len"))
        o, h, l, c = float(get("open")), float(get("high")), float(get("low")), float(get("close"))
        iv = float(get("iv_used"))
        eod_tau = session_len - cfg.eod_exit_minutes_before_close
        ml_open, ml_close = session_len - tau, session_len - tau - 1

        if session != self._session:
            self._session, self._trades_today, self._last_mid = session, 0, {}
            self.pending = None
            if self.position is not None:
                raise AssertionError("position carried overnight -- engine invariant violated")

        if self.position is not None and tau >= eod_tau:                  # 1) flat at 15:45
            self._close(ts, o, ml_open, iv, "eod_1545")
        if self.pending is not None:                                       # 2) fill at this open
            if tau < eod_tau and self.position is None:
                self._open(ts, o, ml_open, iv, session, self.pending)
            self.pending = None
        if self.position is not None:                                      # 3) manage
            self._manage(ts, o, h, l, c, ml_open, ml_close, iv)
        value = self._mark(ts, c, ml_close, iv)                            # 4) mark at close
        self.bar_log.append(dict(time=ts, underlying=c,
                                 position=self.position.kind if self.position else None,
                                 position_value=value, realized_usd=self.realized_usd,
                                 equity_usd=self.realized_usd + self._unrealized_usd(value),
                                 theta_per_min_usd=self._theta(c, ml_close, iv)))
        if self.position is None and self.pending is None and tau < eod_tau - 1:
            self.pending = self._decide(ts, get, session, tau, c, ml_close, iv)   # 5) decide at close

    # ----------------------------------------------------------------- decisions
    def _decide(self, ts, get, session, tau, S0, T0, iv) -> Optional[dict]:
        cfg = self.cfg
        if not bool(get("ready")) or self._trades_today >= cfg.max_trades_per_day or not (iv > 0):
            return None
        regime = get("regime")
        allowed = []
        if regime == BREAKOUT and cfg.long_entry_window[0] <= tau < cfg.long_entry_window[1]:
            allowed += ["long_call", "long_put"]
        if regime == CHOP and cfg.spread_entry_window[0] <= tau < cfg.spread_entry_window[1]:
            allowed += ["short_put_spread", "short_call_spread"]
        if not allowed:
            return None
        pdir = tuple(float(get(k)) for k in ("p_down", "p_chop", "p_up"))
        pspr = tuple(float(get(k)) for k in self.SPREAD_PROBS) if cfg.decision_rule == "ev" else pdir
        if not np.all(np.isfinite(pdir)):
            return None
        self.stats["eligible"] += 1
        base = dict(signal_time=ts, signal_close=S0, regime=regime, p_dir=pdir, p_spread=pspr, iv_signal=iv,
                    atr=float(get("atr14")), T0=T0)

        if cfg.decision_rule == "threshold":
            cand = self._threshold_pick(allowed, pdir, session, S0, T0, iv)
            return {**base, **cand} if cand else None

        best = None
        for kind in allowed:
            probs = pdir if kind.startswith("long") else pspr
            if not np.all(np.isfinite(probs)):
                continue
            cand = self._evaluate(kind, ts, session, S0, T0, iv, probs)
            if cand and (best is None or cand["ev_usd"] > best["ev_usd"]):
                best = cand
        if best is None:
            self.stats["no_quote"] += 1
            return None
        if best["ev_usd"] <= cfg.ev_buffer_usd:
            self.stats["abstain_ev"] += 1
            return None
        return {**base, **best}

    def _legs_for(self, kind, session, S0, T0, iv) -> list:
        cfg = self.cfg
        if kind in ("long_call", "long_put"):
            right = "C" if kind == "long_call" else "P"
            return [Leg(right, strike_for_delta(S0, T0, iv, cfg.long_delta, right, cfg.strike_increment, cfg.risk_free), +1)]
        right = "P" if kind == "short_put_spread" else "C"
        k_s = strike_for_delta(S0, T0, iv, cfg.short_delta, right, cfg.strike_increment, cfg.risk_free)
        k_w = strike_for_delta(S0, T0, iv, cfg.wing_delta, right, cfg.strike_increment, cfg.risk_free)
        width = min(max(abs(k_s - k_w), cfg.spread_min_width), cfg.spread_max_width)
        width = max(cfg.strike_increment, round(width / cfg.strike_increment) * cfg.strike_increment)
        k_w = k_s - width if right == "P" else k_s + width
        return [Leg(right, k_s, -1), Leg(right, k_w, +1)]

    def _evaluate(self, kind, ts, session, S0, T0, iv, probs) -> Optional[dict]:
        """After-cost expected value per contract of one structure."""
        cfg, q = self.cfg, self.q
        legs = self._legs_for(kind, session, S0, T0, iv)
        mids = [self.pricer.mid(ts, session, lg.right, lg.strike, S0, T0, iv) for lg in legs]
        if any(m is None or not np.isfinite(m) for m in mids):
            return None
        # Scenario IV: implied from the observed price when real, so the scenario
        # prices start exactly where the market is (audit item 4).
        sig = iv
        if self.pricer.real:
            k0 = 0 if len(legs) == 1 else 0
            iv_mkt = implied_vol(mids[k0], S0, legs[k0].strike, T0, legs[k0].right, cfg.risk_free)
            sig = iv_mkt if np.isfinite(iv_mkt) and iv_mkt > 0 else iv
        comm = cfg.commission_per_contract * len(legs) * 2 / 100.0      # per share, round trip
        p_dn, p_ch, p_up = probs
        bs = lambda lg, S, T: bs_price(S, lg.strike, T, sig, lg.right, cfg.risk_free)

        if kind.startswith("long"):
            lg = legs[0]
            H = int(label_horizon(pd.DataFrame({"mins_left": [T0]}), self.dir_label)[0])
            if H < 1:
                return None
            b = self.dir_label.barrier_k * iv * S0 * math.sqrt(H / MIN_PER_YEAR)
            cost = q.buy(mids[0], T0)
            scen = {"up": bs(lg, S0 + b, T0 - H / 2), "down": bs(lg, S0 - b, T0 - H / 2), "chop": bs(lg, S0, T0 - H)}
            proceeds = {k: q.sell(v, T0 - H) for k, v in scen.items()}
            ev = p_up * proceeds["up"] + p_dn * proceeds["down"] + p_ch * proceeds["chop"] - cost - comm
            comps = dict(entry_cost=cost, horizon=H, barrier=b, **{f"exit_{k}": v for k, v in proceeds.items()})
        else:
            T_exit = cfg.eod_exit_minutes_before_close
            Hs = max(T0 - T_exit, 0)
            if Hs < 1:
                return None
            b = self.spread_label.barrier_k * iv * S0 * math.sqrt(Hs / MIN_PER_YEAR)
            credit = q.sell(mids[0], T0) - q.buy(mids[1], T0)
            if credit <= 0:
                return None

            def buyback(S):
                vs, vw = bs(legs[0], S, T_exit), bs(legs[1], S, T_exit)
                return q.buy(vs, T_exit) - q.sell(vw, T_exit)

            bb = {"up": buyback(S0 + b), "down": buyback(S0 - b), "inside": buyback(S0)}
            ev = credit - (p_up * bb["up"] + p_dn * bb["down"] + p_ch * bb["inside"]) - comm
            comps = dict(credit=credit, barrier=b, **{f"buyback_{k}": v for k, v in bb.items()})

        return dict(kind=kind, legs=legs, ev_usd=ev * 100 * cfg.contracts, ev_components=comps,
                    scenario_iv=sig,
                    model_deltas=[bs_delta(S0, lg.strike, T0, sig, lg.right, cfg.risk_free) for lg in legs])

    def _threshold_pick(self, allowed, pdir, session, S0, T0, iv) -> Optional[dict]:
        cfg = self.cfg
        p_dn, p_ch, p_up = pdir
        conf = max(p_up, p_dn)
        if "long_call" in allowed and conf > cfg.long_conf_thresh:
            kind = "long_call" if p_up > p_dn else "long_put"
        elif "short_put_spread" in allowed and conf < cfg.chop_dir_thresh and \
                (cfg.chop_min_p_chop is None or p_ch >= cfg.chop_min_p_chop):
            kind = "short_put_spread" if p_up >= p_dn else "short_call_spread"
        else:
            return None
        legs = self._legs_for(kind, session, S0, T0, iv)
        return dict(kind=kind, legs=legs, ev_usd=float("nan"), ev_components={}, scenario_iv=iv,
                    model_deltas=[bs_delta(S0, lg.strike, T0, iv, lg.right, cfg.risk_free) for lg in legs])

    # ----------------------------------------------------------------- entries
    def _open(self, ts, S, mins_left, iv, session, d) -> None:
        cfg, q = self.cfg, self.q
        d = {**d, "ev_usd_signal": d["ev_usd"]}
        if cfg.recheck_ev_at_fill and cfg.decision_rule == "ev":
            # t_e = fill-bar open: re-select contracts from S_open, price them at
            # t_e, recompute EV with the signal's probabilities.
            probs = d["p_dir"] if d["kind"].startswith("long") else d["p_spread"]
            fresh = self._evaluate(d["kind"], ts, session, S, mins_left, d["iv_signal"], probs)
            if fresh is None:
                self.stats["fill_no_quote"] += 1
                return
            if fresh["ev_usd"] <= cfg.ev_buffer_usd:
                self.stats["fill_abstain_ev"] += 1
                return
            d.update(fresh)
        legs = d["legs"]
        quotes = [self.pricer.quote(ts, session, lg.right, lg.strike, S, mins_left, iv) for lg in legs]
        mids = [x[0] for x in quotes]
        if any(m is None or not np.isfinite(m) for m in mids):
            self.stats["fill_no_quote"] += 1
            return
        fills = [q.buy(m, mins_left) if lg.qty > 0 else q.sell(m, mins_left) for lg, m in zip(legs, mids)]
        if d["kind"].startswith("short") and -sum(lg.qty * f for lg, f in zip(legs, fills)) <= 0:
            self.stats["no_credit"] += 1
            return
        audit = {k: d[k] for k in ("signal_time", "signal_close", "regime", "p_dir", "p_spread",
                                   "iv_signal", "ev_usd", "ev_usd_signal", "ev_components", "scenario_iv",
                                   "model_deltas")}
        audit.update(entry_mids=mids, entry_widths=[q.width(m, mins_left) for m in mids], pricer=self.pricer.name,
                     lam=q.lam, entry_price_source=_combine_sources(x[1] for x in quotes),
                     entry_staleness_s=max(x[2] for x in quotes))
        pos = Position(d["kind"], legs, ts, S, fills, session, audit)
        if d["kind"].startswith("long"):
            is_call = d["kind"] == "long_call"
            if cfg.long_exit == "barrier":
                H = d["ev_components"].get("horizon") or int(label_horizon(
                    pd.DataFrame({"mins_left": [d["T0"]]}), self.dir_label)[0])
                b = self.dir_label.barrier_k * d["iv_signal"] * S * math.sqrt(max(H, 1) / MIN_PER_YEAR)
                pos.take_profit = S + b if is_call else S - b
                pos.stop_level = S - b if is_call else S + b
                pos.max_hold = int(H)
            else:
                pos.stop_level = S - cfg.atr_stop_mult * d["atr"] if is_call else S + cfg.atr_stop_mult * d["atr"]
                pos.max_hold = cfg.long_max_hold_bars
        self.position = pos
        self._trades_today += 1
        self.stats["traded"] += 1

    # ----------------------------------------------------------------- exits
    def _manage(self, ts, o, h, l, c, ml_open, ml_close, iv) -> None:
        cfg, pos = self.cfg, self.position
        pos.bars_held += 1
        if pos.kind.startswith("long"):
            is_call = pos.kind == "long_call"
            adverse_gap = (o <= pos.stop_level) if is_call else (o >= pos.stop_level)
            if adverse_gap:
                self._close(ts, o, ml_open, iv, "stop_gap")
                return
            tp = pos.take_profit
            if tp is not None and ((o >= tp) if is_call else (o <= tp)):
                self._close(ts, o, ml_open, iv, "take_profit_gap")
                return
            hit_stop = (l <= pos.stop_level) if is_call else (h >= pos.stop_level)
            hit_tp = tp is not None and ((h >= tp) if is_call else (l <= tp))
            if hit_stop:        # both inside one minute: assume the stop came first (conservative)
                self._close(ts, pos.stop_level, ml_close, iv, "stop", conservative=True)
                return
            if hit_tp:
                self._close(ts, tp, ml_close, iv, "take_profit", conservative=True)
                return
            if pos.bars_held >= pos.max_hold:
                self._close(ts, c, ml_close, iv, "time_exit")
                return
        elif cfg.spread_loss_stop_mult is not None:
            credit = -pos.entry_value
            value = self._structure_mid(ts, pos, c, ml_close, iv)
            loss = value - (-credit) if value is not None else None
            # value is the signed mid of the short structure (negative); the
            # per-share loss is entry_value - value (audit item 2, sign fixed).
            if value is not None and (pos.entry_value - value) >= cfg.spread_loss_stop_mult * credit:
                self._close(ts, c, ml_close, iv, "spread_loss_stop")

    def _close(self, ts, S, mins_left, iv, reason, conservative: bool = False) -> None:
        cfg, q, pos = self.cfg, self.q, self.position
        mids, srcs, stale = [], [], []
        for lg in pos.legs:
            m, src, st = self.pricer.quote(ts, pos.session, lg.right, lg.strike, S, mins_left, iv,
                                           conservative=conservative and lg.qty > 0)
            if (m is None or not np.isfinite(m)) and cfg.missing_exit_price == "fallback":
                m, src = self._last_mid.get((lg.right, lg.strike)), "last_known"
                if m is None:
                    m, src = bs_price(S, lg.strike, mins_left, iv, lg.right, cfg.risk_free), "model"
            mids.append(m if m is not None and np.isfinite(m) else float("nan"))
            srcs.append(src)
            stale.append(st)
        invalid = not np.all(np.isfinite(mids))
        commissions = cfg.commission_per_contract * len(pos.legs) * 2 * cfg.contracts
        if invalid:
            # No real exit price: P&L is unknown. Report a worst case instead of
            # inventing one (long option -> 0; credit spread -> full width).
            self.stats["exit_missing"] += 1
            exit_fills = [float("nan")] * len(pos.legs)
            pnl_usd = float("nan")
            if pos.kind.startswith("long"):
                worst_share = -pos.fills[0]
            else:
                width = abs(pos.legs[0].strike - pos.legs[1].strike)
                worst_share = -pos.entry_value - width
            pnl_worst = worst_share * 100 * cfg.contracts - commissions
        else:
            exit_fills = [q.sell(m, mins_left) if lg.qty > 0 else q.buy(m, mins_left) for lg, m in zip(pos.legs, mids)]
            pnl_share = sum(lg.qty * (xf - ef) for lg, xf, ef in zip(pos.legs, exit_fills, pos.fills))
            pnl_usd = pnl_share * 100 * cfg.contracts - commissions
            pnl_worst = pnl_usd
            self.realized_usd += pnl_usd
        a = pos.audit
        self.trades.append(dict(
            kind=pos.kind, regime=a["regime"], session=pos.session,
            signal_time=a["signal_time"], signal_close=a["signal_close"],
            entry_time=pos.entry_time, exit_time=ts, bars_held=pos.bars_held, max_hold=pos.max_hold,
            entry_underlying=pos.entry_underlying, exit_underlying=S,
            strikes="/".join(f"{lg.strike:g}{lg.right}{'+' if lg.qty > 0 else '-'}" for lg in pos.legs),
            model_deltas=[round(x, 3) for x in a["model_deltas"]],
            p_dir=[round(x, 3) for x in a["p_dir"]], p_spread=[round(x, 3) for x in a["p_spread"]],
            ev_usd_signal=a.get("ev_usd_signal", a["ev_usd"]), ev_usd=a["ev_usd"],
            ev_components={k: round(v, 4) for k, v in a["ev_components"].items()},
            scenario_iv=a["scenario_iv"], iv_signal=a["iv_signal"],
            entry_mids=[round(x, 4) for x in a["entry_mids"]], entry_fills=[round(x, 4) for x in pos.fills],
            exit_mids=[round(x, 4) for x in mids], exit_fills=[round(x, 4) for x in exit_fills],
            entry_price_source=a.get("entry_price_source", "model"), exit_price_source=_combine_sources(srcs),
            entry_staleness_s=a.get("entry_staleness_s", 0.0), exit_staleness_s=float(np.nanmax(stale)) if
            np.isfinite(stale).any() else float("nan"),
            pricer=a["pricer"], lam=a["lam"], commissions_usd=commissions,
            invalid=invalid, pnl_usd=pnl_usd, pnl_usd_worst=pnl_worst, reason=reason))
        self.position = None

    # ----------------------------------------------------------------- marking
    def _structure_mid(self, ts, pos, S, mins_left, iv) -> Optional[float]:
        total = 0.0
        for lg in pos.legs:
            m = self.pricer.mid(ts, pos.session, lg.right, lg.strike, S, mins_left, iv)
            if m is None or not np.isfinite(m):
                m = self._last_mid.get((lg.right, lg.strike))
            if m is None:
                return None
            self._last_mid[(lg.right, lg.strike)] = m
            total += lg.qty * m
        return total

    def _mark(self, ts, S, mins_left, iv) -> float:
        pos = self.position
        if pos is None:
            return 0.0
        v = self._structure_mid(ts, pos, S, mins_left, iv)
        return float("nan") if v is None else v

    def _unrealized_usd(self, value: float) -> float:
        pos = self.position
        if pos is None or not np.isfinite(value):
            return 0.0
        return (value - pos.entry_value) * 100 * self.cfg.contracts

    def _theta(self, S, mins_left, iv) -> float:
        pos = self.position
        if pos is None or not (iv > 0):
            return 0.0
        return sum(lg.qty * bs_theta_per_minute(S, lg.strike, mins_left, iv, lg.right, self.cfg.risk_free)
                   for lg in pos.legs) * 100 * self.cfg.contracts

    # ----------------------------------------------------------------- export
    def signal_frame(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Boolean entry/exit columns per structure, stamped on the FILL bars."""
        kinds = ["long_call", "long_put", "short_put_spread", "short_call_spread"]
        sig = pd.DataFrame(False, index=index, columns=[f"{k}_entry" for k in kinds] + [f"{k}_exit" for k in kinds])
        for t in self.trades:
            if t["entry_time"] in sig.index:
                sig.loc[t["entry_time"], f"{t['kind']}_entry"] = True
            if t["exit_time"] in sig.index:
                sig.loc[t["exit_time"], f"{t['kind']}_exit"] = True
        return sig


# =============================================================================
# 9. Look-ahead guard
# =============================================================================

def assert_no_lookahead(bars: pd.DataFrame, builder: FeatureBuilder,
                        classifier: Optional[RegimeClassifier] = None,
                        n_checks: int = 8, seed: int = 7, rtol: float = 1e-9, atol: float = 1e-9) -> None:
    """Truncation test. For random cut times t: recompute everything on data
    ending at t and require the value stamped on bar t to equal the value from
    the full-history run. Any centred window, back-fill, whole-session aggregate
    broadcast backwards, or data-derived session length fails this test."""
    full = builder.transform(bars)
    full_reg = classifier.classify(full) if classifier else None
    rng = np.random.default_rng(seed)
    start = len(bars) // 2
    cuts = sorted(rng.choice(np.arange(start, len(bars)), size=min(n_checks, len(bars) - start), replace=False))
    cols = FeatureBuilder.MODEL_FEATURES + ["atr14", "vwap", "iv_used"]
    for pos in cuts:
        ts = bars.index[pos]
        part = builder.transform(bars.iloc[: pos + 1])
        a = full.loc[ts, cols].astype(float).to_numpy()
        b = part.loc[ts, cols].astype(float).to_numpy()
        same = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if not same.all():
            bad = [c for c, ok in zip(cols, same) if not ok]
            raise AssertionError(f"look-ahead in features at {ts}: {bad}")
        if classifier:
            part_reg = classifier.classify(part)
            if full_reg.loc[ts, "regime"] != part_reg.loc[ts, "regime"]:
                raise AssertionError(f"look-ahead in regime at {ts}: "
                                     f"{full_reg.loc[ts, 'regime']} vs {part_reg.loc[ts, 'regime']}")
    log.info("look-ahead check passed on %d cut points", len(cuts))


# =============================================================================
# 10. Convenience: full pipeline
# =============================================================================

def build_pipeline(raw: pd.DataFrame,
                   calendar: Optional[CalendarConfig] = None,
                   feature_cfg: Optional[FeatureConfig] = None,
                   regime_cfg: Optional[RegimeConfig] = None,
                   direction_cfg: Optional[ModelConfig] = None,
                   spread_cfg: Optional[ModelConfig] = None) -> tuple:
    """raw bars -> (frame, direction_model, spread_model). The frame carries
    features, regime, direction probabilities (p_*) and spread probabilities
    (s_*): exactly what EntryExitEngine.run expects."""
    bars = prepare_bars(raw, calendar)
    feats = FeatureBuilder(feature_cfg).transform(bars)
    regimes = RegimeClassifier(regime_cfg).classify(feats)
    dmod = WalkForwardModel(config=direction_cfg or ModelConfig(label=DIRECTION_LABEL))
    smod = WalkForwardModel(config=spread_cfg or ModelConfig(label=SPREAD_LABEL))
    frame = pd.concat([feats, regimes, dmod.fit_predict(feats, "p_"), smod.fit_predict(feats, "s_")], axis=1)
    return frame, dmod, smod


# =============================================================================
# 11. Synthetic demo (plumbing check only -- synthetic data has no real edge)
# =============================================================================

def synthetic_spy(n_sessions: int = 260, seed: int = 11) -> pd.DataFrame:
    """SPY-like 1-min bars with U-shaped intraday vol and volume, occasional
    trend days, and a noisy implied-vol and VIX series. For testing only."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", periods=n_sessions)
    tau = np.arange(REGULAR_SESSION_BARS)
    u_shape = 1.0 + 1.6 * np.exp(-tau / 25.0) + 0.6 * np.exp(-(389 - tau) / 30.0)
    frames, price, vix = [], 480.0, 15.0
    for d in days:
        vix = float(np.clip(vix + rng.normal(0, 0.8), 10, 40))
        daily_sigma = vix / 100 / math.sqrt(TRADING_DAYS)
        per_min = daily_sigma / math.sqrt(REGULAR_SESSION_BARS) * u_shape / u_shape.mean()
        drift = rng.choice([0.0, 0.0, 0.0, 1.0, -1.0]) * daily_sigma / REGULAR_SESSION_BARS * 0.8
        price *= math.exp(rng.normal(0, daily_sigma * 0.4))                 # overnight gap
        rets = drift + rng.normal(0, 1, REGULAR_SESSION_BARS) * per_min
        close = price * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[price], close[:-1]])
        wick = np.abs(rng.normal(0, 1, REGULAR_SESSION_BARS)) * per_min * close * 0.6
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - wick
        vol = (2e5 * u_shape * rng.lognormal(0, 0.3, REGULAR_SESSION_BARS)).astype(int)
        iv = np.full(REGULAR_SESSION_BARS, vix / 100 * rng.uniform(0.85, 1.2))
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=REGULAR_SESSION_BARS, freq="1min", tz=NY)
        frames.append(pd.DataFrame(dict(open=open_, high=high, low=low, close=close, volume=vol,
                                        iv=iv, vix=vix, vix9d=vix * rng.uniform(0.88, 1.08)), index=idx))
        price = close[-1]
    return pd.concat(frames)


LONG_EXIT_REASONS = {"take_profit", "take_profit_gap", "stop", "stop_gap", "time_exit", "eod_1545", "end_of_data"}


def check_engine_invariants(trades: pd.DataFrame, frame: pd.DataFrame, cfg: EngineConfig) -> list:
    """Structural checks on a finished run. Returns a list of failure messages."""
    fails = []
    if trades.empty:
        return ["no trades to check"]
    tau = frame["tau"]
    if "entry_price_source" in trades and (trades["pricer"] != "black_scholes").all():
        valid = trades[~trades["invalid"]]
        if not (valid["entry_price_source"] == "real").all():
            fails.append("real-price run has an entry priced from a non-real source")
        if not (valid["exit_price_source"] == "real").all():
            fails.append("real-price run has a valid exit priced from a non-real source")
        if trades.loc[trades["invalid"], "pnl_usd"].notna().any():
            fails.append("invalid trade carries a P&L")
    for t in trades.itertuples():
        if t.session != frame.loc[t.exit_time, "session"]:
            fails.append(f"{t.entry_time}: held overnight")
        if not np.isclose(t.entry_underlying, frame.loc[t.entry_time, "open"]):
            fails.append(f"{t.entry_time}: entry not at the bar open")
        prev = frame.index.get_loc(t.entry_time) - 1
        if prev < 0 or frame.index[prev] != t.signal_time:
            fails.append(f"{t.entry_time}: fill is not the bar right after its decision bar")
        eod = frame.loc[t.exit_time, "session_len"] - cfg.eod_exit_minutes_before_close
        if t.kind.startswith("short") and cfg.spread_loss_stop_mult is None:
            if t.reason != "eod_1545" or tau[t.exit_time] != eod:
                fails.append(f"{t.entry_time}: spread exited by {t.reason} at tau {tau[t.exit_time]}, expected 15:45")
        if t.kind.startswith("long"):
            if t.reason not in LONG_EXIT_REASONS:
                fails.append(f"{t.entry_time}: unexpected long exit {t.reason}")
            if t.max_hold is not None and t.bars_held > t.max_hold:
                fails.append(f"{t.entry_time}: long held {t.bars_held} bars > limit {t.max_hold}")
        if tau[t.exit_time] > eod:
            fails.append(f"{t.entry_time}: exited after 15:45")
    return fails


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw = synthetic_spy()
    bars = prepare_bars(raw)
    log.info("running look-ahead truncation test ...")
    assert_no_lookahead(bars.iloc[: 60 * REGULAR_SESSION_BARS], FeatureBuilder(), RegimeClassifier(), n_checks=6)

    loose_regime = RegimeConfig(ratr_breakout=1.0, ratr_chop=1.05, vrp_breakout=0.9, vrp_chop=1.1,
                                ts_breakout=0.95, ts_chop=1.0, vix_chop_low=10, vix_chop_high=40,
                                xvwap_breakout_rate=4, xvwap_chop_rate=2)
    small = dict(min_train_sessions=80, min_calib_per_class=20)
    frame, dmod, smod = build_pipeline(raw, regime_cfg=loose_regime,
                                       direction_cfg=ModelConfig(label=DIRECTION_LABEL, **small),
                                       spread_cfg=ModelConfig(label=SPREAD_LABEL, **small))
    print(f"\nfolds: direction {len(dmod.fold_log)}, spread {len(smod.fold_log)} "
          f"(trees used, last fold: {dmod.fold_log[-1]['trees'] if dmod.fold_log else '-'})")
    print("direction model scorecard (synthetic, plumbing only):")
    print(evaluate_forecasts(frame, dmod.labels, dmod.variants, dmod.step()).round(4).to_string())
    print("\nregime at 10:00 (RELAXED thresholds, plumbing only):",
          frame.loc[frame["tau"] == 29, "regime"].value_counts().to_dict())
    eng = EntryExitEngine(EngineConfig(ev_buffer_usd=-1e9))       # accept any EV: exercise every code path
    trades, _ = eng.run(frame)
    print("engine stats:", eng.stats)
    if len(trades):
        print(trades.groupby("kind").agg(n=("pnl_usd", "size"), avg_pnl=("pnl_usd", "mean"),
                                         avg_ev=("ev_usd", "mean")).round(2).to_string())
        print("exit reasons:", trades["reason"].value_counts().to_dict())
    fails = check_engine_invariants(trades, frame, eng.cfg)
    print("engine invariants:", "ALL PASS" if not fails else f"{len(fails)} FAIL -> {fails[:5]}")
    print("\nSynthetic data: this checks the plumbing, not profitability.")
