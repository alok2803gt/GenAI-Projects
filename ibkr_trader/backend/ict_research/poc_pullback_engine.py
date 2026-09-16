"""
Full Accumulation -> Consolidation -> Manipulation -> Pullback-to-POC ->
Confirmation -> Entry framework. Extends sweep_engine.py's sweep/reversal
detection (reused directly via compute_levels/the same sweep+confirm state
machine, not reimplemented) with a POC-pullback entry trigger -- testing
whether waiting for a retrace to the volume-profile Point of Control before
entering, instead of entering immediately on reversal confirmation the way
sweep_engine.py does, changes the (negative) result already found there.

Methodology (every approximation stated, same discipline as sweep_engine.py):

- RTH only (9:30-16:00 ET), same reasoning as sweep_engine.py.
- Accumulation/Consolidation: treated as ONE combined phase here, not
  separately modeled -- a rolling LOOKBACK-minute range where
  (rolling_high - rolling_low) / atr_proxy stays under RANGE_TIGHTNESS_MULT.
  This range is what the POC volume profile is built from. Reduces the
  6-step framework to 5 testable stages (accumulation+consolidation -> POC
  -> manipulation -> pullback/confirmation -> entry) since the two aren't
  given distinct operational definitions to separate them further.
- POC (Point of Control): within the SAME lookback window used for the
  swing high/low (so it's the identical range being swept), bucket volume
  into N_BINS price bins and take the bin with the most cumulative volume.
  Recomputed per-bar (rolling), not looked up once.
- Manipulation (sweep) + initial reversal: IDENTICAL mechanism to
  sweep_engine.py's compute_levels/simulate state machine (swing break by
  a volatility-scaled buffer, then close back inside within CONFIRM_WINDOW
  bars). Reused via direct import, not reimplemented -- avoids silently
  drifting from the already-validated (as a detection mechanism) sweep
  logic.
- Pullback to POC: NEW stage. After the initial reversal confirms, wait
  up to PULLBACK_WINDOW bars for price to retrace to within
  PULLBACK_TOLERANCE (fraction of the swing range) of the POC level
  computed at sweep time.
- Confirmation: NEW stage. At/near POC, require a rejection bar -- a
  close that moves AWAY from POC in the trade direction (not just a touch)
  -- before entering. A touch-with-no-rejection expires the setup.
- Entry: on confirmation, in the direction of the ORIGINAL reversal
  (opposite the sweep) -- same directional logic as sweep_engine.py.
- Exit: target = TARGET_FRACTION of the original swing range from entry;
  stop = price re-breaking back through the sweep's own extreme (the
  whole thesis -- sweep, reversal, AND pullback-hold -- failing); max
  hold; forced flat at session close. Same conventions as sweep_engine.py
  for direct comparability.
- Cost: 6bps round-trip, same assumption as every other strategy tested
  this session.
- Fair Value Gap (FVG), optional (require_fvg=True): a real ICT concept
  NOT in the original version of this file -- added on request to test
  whether requiring genuine FVG confluence changes the (negative) result.
  Standard 3-candle definition: bullish FVG when bar[i]'s low sits above
  bar[i-2]'s high (an untraded gap left behind by the middle bar's
  displacement); bearish FVG is the mirror image. Scanned bar-by-bar
  across the sweep -> pullback path, same-day only, in the reversal's
  direction; the first one found is remembered as confluence for that
  setup. When require_fvg=True, entry additionally requires an FVG was
  found somewhere along the path -- existence, not that price is inside
  it at the entry bar (the POC-rejection gate already answers "where").
  Default False, so every existing PocConfig() call -- including the
  already-recorded poc_results.json run -- is bit-for-bit unaffected.
- Bullish trend filter, optional (require_bullish_trend=True): requested on
  top of the FVG result to test whether taking only LONG reversals (the
  down-sweep -> bounce case) with the broader trend, instead of every
  reversal regardless of context, changes anything. Deliberately
  ASYMMETRIC per the request -- only gates LONG setups; SHORT (up-sweep)
  setups are untouched, there is no mirrored bearish-trend requirement.
  "Trend" = yesterday's daily close > yesterday's 50-day SMA of daily
  closes (both computed through yesterday only, no same-day lookahead --
  the trend context has to already exist before today's sweep, not be
  decided by today's own price action). Gates at setup INITIATION (a
  down-sweep is never even watched if the trend isn't bullish), not at
  entry, so a trend flip mid-setup can't retroactively validate a bad
  setup. Default False, existing configs unaffected.
- Bullish market STRUCTURE filter, optional (require_bullish_structure=
  True): a second, more ICT-native alternative to the SMA-based trend
  filter above -- requested because an SMA crossover isn't how ICT
  actually defines trend. Real market structure: a standard N-day fractal
  (a day's high/low that's the local extreme across N days on both sides)
  identifies daily swing highs/lows; bullish structure = the two most
  recently CONFIRMED swing highs are ascending AND the two most recently
  confirmed swing lows are ascending (both required, matching "higher
  highs AND higher lows"). A swing isn't knowable until N days after it
  forms (need the following bars to know it was a local extreme), and the
  resulting flag is shifted one further day, so -- same discipline as the
  SMA filter -- only swing structure fully confirmed through yesterday
  gates today's setups. Noticeably stricter than the SMA filter (real
  check on SPY: ~19% of days qualify vs ~51% for SMA), since an actual
  ascending swing sequence is a higher bar than merely being above an
  average. Also LONG-only, same as require_bullish_trend; can be combined
  with it or tested alone. Default False, existing configs unaffected.
- Killzone session filter, optional (require_killzone=True): real ICT
  practice restricts entries to narrow session windows (London open, NY
  open, etc.), not any RTH hour -- flagged as a real gap, since every test
  above let a sweep fire at any time 9:30-16:00 ET. Gates setup
  INITIATION for BOTH UP_SWEEP and DOWN_SWEEP (unlike the trend/structure
  filters, a killzone isn't a directional concept). Windows passed as a
  list of (start, end) "HH:MM" strings via apply_killzone_gate(); the
  standard test here uses NY AM (9:30-11:00 ET) and NY PM (13:30-16:00
  ET), the two RTH-hours ICT sessions.
- Prior-day/week liquidity targets, optional, via apply_prior_day_liquidity
  (NOT a PocConfig field -- a data-prep substitution, like choosing
  lookback=20 vs 60): flagged as a real gap alongside the killzone filter
  -- every test above swept a rolling N-minute intraday lookback extreme,
  not the specific pools a practitioner actually targets (prior day
  high/low, prior week high/low, equal highs/lows). compute_prior_day_
  levels() computes each session's PDH/PDL from the prior COMPLETED
  session only (no lookahead -- the first session in any dataset has no
  prior day and is correctly left NaN, excluded same as any other
  insufficient-history case). apply_prior_day_liquidity() then overrides
  the swing_high/swing_low columns compute_levels() already produces with
  PDH/PDL -- everything downstream (buffer sweep detection, POC/pullback,
  target/stop sizing) is unchanged, since it already just reads
  swing_high/swing_low generically. Pure substitution of WHAT gets swept,
  not a new mechanism; the rolling-lookback consolidation gate and POC
  computation (the LOCAL price action right before the sweep) are
  untouched -- those are a different concept from which liquidity pool is
  being targeted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

ROUND_TRIP_COST_PCT = 0.0006
ATR_WINDOW = 20
N_BINS = 20  # price buckets for the volume profile within each lookback window


@dataclass
class PocConfig:
    lookback: int             # minutes, swing high/low + POC window
    buffer_mult: float         # x ATR proxy, sweep detection
    confirm_window: int        # bars to wait for initial reversal confirmation
    pullback_window: int       # bars to wait for pullback to POC after reversal confirms
    pullback_tolerance: float  # fraction of swing range counted as "at POC"
    target_fraction: float     # fraction of swing range as profit target
    require_fvg: bool = False  # require a same-direction Fair Value Gap along the
                                # sweep->pullback path before allowing entry (see module docstring)
    require_bullish_trend: bool = False  # gate LONG (down-sweep) setups on daily trend (see module docstring)
    require_bullish_structure: bool = False  # gate LONG setups on real HH/HL swing structure (see module docstring)
    require_killzone: bool = False  # gate ALL setups (both directions) to ICT session windows (see module docstring)


@dataclass
class PocTrade:
    ticker: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    poc_price: float
    exit_reason: str
    raw_ret_pct: float
    net_ret_pct: float
    regime: str = ""


def _rth_only(df: pd.DataFrame) -> pd.DataFrame:
    idx_time = df.index.time
    mask = (idx_time >= pd.Timestamp("09:30").time()) & (idx_time < pd.Timestamp("16:00").time())
    return df[mask]


def _poc_price(highs: np.ndarray, lows: np.ndarray, vols: np.ndarray, n_bins: int) -> float | None:
    """Volume profile POC for one lookback window: bucket each bar's volume
    into price bins spanning [min(low), max(high)], return the bin midpoint
    with the most cumulative volume. Returns None if the window is degenerate
    (flat range, no volume)."""
    lo, hi = lows.min(), highs.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or vols.sum() <= 0:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_vol = np.zeros(n_bins)
    # Each bar's volume is credited to the bin containing its (h+l)/2 typical
    # price -- a coarse but standard simplification (no intrabar tick data
    # to split volume across bins more precisely).
    typical = (highs + lows) / 2.0
    bin_idx = np.clip(np.digitize(typical, edges) - 1, 0, n_bins - 1)
    for idx, v in zip(bin_idx, vols):
        bin_vol[idx] += v
    poc_bin = int(np.argmax(bin_vol))
    return float((edges[poc_bin] + edges[poc_bin + 1]) / 2.0)


def compute_levels(df: pd.DataFrame, lookback: int, rth_only: bool = True) -> pd.DataFrame:
    """Adds swing_high/swing_low/atr (same as sweep_engine.py) plus poc and
    range_over_atr to each bar, computed per-session (no cross-day lookback).
    Does NOT apply the consolidation threshold -- that's a cheap comparison
    against range_over_atr, kept separate (see apply_consolidation_gate) so
    the expensive per-bar POC/tightness computation runs once per (ticker,
    lookback) and every range_tightness_mult in a grid sweep is free.

    rth_only=False (for near-24h instruments like futures): skips the RTH
    filter and groups bars into a real trading SESSION (18:00 ET one day
    -> 17:00 ET the next, the standard CME settlement-to-settlement day)
    instead of equity market hours. Grouping key is each bar's timestamp
    minus 18h, then its date -- a bar at 18:00 ET Mon and a bar at 16:59 ET
    Tue (just before the next session opens) both land in "Monday's"
    session bucket. The resulting session_date column is what simulate()
    uses for day-boundary detection (EOD-flat, state resets) instead of
    raw calendar date, so a session spanning midnight doesn't get
    incorrectly treated as two separate days mid-session. Default True
    (plain calendar date, identical to the original behavior), so every
    existing call site -- every equity test in this file -- is
    bit-for-bit unaffected."""
    if rth_only:
        df = _rth_only(df).copy()
        group_key = df.index.date
    else:
        df = df.copy()
        group_key = (df.index - pd.Timedelta(hours=18)).date
    out_frames = []
    for key, day_df in df.groupby(group_key):
        d = day_df.copy()
        d["session_date"] = key
        d["swing_high"] = d["high"].rolling(lookback, min_periods=5).max().shift(1)
        d["swing_low"] = d["low"].rolling(lookback, min_periods=5).min().shift(1)
        d["atr"] = (d["high"] - d["low"]).rolling(ATR_WINDOW, min_periods=5).mean()

        poc_vals = [np.nan] * len(d)
        tightness = [np.nan] * len(d)
        h, l, v = d["high"].to_numpy(), d["low"].to_numpy(), d["volume"].to_numpy()
        for i in range(len(d)):
            if i < lookback:
                continue
            window_h = h[i - lookback:i]
            window_l = l[i - lookback:i]
            window_v = v[i - lookback:i]
            atr_i = d["atr"].iloc[i]
            if pd.isna(atr_i) or atr_i <= 0:
                continue
            rng = window_h.max() - window_l.min()
            tightness[i] = rng / atr_i
            poc = _poc_price(window_h, window_l, window_v, N_BINS)
            if poc is not None:
                poc_vals[i] = poc
        d["poc"] = poc_vals
        d["range_over_atr"] = tightness
        out_frames.append(d)
    return pd.concat(out_frames).sort_index()


def apply_consolidation_gate(df_lvl: pd.DataFrame, range_tightness_mult: float) -> pd.DataFrame:
    """Cheap: sets is_consolidating from the already-computed range_over_atr
    column. Returns a shallow copy (does not mutate the shared precomputed
    frame passed into a grid sweep)."""
    out = df_lvl.copy()
    out["is_consolidating"] = out["range_over_atr"] <= range_tightness_mult
    return out


def compute_daily_trend(df_1m: pd.DataFrame, sma_period: int = 50) -> pd.Series:
    """One row per session: True if that day's trend context is bullish --
    the PRIOR day's close was above the PRIOR day's SMA(sma_period) of daily
    closes. Both sides of the comparison use data through yesterday only, so
    "today is bullish" is a fact established before today's session opens,
    never decided by today's own price action. Returned Series is indexed
    by date (not timestamp), for a per-bar merge via apply_trend_gate."""
    daily_close = df_1m["close"].resample("1D").last().dropna()
    sma = daily_close.rolling(sma_period).mean()
    bullish_asof_yesterday = (daily_close.shift(1) > sma.shift(1))
    bullish_asof_yesterday.index = bullish_asof_yesterday.index.date
    return bullish_asof_yesterday


def apply_trend_gate(df_lvl: pd.DataFrame, daily_trend: pd.Series) -> pd.DataFrame:
    """Cheap: merges the precomputed per-day bullish flag onto each bar via
    its date. Returns a shallow copy, same pattern as apply_consolidation_gate."""
    out = df_lvl.copy()
    dates = out.index.date
    out["is_bullish_trend"] = pd.Series(dates, index=out.index).map(daily_trend).fillna(False)
    return out


def compute_swing_structure(df_1m: pd.DataFrame, fractal_n: int = 2) -> pd.Series:
    """One row per session: True if real market structure is bullish -- the
    two most recently CONFIRMED daily swing highs are ascending AND the two
    most recently confirmed daily swing lows are ascending, using only
    structure confirmed strictly before today. A daily swing high/low uses
    the standard N-day fractal (the local extreme across fractal_n days on
    both sides); a swing at day i isn't knowable until day i+fractal_n, once
    the following bars exist to confirm it was a real local extreme. Returns
    a boolean Series indexed by date, already shifted so "bullish on day d"
    reflects structure confirmed through d-1 only -- ready to gate day d's
    trading with no lookahead."""
    daily = df_1m.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    h, l = daily["high"], daily["low"]
    window = 2 * fractal_n + 1
    is_swing_high = (h == h.rolling(window, center=True).max())
    is_swing_low = (l == l.rolling(window, center=True).min())
    # Confirmed value becomes visible fractal_n days after the swing bar itself.
    sh_val = h.where(is_swing_high).shift(fractal_n)
    sl_val = l.where(is_swing_low).shift(fractal_n)

    bullish = pd.Series(False, index=daily.index)
    highs_seen: list[float] = []
    lows_seen: list[float] = []
    for i in range(len(daily)):
        v = sh_val.iloc[i]
        if pd.notna(v):
            highs_seen.append(float(v))
        v2 = sl_val.iloc[i]
        if pd.notna(v2):
            lows_seen.append(float(v2))
        if len(highs_seen) >= 2 and len(lows_seen) >= 2:
            bullish.iloc[i] = (highs_seen[-1] > highs_seen[-2]) and (lows_seen[-1] > lows_seen[-2])

    bullish = bullish.shift(1).fillna(False)  # one more day: today's gate uses only data through yesterday
    bullish.index = bullish.index.date
    return bullish


def apply_structure_gate(df_lvl: pd.DataFrame, daily_structure: pd.Series) -> pd.DataFrame:
    """Cheap: merges the precomputed per-day bullish-structure flag onto
    each bar via its date. Same pattern as apply_trend_gate."""
    out = df_lvl.copy()
    dates = out.index.date
    out["is_bullish_structure"] = pd.Series(dates, index=out.index).map(daily_structure).fillna(False)
    return out


def compute_prior_day_levels(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Real ICT liquidity targets, not a rolling intraday lookback: each
    session's Prior Day High (PDH) and Prior Day Low (PDL), computed from
    the prior COMPLETED RTH session only. Broadcast to every bar of the
    current session -- no lookahead, since a "prior" day is by definition
    already finished. Returns the input frame (RTH-filtered) with pdh/pdl
    columns added; the first session in the data has no prior day and is
    left NaN, same as any other insufficient-history case elsewhere in
    this file."""
    df = _rth_only(df_1m).copy()
    daily_hi = df.groupby(df.index.date)["high"].max()
    daily_lo = df.groupby(df.index.date)["low"].min()
    dates = sorted(daily_hi.index)
    pdh_map = {dates[i]: daily_hi[dates[i - 1]] for i in range(1, len(dates))}
    pdl_map = {dates[i]: daily_lo[dates[i - 1]] for i in range(1, len(dates))}
    bar_dates = df.index.date
    df["pdh"] = pd.Series(bar_dates, index=df.index).map(pdh_map)
    df["pdl"] = pd.Series(bar_dates, index=df.index).map(pdl_map)
    return df


def apply_prior_day_liquidity(df_lvl: pd.DataFrame, df_prior: pd.DataFrame) -> pd.DataFrame:
    """Overrides compute_levels()'s rolling-lookback swing_high/swing_low
    with real PDH/PDL -- see module docstring. Pure substitution: every
    downstream consumer (sweep-buffer check, POC/pullback, target/stop
    sizing) already just reads swing_high/swing_low generically, so
    nothing else needs to change. The consolidation gate and POC itself
    (computed by compute_levels/apply_consolidation_gate) are untouched --
    those describe the local price action before the sweep, a different
    concept from which liquidity pool is being targeted."""
    out = df_lvl.copy()
    out["swing_high"] = df_prior["pdh"].reindex(out.index)
    out["swing_low"] = df_prior["pdl"].reindex(out.index)
    return out


def apply_killzone_gate(df_lvl: pd.DataFrame, windows: list[tuple[str, str]]) -> pd.DataFrame:
    """Restricts sweep-setup INITIATION to real ICT session windows (e.g.
    NY AM 09:30-11:00 ET, NY PM 13:30-16:00 ET), each given as ('HH:MM',
    'HH:MM'). Real ICT practice treats session opens and specific hourly
    blocks as where liquidity sweeps are actually considered high-
    probability -- every prior test in this file let a sweep fire at any
    RTH hour. Applies to BOTH directions (not LONG-only like the trend/
    structure filters -- a killzone isn't a directional concept)."""
    out = df_lvl.copy()
    t = out.index.time
    mask = np.zeros(len(out), dtype=bool)
    for lo, hi in windows:
        lo_t = pd.Timestamp(lo).time()
        hi_t = pd.Timestamp(hi).time()
        mask |= (t >= lo_t) & (t < hi_t)
    out["in_killzone"] = mask
    return out


def simulate(df_lvl: pd.DataFrame, ticker: str, config: PocConfig) -> list[PocTrade]:
    trades: list[PocTrade] = []
    position = None
    entry_time = entry_price = stop_level = target_level = poc_at_entry = None
    hold_bars = 0

    # NONE -> WAITING_CONFIRM (post-sweep) -> WAITING_PULLBACK (post-reversal,
    # waiting for POC retrace) -> [entry on POC-rejection confirmation]
    state = "NONE"
    pending_dir = None            # "UP_SWEEP" | "DOWN_SWEEP"
    pending_extreme = None
    pending_swing_high = pending_swing_low = pending_poc = None
    pending_bars_left = 0
    pullback_bars_left = 0
    touched_poc_zone = False      # must touch POC before a rejection bar counts as confirmation
    pending_fvg = None            # first same-direction FVG found along the sweep->pullback path

    # session_date (set by compute_levels) instead of raw calendar date --
    # for rth_only=False data (futures), a real trading session spans
    # midnight, so using the raw calendar date here would incorrectly
    # treat one session as two separate days mid-session (spurious
    # EOD-flat exit, spurious state reset). Falls back to raw calendar
    # date if the column is missing (shouldn't happen via compute_levels,
    # but keeps this function usable standalone).
    dates = df_lvl["session_date"].to_numpy() if "session_date" in df_lvl.columns else df_lvl.index.date
    n = len(df_lvl)
    highs_arr = df_lvl["high"].to_numpy()
    lows_arr = df_lvl["low"].to_numpy()
    for i in range(n):
        t = df_lvl.index[i]
        row = df_lvl.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        sh, sl, atr = row["swing_high"], row["swing_low"], row["atr"]
        is_consolidating = bool(row.get("is_consolidating", False))
        is_last_bar_of_day = (i == n - 1) or (dates[i] != dates[i + 1])

        if is_last_bar_of_day:
            state, pending_dir, touched_poc_zone, pending_fvg = "NONE", None, False, None

        # FVG scan: while a sweep is pending (either waiting for the initial
        # reversal or waiting for the POC pullback), look for the first
        # same-direction 3-bar Fair Value Gap along the path. Same-day only.
        if config.require_fvg and pending_dir is not None and pending_fvg is None and i >= 2 and dates[i] == dates[i - 2]:
            if pending_dir == "DOWN_SWEEP" and lows_arr[i] > highs_arr[i - 2]:
                pending_fvg = (highs_arr[i - 2], lows_arr[i])          # bullish FVG
            elif pending_dir == "UP_SWEEP" and highs_arr[i] < lows_arr[i - 2]:
                pending_fvg = (highs_arr[i], lows_arr[i - 2])          # bearish FVG

        if position is not None:
            hold_bars += 1
            exit_reason = None
            if position == "LONG":
                if c >= target_level:
                    exit_reason = "target"
                elif c <= stop_level:
                    exit_reason = "stop"
            else:
                if c <= target_level:
                    exit_reason = "target"
                elif c >= stop_level:
                    exit_reason = "stop"
            if not exit_reason and hold_bars >= 30:
                exit_reason = "time"
            if not exit_reason and is_last_bar_of_day:
                exit_reason = "eod"

            if exit_reason:
                raw = (c - entry_price) / entry_price if position == "LONG" else (entry_price - c) / entry_price
                trades.append(PocTrade(
                    ticker, position, entry_time, t, entry_price, c, poc_at_entry, exit_reason,
                    raw * 100, (raw - ROUND_TRIP_COST_PCT) * 100,
                ))
                position = None
            continue

        if state == "WAITING_PULLBACK":
            pullback_bars_left -= 1
            rng = pending_swing_high - pending_swing_low
            tol = config.pullback_tolerance * rng
            near_poc = pending_poc is not None and abs(c - pending_poc) <= tol

            if near_poc:
                touched_poc_zone = True

            # Confirmation: having touched the POC zone, a bar now closes
            # AWAY from POC in the trade direction -- the pullback holding,
            # not breaking through.
            fvg_ok = (not config.require_fvg) or (pending_fvg is not None)
            if touched_poc_zone and pending_dir == "UP_SWEEP" and c > pending_poc + tol * 0.25 and fvg_ok:
                entry_price, entry_time, hold_bars = c, t, 0
                poc_at_entry = pending_poc
                stop_level = pending_extreme
                target_level = entry_price - config.target_fraction * rng
                position = "SHORT"
                state, pending_dir, touched_poc_zone, pending_fvg = "NONE", None, False, None
                continue
            if touched_poc_zone and pending_dir == "DOWN_SWEEP" and c < pending_poc - tol * 0.25 and fvg_ok:
                entry_price, entry_time, hold_bars = c, t, 0
                poc_at_entry = pending_poc
                stop_level = pending_extreme
                target_level = entry_price + config.target_fraction * rng
                position = "LONG"
                state, pending_dir, touched_poc_zone, pending_fvg = "NONE", None, False, None
                continue

            # Invalidation: price re-breaks the original sweep extreme while
            # waiting -- the reversal thesis has failed, abandon the setup.
            if pending_dir == "UP_SWEEP" and h > pending_extreme:
                state, pending_dir, touched_poc_zone, pending_fvg = "NONE", None, False, None
                continue
            if pending_dir == "DOWN_SWEEP" and l < pending_extreme:
                state, pending_dir, touched_poc_zone, pending_fvg = "NONE", None, False, None
                continue

            if pullback_bars_left <= 0:
                state, pending_dir, touched_poc_zone, pending_fvg = "NONE", None, False, None
            continue

        if state == "WAITING_CONFIRM":
            pending_bars_left -= 1
            if pending_dir == "UP_SWEEP" and c < pending_swing_high:
                state = "WAITING_PULLBACK"
                pullback_bars_left = config.pullback_window
                continue
            if pending_dir == "DOWN_SWEEP" and c > pending_swing_low:
                state = "WAITING_PULLBACK"
                pullback_bars_left = config.pullback_window
                continue
            if pending_bars_left <= 0:
                state, pending_dir, pending_fvg = "NONE", None, None
            continue

        # NONE: only look for a fresh sweep if the pre-sweep range actually
        # qualified as a consolidation (the accumulation+consolidation gate).
        if pd.isna(sh) or pd.isna(sl) or pd.isna(atr) or atr <= 0 or is_last_bar_of_day:
            continue
        if not is_consolidating:
            continue
        if config.require_killzone and not bool(row.get("in_killzone", False)):
            continue   # killzone gate: only watch for a fresh sweep inside a real ICT session window
        buffer = config.buffer_mult * atr
        if h > sh + buffer:
            state, pending_dir = "WAITING_CONFIRM", "UP_SWEEP"
            pending_extreme, pending_swing_high, pending_swing_low = h, sh, sl
            pending_poc = row["poc"] if pd.notna(row["poc"]) else None
            pending_bars_left = config.confirm_window
        elif l < sl - buffer:
            if config.require_bullish_trend and not bool(row.get("is_bullish_trend", False)):
                continue   # bullish-trend gate: only watch DOWN_SWEEP (-> LONG) setups when the daily trend is already bullish
            if config.require_bullish_structure and not bool(row.get("is_bullish_structure", False)):
                continue   # bullish-structure gate: only watch DOWN_SWEEP (-> LONG) setups when real HH/HL swing structure is bullish
            state, pending_dir = "WAITING_CONFIRM", "DOWN_SWEEP"
            pending_extreme, pending_swing_high, pending_swing_low = l, sh, sl
            pending_poc = row["poc"] if pd.notna(row["poc"]) else None
            pending_bars_left = config.confirm_window

    return trades


def summarize(trades: list[PocTrade]) -> dict:
    if not trades:
        return {"n_trades": 0}
    rets = np.array([t.net_ret_pct for t in trades])
    wins = (rets > 0).sum()
    equity = np.cumsum(rets)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    return {
        "n_trades": len(trades),
        "win_rate_pct": round(100 * wins / len(trades), 2),
        "avg_ret_pct": round(float(rets.mean()), 4),
        "total_ret_sum_pct": round(float(rets.sum()), 2),
        "best_pct": round(float(rets.max()), 4),
        "worst_pct": round(float(rets.min()), 4),
        "max_drawdown_pct": round(float(drawdown.min()), 2),
        "exit_reasons": exit_reasons,
    }
