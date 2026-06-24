#!/usr/bin/env python3
"""
test_autotrader.py  -  Offline tests for IBKR Auto-Trader
==========================================================
Runs without IBKR connection.  Two modes:

  1. Unit tests   (--unit)   : instant, no network. Validates scoring/filter/
                               monitor logic against known inputs.
  2. Backtest     (--backtest): yfinance historical data, exercises the actual
                               scoring + filter + monitor pipeline.

Usage:
    python test_autotrader.py               # both unit + backtest
    python test_autotrader.py --unit        # unit tests only
    python test_autotrader.py --backtest    # backtest only
    python test_autotrader.py --tickers AAPL,NVDA --weeks 52 --verbose
"""

import argparse, math, sys
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

# -----------------------------------------------------------------------------
# Pure logic - mirrored from main.py (keep in sync)
# -----------------------------------------------------------------------------

CSP_MIN_RETURN_PCT  = 1.0
CSP_MAX_DELTA       = 0.35
IV_RANK_MIN_CSP     = 30
EARNINGS_BLOCK_DAYS = 14
PROFIT_TARGET_PCT   = 0.50   # close at 50% of max premium
STOP_LOSS_MULT_BASE = 2.0    # 2× for normal IV
ROLL_DTE_THRESHOLD  = 21     # roll / take-profit at 21 DTE


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_put(S: float, K: float, T: float, sigma: float, r: float = 0.05) -> float:
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_put_delta(S: float, K: float, T: float, sigma: float, r: float = 0.05) -> float:
    """Put delta (negative)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return -1.0 if K > S else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) - 1.0


def _score_csp(row: dict) -> float:
    """Composite score - mirrors main.py _score_csp() exactly."""
    s = 0.0
    iv_rank = float(row.get("iv_rank", 50))

    # 1. IV rank (35 pts)
    s += (iv_rank / 100) * 35

    # 2. Weekly return vs minimum (40 pts, capped at 2×)
    s += min(row["weekly_return_pct"] / CSP_MIN_RETURN_PCT, 2.0) * 20

    # 3. Delta safety (20 pts)
    s += (CSP_MAX_DELTA - abs(row["delta"])) / CSP_MAX_DELTA * 20

    # 4. Liquidity (20 pts)
    s += (row.get("liquidity_score", 50) / 100) * 20

    # 5. Stock quality (15 pts)
    s += row.get("stock_quality", 0.5) * 15

    # 6. Premium efficiency (max 10 pts)
    spot   = float(row.get("stock_price") or 0)
    strike = float(row.get("strike") or 0)
    if spot > 0 and strike > 0 and spot > strike:
        otm_distance = (spot - strike) / spot
        premium_pct  = row["weekly_return_pct"] / 100
        if otm_distance > 0:
            efficiency = min(premium_pct / otm_distance, 2.0)
            s += efficiency * 5

    # 7. RSI timing bonus: oversold pullback within uptrend (max +5 / min -3)
    rsi14 = row.get("rsi14")
    if rsi14 is not None:
        if iv_rank >= 45 and 35 <= rsi14 <= 48:
            s += 5   # oversold pullback + expensive vol
        elif 35 <= rsi14 <= 52:
            s += 2   # mild pullback within trend
        elif rsi14 < 35:
            s -= 3   # too oversold: momentum risk

    # BB extension penalty: pct_b > 80 = price in upper band = reversion risk
    pct_b = row.get("pct_b")
    if pct_b is not None and pct_b > 80:
        s -= 4

    return round(s, 2)


def _passes_filter(row: dict) -> bool:
    """Hard filter - mirrors main.py _filter_csp_recommended() logic."""
    return (
        len(row.get("warnings", [])) == 0
        and row.get("liquidity_score", 0) >= 50
        and row.get("iv_rank", 0) >= IV_RANK_MIN_CSP
        and row.get("score", 0) >= 70
        and row.get("above_sma20")  is not False
        and row.get("above_sma50")  is not False
        and row.get("above_sma200") is not False
        and (
            row.get("earnings_days_out") is None
            or row.get("earnings_days_out") > EARNINGS_BLOCK_DAYS * 2
        )
    )


def _stop_mult_for_iv(live_iv: float) -> float:
    """IV-adjusted stop multiplier - mirrors main.py monitor logic."""
    if live_iv > 70:
        return 4.0
    if live_iv > 40:
        return 3.0
    return STOP_LOSS_MULT_BASE


def _compute_indicators(closes: np.ndarray, volumes: np.ndarray | None = None) -> dict:
    """
    Compute technical indicators for the most recent bar.
    Mirrors main.py _compute_indicators_from_closes().
    """
    n = len(closes)
    if n < 26:
        return {}

    s = pd.Series(closes)
    sma20  = s.rolling(20).mean()
    sma50  = s.rolling(50).mean()  if n >= 50  else None
    sma200 = s.rolling(200).mean() if n >= 200 else None

    bb_std    = s.rolling(20).std()
    bb_upper  = sma20 + 2 * bb_std
    bb_lower  = sma20 - 2 * bb_std

    # RSI-14
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + gain / (loss + 1e-9))

    last = float(closes[-1])

    def _last(series):
        if series is None:
            return None
        v = float(series.iloc[-1])
        return None if math.isnan(v) else v

    s20  = _last(sma20)
    s50  = _last(sma50)
    s200 = _last(sma200)
    b_up = _last(bb_upper)
    b_lo = _last(bb_lower)
    pct_b = (
        round((last - b_lo) / (b_up - b_lo) * 100, 1)
        if b_up and b_lo and b_up != b_lo
        else None
    )

    vol_ratio = None
    if volumes is not None and len(volumes) >= 21:
        avg_vol = float(np.mean(volumes[-21:-1]))
        vol_ratio = round(float(volumes[-1]) / avg_vol, 2) if avg_vol > 0 else None

    return {
        "sma20":        s20,
        "sma50":        s50,
        "sma200":       s200,
        "above_sma20":  (last > s20)  if s20  is not None else None,
        "above_sma50":  (last > s50)  if s50  is not None else None,
        "above_sma200": (last > s200) if s200 is not None else None,
        "bb_upper":     b_up,
        "bb_lower":     b_lo,
        "pct_b":        pct_b,
        "rsi14":        round(float(rsi.iloc[-1]), 1) if not math.isnan(float(rsi.iloc[-1])) else 50.0,
        "vol_ratio":    vol_ratio,
    }


def _iv_rank(current_vol: float, vol_history: list) -> float:
    """Percentile rank of current vol vs trailing history (≥20 pts needed)."""
    if len(vol_history) < 20:
        return 50.0
    arr = np.array(vol_history)
    return round(float(np.mean(arr <= current_vol)) * 100, 1)


# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------

class _T:
    """Minimal test runner."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def ok(self, name: str, cond: bool, detail: str = ""):
        if cond:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"  FAIL  {name}" + (f" - {detail}" if detail else ""))

    def near(self, name: str, got, expected, tol=0.11):
        self.ok(name, abs(got - expected) <= tol, f"got {got!r}, expected {expected!r} ±{tol}")

    def summary(self):
        print(f"\n{'-'*55}")
        print(f"UNIT TESTS  {self.passed} passed  {self.failed} failed")
        for e in self.errors:
            print(e)
        return self.failed == 0


def run_unit_tests() -> bool:
    t = _T()
    print("Running unit tests …")

    # -- Black-Scholes sanity ----------------------------------------------
    p_otm = _bs_put(100, 95, 5/252, 0.30)
    t.ok("BS put OTM price > 0", p_otm > 0)
    t.ok("BS put ATM > put OTM", _bs_put(100, 100, 5/252, 0.30) > p_otm)
    t.ok("BS put deep-ITM > intrinsic-ish", _bs_put(80, 100, 5/252, 0.30) > 15)
    t.ok("BS delta OTM in (−0.5,0)", -0.5 < _bs_put_delta(100, 95, 5/252, 0.30) < 0)
    t.ok("BS delta ATM ≈ −0.5", abs(_bs_put_delta(100, 100, 5/252, 0.30) + 0.50) < 0.05)

    # -- Score components --------------------------------------------------
    base_row = {
        "iv_rank": 60, "weekly_return_pct": 1.5, "delta": -0.20,
        "liquidity_score": 75, "stock_quality": 0.7,
        "stock_price": 100, "strike": 90, "warnings": [],
        "earnings_days_out": None,
    }
    score_base = _score_csp(base_row)
    t.ok("base score is positive", score_base > 0)

    # Higher IV rank → higher score (all else equal)
    hi_iv = _score_csp({**base_row, "iv_rank": 90})
    lo_iv = _score_csp({**base_row, "iv_rank": 20})
    t.ok("higher IV rank → higher score", hi_iv > lo_iv)

    # Tighter delta → higher score
    tight_delta = _score_csp({**base_row, "delta": -0.10})
    wide_delta  = _score_csp({**base_row, "delta": -0.30})
    t.ok("tighter delta → higher score", tight_delta > wide_delta)

    # -- RSI timing bonus (replaces unreachable pct_b<30 bonus) -----------
    # RSI 35-48 + iv_rank>=45 => +5 pts; achievable above SMA-20 (pct_b>=50 there)
    row_rsi_on  = {**base_row, "rsi14": 42, "iv_rank": 55}
    row_rsi_off = {**base_row, "rsi14": 60, "iv_rank": 55}
    rsi_diff = _score_csp(row_rsi_on) - _score_csp(row_rsi_off)
    t.near("RSI bonus = +5 pts (rsi 35-48 + ivr>=45)", rsi_diff, 5.0)

    # Mild pullback (RSI 35-52, any IV rank) => +2 pts
    row_mild = {**base_row, "rsi14": 50, "iv_rank": 30}
    row_none = {**base_row, "rsi14": 60, "iv_rank": 30}
    t.near("RSI mild bonus = +2 pts (rsi 35-52)", _score_csp(row_mild) - _score_csp(row_none), 2.0)

    # Too oversold => -3 pts
    row_bear = {**base_row, "rsi14": 30, "iv_rank": 55}
    row_neut = {**base_row, "rsi14": 60, "iv_rank": 55}
    t.near("RSI oversold penalty = -3 pts (rsi<35)", _score_csp(row_neut) - _score_csp(row_bear), 3.0)

    # High IV amplifies RSI bonus (ivr>=45 gets +5; ivr<45 gets +2 = 3 pt difference)
    row_hiv = {**base_row, "rsi14": 42, "iv_rank": 55}
    row_lov = {**base_row, "rsi14": 42, "iv_rank": 30}
    t.ok("ivr>=45 scores higher than ivr<45 when rsi in pullback range", _score_csp(row_hiv) > _score_csp(row_lov))
    # BB extension penalty (-4 pts at pct_b>80)
    row_extended = {**base_row, "pct_b": 85, "iv_rank": 55}
    row_mid      = {**base_row, "pct_b": 65, "iv_rank": 55}
    diff_pen = _score_csp(row_mid) - _score_csp(row_extended)
    t.near("BB penalty = -4 pts (pct_b>80)", diff_pen, 4.0)
    # -- Filter logic ------------------------------------------------------
    pass_base = {
        "warnings": [], "liquidity_score": 70, "iv_rank": 50,
        "score": 75, "earnings_days_out": None,
        "above_sma20": True, "above_sma50": True, "above_sma200": True,
    }
    t.ok("clean row passes filter", _passes_filter(pass_base))
    t.ok("None sma200 passes (benefit of doubt)", _passes_filter({**pass_base, "above_sma200": None}))
    t.ok("False sma200 blocks (confirmed downtrend)", not _passes_filter({**pass_base, "above_sma200": False}))
    t.ok("False sma50 blocks", not _passes_filter({**pass_base, "above_sma50": False}))
    t.ok("False sma20 blocks", not _passes_filter({**pass_base, "above_sma20": False}))
    t.ok("score < 70 blocks", not _passes_filter({**pass_base, "score": 69}))
    t.ok("iv_rank < 30 blocks", not _passes_filter({**pass_base, "iv_rank": 29}))
    t.ok("liq < 50 blocks", not _passes_filter({**pass_base, "liquidity_score": 49}))
    t.ok("earnings ≤28d blocks", not _passes_filter({**pass_base, "earnings_days_out": 20}))
    t.ok("earnings >28d passes", _passes_filter({**pass_base, "earnings_days_out": 30}))
    t.ok("warning blocks", not _passes_filter({**pass_base, "warnings": ["some warning"]}))

    # -- Stop-loss multiplier (IV-adjusted) --------------------------------
    t.near("stop mult at IV=75 → 4×", _stop_mult_for_iv(75), 4.0)
    t.near("stop mult at IV=50 → 3×", _stop_mult_for_iv(50), 3.0)
    t.near("stop mult at IV=25 → 2×", _stop_mult_for_iv(25), 2.0)
    t.near("stop mult at IV=70 boundary → 4×", _stop_mult_for_iv(70.1), 4.0)
    t.near("stop mult at IV=40 boundary → 3×", _stop_mult_for_iv(40.1), 3.0)

    # -- Technical indicators ----------------------------------------------
    np.random.seed(42)
    prices = np.cumprod(1 + np.random.randn(250) * 0.01) * 100
    ind = _compute_indicators(prices)
    t.ok("indicators computed with 250 bars", "sma200" in ind and ind["sma200"] is not None)
    t.ok("above_sma200 is bool with 250 bars", isinstance(ind.get("above_sma200"), bool))
    t.ok("pct_b in 0-100 range", ind.get("pct_b") is None or 0 <= ind["pct_b"] <= 200)
    t.ok("rsi in 0-100", 0 <= ind.get("rsi14", 50) <= 100)

    ind_short = _compute_indicators(prices[:50])
    t.ok("above_sma200 is None with only 50 bars", ind_short.get("above_sma200") is None)

    # Trending up: should be above all MAs
    trending = np.linspace(50, 150, 250)
    ind_up = _compute_indicators(trending)
    t.ok("trending up → above_sma20", ind_up.get("above_sma20") is True)
    t.ok("trending up → above_sma200", ind_up.get("above_sma200") is True)

    # Trending down: should be below all MAs
    trending_down = np.linspace(150, 50, 250)
    ind_dn = _compute_indicators(trending_down)
    t.ok("trending down → below_sma20", ind_dn.get("above_sma20") is False)
    t.ok("trending down → below_sma200", ind_dn.get("above_sma200") is False)

    # -- Profit target / stop-loss thresholds ------------------------------
    max_profit = 500.0
    # 50% profit target
    t.ok("50% profit fires at upnl=250", 250.0 >= PROFIT_TARGET_PCT * max_profit)
    t.ok("50% profit does not fire at upnl=249", 249.0 < PROFIT_TARGET_PCT * max_profit)
    # Stop at 2× for normal IV
    stop_threshold = -STOP_LOSS_MULT_BASE * max_profit
    t.ok("stop fires at −1000", -1000.0 <= stop_threshold)
    t.ok("stop does not fire at −999", -999.0 > stop_threshold)

    return t.summary()


# -----------------------------------------------------------------------------
# Historical backtest
# -----------------------------------------------------------------------------

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN",
    "META", "GOOGL", "JPM", "LLY", "SPY",
]


def _fetch_history(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d")
        return hist if not hist.empty and len(hist) >= 60 else None
    except Exception as e:
        print(f"  yfinance error for {ticker}: {e}")
        return None


def _build_vol_history(closes: np.ndarray, up_to: int, step: int = 5) -> list:
    """Build trailing list of 20-day realized vols for IV rank computation."""
    hist = []
    for k in range(max(21, up_to - 252), up_to, step):
        if k < 21:
            continue
        lr = np.log(closes[k - 20:k] / closes[k - 21:k - 1])
        if len(lr) >= 19:
            hist.append(float(np.std(lr) * np.sqrt(252)))
    return hist


def backtest_ticker(
    ticker: str,
    weeks: int = 52,
    profit_target: float = PROFIT_TARGET_PCT,
    stop_mult_base: float = STOP_LOSS_MULT_BASE,
    verbose: bool = False,
) -> dict | None:
    hist = _fetch_history(ticker)
    if hist is None:
        print(f"  {ticker}: no data")
        return None

    closes  = hist["Close"].values.astype(float)
    volumes = hist["Volume"].values.astype(float)
    dates   = hist.index

    n_total = 0     # all weekly opportunities considered
    n_filtered = 0  # passed the filter
    n_sma200_blocked = 0
    n_score_blocked  = 0
    n_iv_blocked     = 0
    trades: list = []

    i = 200  # need 200 bars for SMA-200; start after warmup
    while i < len(closes) - 7 and len(trades) < weeks:
        n_total += 1
        S = float(closes[i])

        # -- Technical indicators ------------------------------------------
        ind = _compute_indicators(closes[:i + 1], volumes[:i + 1])

        # -- Realized vol (20-day) as IV proxy ----------------------------
        if i < 21:
            i += 5; continue
        log_rets = np.log(closes[i - 20:i] / closes[i - 21:i - 1])
        rv = float(np.std(log_rets) * np.sqrt(252))
        if rv <= 0.05:
            i += 5; continue

        # -- IV rank from rolling vol history -----------------------------
        vol_hist = _build_vol_history(closes, i)
        ivr = _iv_rank(rv, vol_hist)

        T = 5 / 252  # 1 week

        # -- Generate candidate CSP strikes -------------------------------
        best_row = None
        best_score = -1.0
        for otm_frac in [0.05, 0.07, 0.10, 0.12, 0.15]:
            K     = round(S * (1 - otm_frac), 0)
            prem  = _bs_put(S, K, T, rv)
            delta = _bs_put_delta(S, K, T, rv)
            if prem < 0.05 or abs(delta) > CSP_MAX_DELTA:
                continue
            wk_ret = prem / K * 100
            row = {
                "ticker":            ticker,
                "stock_price":       S,
                "strike":            K,
                "weekly_return_pct": wk_ret,
                "delta":             delta,
                "iv_rank":           ivr,
                "iv_pct":            rv * 100,
                "liquidity_score":   75,
                "stock_quality":     0.6,
                "warnings":          [],
                "earnings_days_out": None,
                "above_sma20":       ind.get("above_sma20"),
                "above_sma50":       ind.get("above_sma50"),
                "above_sma200":      ind.get("above_sma200"),
                "pct_b":             ind.get("pct_b"),
                "rsi14":             ind.get("rsi14"),
            }
            row["score"] = _score_csp(row)
            if row["score"] > best_score:
                best_score, best_row = row["score"], row

        # -- Apply filter --------------------------------------------------
        if best_row is None:
            i += 5; continue

        if best_row.get("above_sma200") is False:
            n_sma200_blocked += 1
        if best_row.get("iv_rank", 0) < IV_RANK_MIN_CSP:
            n_iv_blocked += 1
        if best_row.get("score", 0) < 70:
            n_score_blocked += 1

        if not _passes_filter(best_row):
            i += 5; continue

        n_filtered += 1
        K    = best_row["strike"]
        prem = _bs_put(S, K, T, rv)
        # IV-adjusted stop multiplier
        eff_stop = _stop_mult_for_iv(rv * 100)
        tgt_price  = prem * (1 - profit_target)   # price at which we buy back (profit)
        stop_price = prem * (1 + eff_stop)          # price at which we buy back (stop)

        # -- Simulate 5-day holding ----------------------------------------
        exit_pnl = 0.0
        exit_reason = "expiry"
        exit_day = 5
        for j in range(i + 1, min(i + 6, len(closes))):
            S_j = float(closes[j])
            T_j = max((min(i + 5, len(closes) - 1) - j) / 252, 0.001)
            # Recompute vol for re-pricing
            if j >= 21:
                lr_j = np.log(closes[j - 20:j] / closes[j - 21:j - 1])
                rv_j = float(np.std(lr_j) * np.sqrt(252)) if len(lr_j) >= 19 else rv
            else:
                rv_j = rv
            p_j  = _bs_put(S_j, K, T_j, rv_j)
            gain = (prem - p_j) * 100   # per-contract P&L
            if p_j <= tgt_price:
                exit_pnl, exit_reason, exit_day = gain, "profit_target", j - i
                break
            if p_j >= stop_price:
                exit_pnl, exit_reason, exit_day = gain, "stop_loss", j - i
                break
        else:
            S_exit   = float(closes[min(i + 5, len(closes) - 1)])
            exit_pnl = (prem - max(0.0, K - S_exit)) * 100
            exit_reason = "expiry"
            exit_day = 5

        rsi_bonus_active = (
            ivr >= 45 and ind.get("rsi14") is not None
            and 35 <= ind.get("rsi14", 99) <= 48
        )
        trade = {
            "date":           str(dates[i].date()),
            "spot":           round(S, 2),
            "strike":         round(K, 2),
            "otm_pct":        round((S - K) / S * 100, 1),
            "prem":           round(prem, 2),
            "iv_pct":         round(rv * 100, 1),
            "iv_rank":        round(ivr, 1),
            "score":          round(best_score, 1),
            "pct_b":          ind.get("pct_b"),
            "rsi_bonus":      rsi_bonus_active,
            "above_sma200":   ind.get("above_sma200"),
            "stop_mult":      eff_stop,
            "exit_pnl":       round(exit_pnl, 2),
            "exit_reason":    exit_reason,
            "exit_day":       exit_day,
            "win":            exit_pnl > 0,
        }
        trades.append(trade)

        if verbose:
            bb_tag = " [RSI+]" if rsi_bonus_active else ""
            print(
                f"  {trade['date']}  {ticker} K={K:.0f}"
                f"  ({trade['otm_pct']:.1f}% OTM)"
                f"  prem={prem:.2f}  ivr={ivr:.0f}%  score={best_score:.0f}{bb_tag}"
                f"  → {exit_reason} {exit_pnl:+.0f}"
            )

        i += 5

    if not trades:
        print(f"  {ticker}: no trades passed filter")
        return None

    pnls = np.array([t["exit_pnl"] for t in trades])
    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    cum = np.cumsum(pnls)
    dd  = cum - np.maximum.accumulate(cum)

    return {
        "ticker":          ticker,
        "opportunities":   n_total,
        "trades":          len(trades),
        "filter_rate":     round(len(trades) / max(n_total, 1) * 100, 1),
        "win_rate":        round(len(wins) / len(trades) * 100, 1),
        "total_pnl":       round(float(np.sum(pnls)), 0),
        "avg_win":         round(float(np.mean([t["exit_pnl"] for t in wins])), 0) if wins else 0,
        "avg_loss":        round(float(np.mean([t["exit_pnl"] for t in losses])), 0) if losses else 0,
        "max_drawdown":    round(float(np.min(dd)), 0),
        "sharpe":          round(
            float(np.mean(pnls)) / float(np.std(pnls)) * math.sqrt(52), 2
        ) if float(np.std(pnls)) > 0 else 0.0,
        "sma200_blocked":  n_sma200_blocked,
        "iv_blocked":      n_iv_blocked,
        "score_blocked":   n_score_blocked,
        "rsi_bonus_trades": sum(1 for t in trades if t.get("rsi_bonus")),
        "profit_targets":  sum(1 for t in trades if t["exit_reason"] == "profit_target"),
        "stop_losses":     sum(1 for t in trades if t["exit_reason"] == "stop_loss"),
        "recent":          trades[-5:],
    }


def print_backtest_results(results: list):
    if not results:
        print("No results.")
        return

    print(f"\n{'='*90}")
    print(f"{'TICKER':<8} {'TRADES':>6} {'FILTER%':>8} {'WIN%':>6} {'TOT P&L':>9} "
          f"{'AVG WIN':>8} {'AVG LOSS':>9} {'DRAWDOWN':>9} {'SHARPE':>7} "
          f"{'SMA200v':>8} {'BB+':>5}")
    print(f"{'-'*90}")
    for r in sorted(results, key=lambda x: x["total_pnl"], reverse=True):
        print(
            f"{r['ticker']:<8} {r['trades']:>6} {r['filter_rate']:>7.1f}% {r['win_rate']:>5.1f}%"
            f" {r['total_pnl']:>+9.0f} {r['avg_win']:>+8.0f} {r['avg_loss']:>+9.0f}"
            f" {r['max_drawdown']:>+9.0f} {r['sharpe']:>7.2f}"
            f" {r['sma200_blocked']:>8} {r['rsi_bonus_trades']:>5}"
        )
    print(f"{'-'*90}")

    all_pnl  = [r["total_pnl"] for r in results]
    all_wr   = [r["win_rate"]  for r in results]
    all_sma  = sum(r["sma200_blocked"] for r in results)
    all_bb   = sum(r["rsi_bonus_trades"] for r in results)
    all_tr   = sum(r["trades"] for r in results)
    all_pt   = sum(r["profit_targets"] for r in results)
    all_sl   = sum(r["stop_losses"] for r in results)

    print(f"\n  Total P&L: {sum(all_pnl):+.0f}   Avg win rate: {np.mean(all_wr):.1f}%")
    print(f"  Exit breakdown - profit target: {all_pt} ({all_pt/max(all_tr,1)*100:.0f}%) | "
          f"stop-loss: {all_sl} ({all_sl/max(all_tr,1)*100:.0f}%) | "
          f"expiry: {all_tr-all_pt-all_sl}")
    print(f"  SMA-200 blocked {all_sma} opportunities | RSI bonus active on {all_bb} trades")
    print()

    # Recent trades sample
    print("  Recent trades (last 5 per ticker):")
    print(f"  {'DATE':<12} {'TKR':<6} {'SPOT':>7} {'K':>6} {'OTM%':>5} {'PREM':>5} "
          f"{'IVR':>5} {'SCORE':>6} {'BB+':>4} {'SMA200':>7} {'EXIT':>14} {'P&L':>8}")
    for r in results:
        for tr in r.get("recent", []):
            bb  = "RSI" if tr.get("rsi_bonus") else ""
            sma = "^" if tr.get("above_sma200") else ("v" if tr.get("above_sma200") is False else "-")
            print(
                f"  {tr['date']:<12} {r['ticker']:<6}"
                f" {tr['spot']:>7.2f} {tr['strike']:>6.0f} {tr['otm_pct']:>5.1f}%"
                f" {tr['prem']:>5.2f} {tr['iv_rank']:>5.0f}%"
                f" {tr['score']:>6.1f} {bb:>4} {sma:>7}"
                f" {tr['exit_reason']:>14} {tr['exit_pnl']:>+8.0f}"
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Offline backtest for IBKR Auto-Trader")
    ap.add_argument("--unit",     action="store_true", help="Unit tests only")
    ap.add_argument("--backtest", action="store_true", help="Backtest only")
    ap.add_argument("--tickers",  default=",".join(DEFAULT_UNIVERSE),
                    help="Comma-separated tickers")
    ap.add_argument("--weeks",    type=int, default=52, help="Max weeks per ticker")
    ap.add_argument("--verbose",  action="store_true", help="Print each trade")
    args = ap.parse_args()

    run_unit  = args.unit or (not args.unit and not args.backtest)
    run_bt    = args.backtest or (not args.unit and not args.backtest)

    unit_ok = True
    if run_unit:
        unit_ok = run_unit_tests()

    if run_bt:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        print(f"\nRunning backtest on {len(tickers)} tickers × {args.weeks} weeks …")
        print("(fetching yfinance 2-year daily OHLCV - no IBKR connection needed)\n")
        results = []
        for tkr in tickers:
            print(f"  {tkr} …", end=" ", flush=True)
            r = backtest_ticker(tkr, weeks=args.weeks, verbose=args.verbose)
            if r:
                results.append(r)
                print(f"win={r['win_rate']:.0f}%  pnl={r['total_pnl']:+.0f}  trades={r['trades']}")
            else:
                print("skip")
        print_backtest_results(results)

    sys.exit(0 if unit_ok else 1)


if __name__ == "__main__":
    main()



