#!/usr/bin/env python3
"""
run_leap_backtest.py  —  Replay the LEAP scan for any entry date
================================================================
Downloads historical data, runs the full LEAP scan/filter/score pipeline
as of a given entry date, then tracks each position through the
evaluation date showing current P&L, stop-out, or profit target hits.

Usage:
    python run_leap_backtest.py --entry 2026-05-24 --eval 2026-06-24
    python run_leap_backtest.py              # defaults: 1 month ago -> today
"""

import argparse, math, sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Universe (mirrors main.py CSP_UNIVERSE)
# ---------------------------------------------------------------------------
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "JPM", "GS", "V", "MA",
    "SPY", "QQQ",
    "AMD", "MU", "AVGO",
    "COST", "HD", "LLY", "UNH",
]

# ---------------------------------------------------------------------------
# Constants (mirror main.py LEAP scanner)
# ---------------------------------------------------------------------------
LEAP_TARGET_DTE   = 365       # target 12-month LEAP
LEAP_MIN_DELTA    = 0.65      # deep ITM call = high delta
LEAP_MAX_DELTA    = 0.85
LEAP_MAX_IV       = 0.70      # don't overpay for vol
LEAP_MIN_LIQ      = 60        # liquidity threshold
LEAP_MAX_IV_RANK  = 75        # low IV = cheap premiums = good for buyers
PROFIT_TARGET_PCT = 0.50      # 50% of cost = take profit
RF_RATE           = 0.05      # risk-free rate


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(S: float, K: float, T: float, sigma: float, r: float = RF_RATE) -> float:
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def _bs_call_delta(S: float, K: float, T: float, sigma: float, r: float = RF_RATE) -> float:
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1)


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------
def _indicators(closes: np.ndarray, volumes: np.ndarray | None = None) -> dict:
    n = len(closes)
    if n < 26:
        return {}
    s = pd.Series(closes)
    sma20  = s.rolling(20).mean()
    sma50  = s.rolling(50).mean()  if n >= 50  else None
    sma200 = s.rolling(200).mean() if n >= 200 else None
    bb_std   = s.rolling(20).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    delta_ = s.diff()
    gain   = delta_.clip(lower=0).rolling(14).mean()
    loss   = (-delta_.clip(upper=0)).rolling(14).mean()
    rsi    = 100 - 100 / (1 + gain / (loss + 1e-9))

    def _v(ser):
        if ser is None:
            return None
        v = float(ser.iloc[-1])
        return None if math.isnan(v) else v

    last = float(closes[-1])
    s20  = _v(sma20); s50 = _v(sma50); s200 = _v(sma200)
    bu   = _v(bb_upper); bl = _v(bb_lower)
    pct_b = round((last - bl) / (bu - bl) * 100, 1) if bu and bl and bu != bl else None
    rsi_v = float(rsi.iloc[-1])
    rsi_v = None if math.isnan(rsi_v) else round(rsi_v, 1)
    macd_hist = None
    try:
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal    = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = round(float((macd_line - signal).iloc[-1]), 4)
    except Exception:
        pass

    return {
        "sma20": s20, "sma50": s50, "sma200": s200,
        "above_sma20":  (last > s20)  if s20  is not None else None,
        "above_sma50":  (last > s50)  if s50  is not None else None,
        "above_sma200": (last > s200) if s200 is not None else None,
        "pct_b": pct_b,
        "rsi14": rsi_v,
        "macd_hist": macd_hist,
    }


def _iv_rank(rv: float, history: list) -> float:
    """Percentile rank of current realized vol vs history."""
    if len(history) < 20:
        return 50.0
    arr = np.array(history)
    return round(float(np.mean(arr <= rv)) * 100, 1)


def _vol_history(closes: np.ndarray, up_to: int) -> list:
    """Build a history of 20-day realized vol samples for IV rank."""
    hist = []
    for k in range(max(21, up_to - 504), up_to, 5):
        if k < 21:
            continue
        lr = np.log(closes[k - 20:k] / closes[k - 21:k - 1])
        if len(lr) >= 19:
            hist.append(float(np.std(lr) * np.sqrt(252)))
    return hist


def _stop_mult_leap(iv_pct: float) -> float:
    """Mirror the IV-adjusted LEAP stop logic from _autotrader_monitor_coro."""
    if iv_pct > 70:
        return 0.40
    if iv_pct > 40:
        return 0.50
    return 0.60


# ---------------------------------------------------------------------------
# LEAP scoring (mirrors scan_leaps in main.py)
# ---------------------------------------------------------------------------
def _score_leap(delta: float, iv_rank: float, liq: float, sq: float) -> float:
    leap_iv_bonus = (1 - iv_rank / 100) * 15
    score = (
        sq * 55
        + (1 - abs(delta - 0.60)) * 35
        + leap_iv_bonus
        + (liq / 100) * 10
    )
    return round(score, 2)


# ---------------------------------------------------------------------------
# Filter (mirrors _filter_leap_recommended)
# ---------------------------------------------------------------------------
def _passes_leap_filter(row: dict) -> tuple[bool, str]:
    if row.get("liquidity_score", 0) < LEAP_MIN_LIQ:
        return False, f"liquidity {row['liquidity_score']:.0f} < {LEAP_MIN_LIQ}"
    if row.get("iv_rank", 50) > LEAP_MAX_IV_RANK:
        return False, f"IV rank {row['iv_rank']:.0f}% > {LEAP_MAX_IV_RANK}% (too expensive)"
    if row.get("above_sma20") is False:
        return False, "below SMA-20"
    if row.get("above_sma50") is False:
        return False, "below SMA-50"
    if row.get("iv_pct", 0) > LEAP_MAX_IV * 100:
        return False, f"IV {row['iv_pct']:.0f}% > {LEAP_MAX_IV*100:.0f}% (overpaying)"
    if not (LEAP_MIN_DELTA <= row.get("delta", 0) <= LEAP_MAX_DELTA):
        return False, f"delta {row.get('delta',0):.2f} outside [{LEAP_MIN_DELTA},{LEAP_MAX_DELTA}]"
    return True, ""


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------
def run_leap_backtest(entry_date: date, eval_date: date, verbose: bool = True):
    hold_days = (eval_date - entry_date).days

    print(f"\n{'='*80}")
    print(f"  LEAP BACKTEST  —  Entry: {entry_date}  |  Eval: {eval_date}  ({hold_days}d)")
    print(f"  LEAP type: deep-ITM 12-month long calls (delta 0.65-0.85)")
    print(f"  Universe: {len(UNIVERSE)} tickers")
    print(f"  Profit target: {PROFIT_TARGET_PCT*100:.0f}% of cost  |  Stop: 40-60% of cost (IV-adjusted)")
    print(f"{'='*80}\n")

    candidates  = []
    rejected    = []

    # Cache full history per ticker to avoid double-download
    hist_cache: dict = {}

    for ticker in UNIVERSE:
        try:
            hist = yf.Ticker(ticker).history(period="3y", interval="1d")
        except Exception as e:
            print(f"  {ticker}: download error — {e}")
            continue

        if hist.empty or len(hist) < 252:
            print(f"  {ticker}: insufficient history")
            continue

        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist_cache[ticker] = hist

        # ── Find entry bar ─────────────────────────────────────────────────
        entry_ts = pd.Timestamp(entry_date)
        idx = np.searchsorted(hist.index, entry_ts)
        idx = min(idx, len(hist) - 1)
        while idx > 0 and hist.index[idx] > entry_ts:
            idx -= 1

        if idx < 200:
            print(f"  {ticker}: not enough bars before {entry_date}")
            continue

        closes  = hist["Close"].values[:idx + 1].astype(float)
        volumes = hist["Volume"].values[:idx + 1].astype(float)
        S_entry = float(closes[-1])
        entry_bar_date = hist.index[idx].date()

        # ── Indicators at entry ────────────────────────────────────────────
        ind = _indicators(closes, volumes)
        if not ind:
            continue

        # ── Realized vol + IV rank at entry ───────────────────────────────
        lr  = np.log(closes[-20:] / closes[-21:-1])
        rv  = float(np.std(lr) * np.sqrt(252))
        if rv <= 0.05 or rv > LEAP_MAX_IV:
            rejected.append((ticker, f"realized vol {rv*100:.0f}% out of range"))
            continue
        vol_hist = _vol_history(closes, idx)
        ivr      = _iv_rank(rv, vol_hist)

        # ── LEAP target: 12-month call, find best delta 0.65-0.85 ─────────
        T_entry = LEAP_TARGET_DTE / 365.0
        best_row = None
        best_score = -999.0

        # Sweep strikes from 65% to 100% of spot (ITM calls have lower strikes)
        for strike_frac in np.arange(0.65, 1.01, 0.01):
            K = round(S_entry * strike_frac, 0)
            if K <= 0:
                continue
            delta = _bs_call_delta(S_entry, K, T_entry, rv)
            if not (LEAP_MIN_DELTA <= delta <= LEAP_MAX_DELTA):
                continue
            prem = _bs_call(S_entry, K, T_entry, rv)
            if prem <= 0.10:
                continue
            itm_pct = (S_entry - K) / S_entry * 100   # positive = ITM
            liq_score = 75.0   # no live order book in backtest
            sq        = 0.65   # uniform stock quality in backtest

            row = {
                "ticker":          ticker,
                "stock_price":     round(S_entry, 2),
                "strike":          K,
                "itm_pct":         round(itm_pct, 1),
                "delta":           round(delta, 3),
                "iv_pct":          round(rv * 100, 1),
                "iv_rank":         ivr,
                "entry_premium":   round(prem, 2),
                "entry_cost":      round(prem * 100, 2),
                "T_entry":         T_entry,
                "entry_bar_date":  entry_bar_date,
                "above_sma20":     ind.get("above_sma20"),
                "above_sma50":     ind.get("above_sma50"),
                "above_sma200":    ind.get("above_sma200"),
                "rsi14":           ind.get("rsi14"),
                "macd_hist":       ind.get("macd_hist"),
                "liquidity_score": liq_score,
                "stock_quality":   sq,
            }
            row["score"] = _score_leap(delta, ivr, liq_score, sq)
            if row["score"] > best_score:
                best_score = row["score"]
                best_row   = row

        if best_row is None:
            rejected.append((ticker, "no valid delta 0.65-0.85 strike found"))
            continue

        # ── Apply filter ───────────────────────────────────────────────────
        ok, reason = _passes_leap_filter(best_row)
        if not ok:
            rejected.append((ticker, reason))
            continue

        candidates.append(best_row)

    # Sort by score
    candidates.sort(key=lambda r: r["score"], reverse=True)

    # --------------------------------------------------------------------------
    # Simulate daily P&L through eval_date
    # --------------------------------------------------------------------------
    results = []
    for row in candidates:
        ticker    = row["ticker"]
        K         = row["strike"]
        S_entry   = row["stock_price"]
        rv_entry  = row["iv_pct"] / 100
        T_entry   = row["T_entry"]
        prem_entry = row["entry_premium"]
        cost_entry = row["entry_cost"]   # per contract (100 shares)
        stop_frac = _stop_mult_leap(row["iv_pct"])
        tgt_pnl   = PROFIT_TARGET_PCT * cost_entry    # $X profit = close
        stop_pnl  = -stop_frac * cost_entry            # $X loss = stop

        hist = hist_cache.get(ticker, pd.DataFrame())
        exit_pnl      = None
        exit_reason   = "open"
        exit_date_act = None
        S_exit        = S_entry
        prem_exit     = prem_entry
        rv_exit       = rv_entry
        daily_pnl     = []   # pnl at each daily bar

        if not hist.empty:
            entry_ts = pd.Timestamp(row["entry_bar_date"])
            eval_ts  = pd.Timestamp(eval_date)
            window = hist[(hist.index > entry_ts) & (hist.index <= eval_ts)]

            for dt, bar in window.iterrows():
                S_j    = float(bar["Close"])
                # Realized vol in this window (20-day trailing)
                idx_j  = hist.index.get_loc(dt)
                if idx_j >= 21:
                    cl_j  = hist["Close"].values[max(0, idx_j - 20):idx_j + 1].astype(float)
                    lr_j  = np.log(cl_j[1:] / cl_j[:-1])
                    rv_j  = float(np.std(lr_j) * np.sqrt(252)) if len(lr_j) >= 5 else rv_entry
                else:
                    rv_j  = rv_entry
                rv_j = max(rv_j, 0.05)

                # Remaining time to LEAP expiry
                days_elapsed = (dt.date() - row["entry_bar_date"]).days
                T_j = max((LEAP_TARGET_DTE - days_elapsed) / 365.0, 0.001)
                prem_j  = _bs_call(S_j, K, T_j, rv_j)
                pnl_j   = (prem_j - prem_entry) * 100   # per contract
                daily_pnl.append((dt.date(), S_j, prem_j, pnl_j, rv_j))

                if exit_pnl is None:   # not yet exited
                    if pnl_j >= tgt_pnl:
                        exit_pnl    = pnl_j
                        exit_reason = f"profit_target({PROFIT_TARGET_PCT*100:.0f}%)"
                        exit_date_act = dt.date()
                        S_exit      = S_j
                        prem_exit   = prem_j
                        rv_exit     = rv_j
                    elif pnl_j <= stop_pnl:
                        exit_pnl    = pnl_j
                        exit_reason = f"stop_loss({stop_frac*100:.0f}%_cost)"
                        exit_date_act = dt.date()
                        S_exit      = S_j
                        prem_exit   = prem_j
                        rv_exit     = rv_j

        # If not stopped/profited, show mark-to-market as of eval date
        if exit_pnl is None and daily_pnl:
            _d, S_exit, prem_exit, exit_pnl, rv_exit = daily_pnl[-1]
            exit_date_act = daily_pnl[-1][0]
            exit_reason   = "open/mtm"
        elif exit_pnl is None:
            exit_pnl = 0.0
            exit_reason = "no data"

        spot_chg = (S_exit - S_entry) / S_entry * 100 if S_entry > 0 else 0.0
        pnl_pct  = exit_pnl / cost_entry * 100 if cost_entry > 0 else 0.0

        results.append({
            **row,
            "S_exit":       round(S_exit, 2),
            "prem_exit":    round(prem_exit, 2),
            "rv_exit":      round(rv_exit * 100, 1),
            "exit_pnl":     round(exit_pnl, 2),
            "pnl_pct":      round(pnl_pct, 1),
            "exit_reason":  exit_reason,
            "exit_date":    exit_date_act,
            "spot_chg_pct": round(spot_chg, 2),
            "stop_frac":    stop_frac,
            "tgt_pnl":      round(tgt_pnl, 2),
            "stop_pnl":     round(stop_pnl, 2),
            "is_winner":    exit_pnl > 0,
            "daily_pnl":    daily_pnl,
        })

    # --------------------------------------------------------------------------
    # Print summary table
    # --------------------------------------------------------------------------
    winners = [r for r in results if r["is_winner"]]
    losers  = [r for r in results if not r["is_winner"]]
    open_pos = [r for r in results if "mtm" in r["exit_reason"] or "open" == r["exit_reason"]]
    closed   = [r for r in results if r not in open_pos]

    print(f"  FILTER RESULTS: {len(candidates)} passed, {len(rejected)} rejected\n")
    print(f"  {'TICKER':<6} {'SCORE':>6} {'SPOT':>8} {'K':>7} {'ITM%':>5} {'DELTA':>6} {'IV%':>5}"
          f" {'IVR':>5} {'COST':>7} {'SMA':>8} {'EXIT SPOT':>10} {'PREM_EXIT':>10}"
          f" {'P&L':>8} {'P&L%':>6} {'STATUS'}")
    print(f"  {'-'*130}")

    for r in results:
        sma = ""
        if r.get("above_sma20") and r.get("above_sma50") and r.get("above_sma200"):
            sma = "^/^/^"
        elif r.get("above_sma20") and r.get("above_sma50"):
            sma = "^/^ "
        elif r.get("above_sma20"):
            sma = "^   "
        else:
            sma = "v   "

        status = r["exit_reason"]
        pnl_sign = "+" if r["exit_pnl"] >= 0 else ""
        print(
            f"  {r['ticker']:<6} {r['score']:>6.1f} {r['stock_price']:>8.2f} {r['strike']:>7.0f}"
            f" {r['itm_pct']:>4.1f}% {r['delta']:>6.3f} {r['iv_pct']:>4.0f}%"
            f" {r['iv_rank']:>4.0f}% ${r['entry_cost']:>6.0f}"
            f" {sma:>8}"
            f" {r['S_exit']:>10.2f} {r['prem_exit']:>10.2f}"
            f" {pnl_sign}{r['exit_pnl']:>7.0f} {pnl_sign}{r['pnl_pct']:>5.1f}%"
            f"  {status}"
        )

    print(f"  {'-'*130}")

    total_pnl = sum(r["exit_pnl"] for r in results)
    print(f"\n  Passed filter:   {len(candidates)} / {len(UNIVERSE)}")
    print(f"  Winners/Losers:  {len(winners)} wins, {len(losers)} losses"
          f"  ({len(winners)/max(len(results),1)*100:.0f}% win rate)")
    print(f"  Still open:      {len(open_pos)} positions (MTM as of {eval_date})")
    print(f"  Closed early:    {len(closed)} (profit target or stop)")
    print(f"  Total P&L:       {total_pnl:+,.0f} per-contract (100 shares)")
    if results:
        print(f"  Avg P&L:         {total_pnl/len(results):+,.0f}")
    if winners:
        print(f"  Avg winner:      +{sum(r['exit_pnl'] for r in winners)/len(winners):,.0f}")
    if losers:
        print(f"  Avg loser:       {sum(r['exit_pnl'] for r in losers)/len(losers):+,.0f}")

    # --------------------------------------------------------------------------
    # Rejected tickers
    # --------------------------------------------------------------------------
    print(f"\n  FILTERED OUT ({len(rejected)} tickers):")
    for tkr, reason in sorted(rejected, key=lambda x: x[0]):
        print(f"    {tkr:<6}  {reason}")

    # --------------------------------------------------------------------------
    # Detailed breakdown per trade
    # --------------------------------------------------------------------------
    if verbose and results:
        print(f"\n  {'='*80}")
        print("  DETAILED TRADE VIEW\n")
        for r in results:
            status_color = "OPEN/MTM" if "mtm" in r["exit_reason"] else r["exit_reason"].upper()
            pnl_label    = f"{r['exit_pnl']:+.0f} ({r['pnl_pct']:+.1f}% of cost)"
            print(f"  {r['ticker']}  (score {r['score']:.1f})")
            print(f"    Entry   {r['entry_bar_date']}  spot=${r['stock_price']:.2f}  "
                  f"K=${r['strike']:.0f}  ({r['itm_pct']:.1f}% ITM)  delta={r['delta']:.3f}")
            print(f"    Premium ${r['entry_premium']:.2f}  cost=${r['entry_cost']:.0f}/contract  "
                  f"IV={r['iv_pct']:.0f}%  IV-rank={r['iv_rank']:.0f}%")
            print(f"    Target: +${r['tgt_pnl']:.0f} ({PROFIT_TARGET_PCT*100:.0f}% of cost)  "
                  f"Stop: -${abs(r['stop_pnl']):.0f} ({r['stop_frac']*100:.0f}% of cost)")
            trend = []
            if r.get("above_sma20")  is True:  trend.append("above SMA-20")
            if r.get("above_sma50")  is True:  trend.append("above SMA-50")
            if r.get("above_sma200") is True:  trend.append("above SMA-200")
            print(f"    Trend:  {', '.join(trend) if trend else 'below key MAs'}")
            print(f"    RSI={r.get('rsi14') or 'N/A'}  MACD hist={r.get('macd_hist') or 'N/A'}")
            print(f"    As of   {r['exit_date']}  spot=${r['S_exit']:.2f}"
                  f"  ({r['spot_chg_pct']:+.2f}%)  call=${r['prem_exit']:.2f}"
                  f"  rv={r['rv_exit']:.0f}%")
            print(f"    P&L     {pnl_label}  [{status_color}]")

            # Show daily bar history
            if r["daily_pnl"]:
                print(f"    Daily P&L trail:")
                for d_, s_, p_, pnl_, rv_ in r["daily_pnl"]:
                    marker = ""
                    if "profit" in r["exit_reason"] and d_ == r["exit_date"]:
                        marker = " <-- PROFIT TARGET HIT"
                    elif "stop" in r["exit_reason"] and d_ == r["exit_date"]:
                        marker = " <-- STOP TRIGGERED"
                    print(f"      {d_}  spot=${s_:.2f}  call=${p_:.2f}  "
                          f"pnl={pnl_:+.0f}  rv={rv_*100:.0f}%{marker}")
            print()

    print(f"  {'='*80}\n")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default=None, help="Entry date YYYY-MM-DD (default: 1 month ago)")
    ap.add_argument("--eval",  default=None, help="Evaluation date YYYY-MM-DD (default: today)")
    ap.add_argument("--quiet", action="store_true", help="Skip detailed per-trade breakdown")
    args = ap.parse_args()

    today = date.today()

    entry_date = date.fromisoformat(args.entry) if args.entry else today - timedelta(days=31)
    eval_date  = date.fromisoformat(args.eval)  if args.eval  else today

    if entry_date >= eval_date:
        print(f"ERROR: entry ({entry_date}) must be before eval ({eval_date})")
        sys.exit(1)

    run_leap_backtest(entry_date, eval_date, verbose=not args.quiet)


if __name__ == "__main__":
    main()
