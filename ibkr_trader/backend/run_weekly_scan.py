#!/usr/bin/env python3
"""
run_weekly_scan.py  —  Replay the CSP scan for a specific week
==============================================================
Downloads historical data, runs the full scan/filter/score pipeline as
of a given entry date, then measures P&L through the exit date.

Usage:
    python run_weekly_scan.py --entry 2026-06-16 --exit 2026-06-20
    python run_weekly_scan.py              # defaults: last Mon -> last Fri
"""

import argparse, math, sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# --- Universe ----------------------------------------------------------------

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "JPM", "GS", "V", "MA",
    "SPY", "QQQ",
    "AMD", "MU", "AVGO",
    "COST", "HD", "LLY", "UNH",
]

# --- Constants (mirror main.py) ----------------------------------------------
IV_RANK_MIN_CSP     = 30
CSP_MAX_DELTA       = 0.35
CSP_MIN_RETURN_PCT  = 1.0
PROFIT_TARGET_PCT   = 0.50
EARNINGS_BLOCK_DAYS = 28


# --- Math helpers ------------------------------------------------------------

def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs_put(S, K, T, sigma, r=0.05):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def _bs_delta(S, K, T, sigma, r=0.05):
    if T <= 0 or sigma <= 0 or S <= 0:
        return -1.0 if K > S else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1) - 1.0


# --- Indicators --------------------------------------------------------------

def _indicators(closes, volumes=None):
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
        if ser is None: return None
        v = float(ser.iloc[-1])
        return None if math.isnan(v) else v

    last = float(closes[-1])
    s20  = _v(sma20); s50 = _v(sma50); s200 = _v(sma200)
    bu   = _v(bb_upper); bl = _v(bb_lower)
    pct_b = round((last - bl) / (bu - bl) * 100, 1) if bu and bl and bu != bl else None
    rsi_val = float(rsi.iloc[-1]); rsi_val = None if math.isnan(rsi_val) else round(rsi_val, 1)

    vol_ratio = None
    if volumes is not None and len(volumes) >= 21:
        avg = float(np.mean(volumes[-21:-1]))
        vol_ratio = round(float(volumes[-1]) / avg, 2) if avg > 0 else None

    return {
        "sma20": s20, "sma50": s50, "sma200": s200,
        "above_sma20":  (last > s20)  if s20  is not None else None,
        "above_sma50":  (last > s50)  if s50  is not None else None,
        "above_sma200": (last > s200) if s200 is not None else None,
        "bb_upper": bu, "bb_lower": bl, "pct_b": pct_b,
        "rsi14": rsi_val, "vol_ratio": vol_ratio,
    }


def _iv_rank(rv, history):
    if len(history) < 20: return 50.0
    arr = np.array(history)
    return round(float(np.mean(arr <= rv)) * 100, 1)


def _vol_history(closes, up_to):
    hist = []
    for k in range(max(21, up_to - 252), up_to, 5):
        if k < 21: continue
        lr = np.log(closes[k-20:k] / closes[k-21:k-1])
        if len(lr) >= 19:
            hist.append(float(np.std(lr) * np.sqrt(252)))
    return hist


# --- Scoring (mirrors main.py _score_csp exactly) ----------------------------

def _score_csp(row):
    s = 0.0
    iv_rank = float(row.get("iv_rank", 50))
    s += (iv_rank / 100) * 35
    s += min(row["weekly_return_pct"] / CSP_MIN_RETURN_PCT, 2.0) * 20
    s += (CSP_MAX_DELTA - abs(row["delta"])) / CSP_MAX_DELTA * 20
    s += (row.get("liquidity_score", 50) / 100) * 20
    s += row.get("stock_quality", 0.5) * 15

    spot = float(row.get("stock_price") or 0)
    strike = float(row.get("strike") or 0)
    if spot > 0 and strike > 0 and spot > strike:
        otm = (spot - strike) / spot
        prem_pct = row["weekly_return_pct"] / 100
        if otm > 0:
            s += min(prem_pct / otm, 2.0) * 5

    # RSI timing bonus
    rsi14 = row.get("rsi14")
    if rsi14 is not None:
        if iv_rank >= 45 and 35 <= rsi14 <= 48:
            s += 5
        elif 35 <= rsi14 <= 52:
            s += 2
        elif rsi14 < 35:
            s -= 3

    pct_b = row.get("pct_b")
    if pct_b is not None and pct_b > 80:
        s -= 4

    return round(s, 2)


def _passes_filter(row):
    return (
        len(row.get("warnings", [])) == 0
        and row.get("liquidity_score", 0) >= 50
        and row.get("iv_rank", 0) >= IV_RANK_MIN_CSP
        and row.get("score", 0) >= 70
        and row.get("above_sma20")  is not False
        and row.get("above_sma50")  is not False
        and row.get("above_sma200") is not False
    )


def _stop_mult(iv_pct):
    if iv_pct > 70: return 4.0
    if iv_pct > 40: return 3.0
    return 2.0


# --- Main analysis ------------------------------------------------------------

def scan_week(entry_date: date, exit_date: date, verbose=True):
    print(f"\n{'='*72}")
    print(f"  CSP SCAN REPLAY  —  Entry: {entry_date}  |  Exit: {exit_date}")
    print(f"  Universe: {len(UNIVERSE)} tickers")
    print(f"{'='*72}\n")

    candidates = []
    rejected   = []

    for ticker in UNIVERSE:
        try:
            hist = yf.Ticker(ticker).history(period="2y", interval="1d")
        except Exception as e:
            print(f"  {ticker}: fetch error — {e}")
            continue

        if hist.empty or len(hist) < 60:
            print(f"  {ticker}: insufficient data")
            continue

        # -- Find bar index closest to entry date --------------------------
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        entry_dt   = pd.Timestamp(entry_date)
        idx_arr    = np.searchsorted(hist.index, entry_dt)
        idx_arr    = min(idx_arr, len(hist) - 1)
        # Use the last available bar on or before entry date
        while idx_arr > 0 and hist.index[idx_arr] > entry_dt:
            idx_arr -= 1

        if idx_arr < 200:
            print(f"  {ticker}: not enough history before {entry_date}")
            continue

        closes  = hist["Close"].values[:idx_arr + 1].astype(float)
        volumes = hist["Volume"].values[:idx_arr + 1].astype(float)
        S       = float(closes[-1])
        entry_bar_date = hist.index[idx_arr].date()

        # -- Compute indicators --------------------------------------------
        ind  = _indicators(closes, volumes)
        if not ind:
            continue

        # -- Realized vol + IV rank ----------------------------------------
        if idx_arr < 21:
            continue
        lr  = np.log(closes[-20:] / closes[-21:-1])
        rv  = float(np.std(lr) * np.sqrt(252))
        if rv <= 0.05:
            continue
        vol_hist = _vol_history(closes, idx_arr)
        ivr      = _iv_rank(rv, vol_hist)

        # -- Days to expiry: nearest weekly expiry after entry -------------
        days_fwd = 7
        T = days_fwd / 252

        # -- Generate best candidate strike --------------------------------
        best_row   = None
        best_score = -1.0
        for otm_frac in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]:
            K     = round(S * (1 - otm_frac), 0)
            prem  = _bs_put(S, K, T, rv)
            delta = _bs_delta(S, K, T, rv)
            if prem < 0.05 or abs(delta) > CSP_MAX_DELTA:
                continue
            wk_ret = prem / K * 100
            row = {
                "ticker":            ticker,
                "stock_price":       S,
                "strike":            K,
                "otm_pct":          round(otm_frac * 100, 1),
                "weekly_return_pct": wk_ret,
                "delta":             delta,
                "iv_rank":           ivr,
                "iv_pct":            round(rv * 100, 1),
                "liquidity_score":   75,
                "stock_quality":     0.6,
                "warnings":          [],
                "earnings_days_out": None,
                "above_sma20":       ind.get("above_sma20"),
                "above_sma50":       ind.get("above_sma50"),
                "above_sma200":      ind.get("above_sma200"),
                "pct_b":             ind.get("pct_b"),
                "rsi14":             ind.get("rsi14"),
                "vol_ratio":         ind.get("vol_ratio"),
                "entry_bar_date":    entry_bar_date,
            }
            row["score"] = _score_csp(row)
            if row["score"] > best_score:
                best_score, best_row = row["score"], row

        if best_row is None:
            rejected.append((ticker, "no valid strike"))
            continue

        # -- Apply filter --------------------------------------------------
        reject_reason = None
        if best_row.get("above_sma200") is False:
            reject_reason = "below SMA-200"
        elif best_row.get("above_sma50") is False:
            reject_reason = "below SMA-50"
        elif best_row.get("above_sma20") is False:
            reject_reason = "below SMA-20"
        elif best_row.get("iv_rank", 0) < IV_RANK_MIN_CSP:
            reject_reason = f"IV rank {best_row['iv_rank']:.0f}% < {IV_RANK_MIN_CSP}%"
        elif best_row.get("score", 0) < 70:
            reject_reason = f"score {best_row['score']:.1f} < 70"

        if reject_reason:
            rejected.append((ticker, reject_reason))
            continue

        candidates.append(best_row)

    # -- Sort by score -----------------------------------------------------
    candidates.sort(key=lambda r: r["score"], reverse=True)

    print(f"  SCAN RESULTS: {len(candidates)} passed filter, {len(rejected)} rejected\n")

    # -- Simulate trade outcomes --------------------------------------------
    trade_results = []
    for row in candidates:
        ticker = row["ticker"]
        K      = row["strike"]
        S      = row["stock_price"]
        rv     = row["iv_pct"] / 100
        T_entry = 7 / 252
        prem   = _bs_put(S, K, T_entry, rv)
        stop_mult = _stop_mult(row["iv_pct"])
        tgt_price  = prem * (1 - PROFIT_TARGET_PCT)   # buy back at 50% profit
        stop_price = prem * (1 + stop_mult)             # stop loss

        try:
            hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        except Exception:
            hist = pd.DataFrame()

        exit_pnl    = 0.0
        exit_reason = "expiry"
        exit_date_actual = exit_date
        S_exit      = S
        p_exit      = None

        if not hist.empty:
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            # Only bars after entry date and through exit date
            window = hist[(hist.index > pd.Timestamp(entry_date)) &
                          (hist.index <= pd.Timestamp(exit_date))]
            for j, (dt, bar) in enumerate(window.iterrows()):
                S_j = float(bar["Close"])
                days_remaining = max((exit_date - dt.date()).days, 0)
                T_j = max(days_remaining / 252, 0.001)
                p_j = _bs_put(S_j, K, T_j, rv)
                if p_j <= tgt_price:
                    exit_pnl   = (prem - p_j) * 100
                    exit_reason = "profit_target(50%)"
                    exit_date_actual = dt.date()
                    S_exit = S_j; p_exit = round(p_j, 2)
                    break
                if p_j >= stop_price:
                    exit_pnl   = (prem - p_j) * 100
                    exit_reason = f"stop_loss({stop_mult:.0f}x)"
                    exit_date_actual = dt.date()
                    S_exit = S_j; p_exit = round(p_j, 2)
                    break
            else:
                # Held to last bar
                if not window.empty:
                    S_exit = float(window.iloc[-1]["Close"])
                    exit_date_actual = window.index[-1].date()
                else:
                    S_exit = S
                intrinsic = max(0.0, K - S_exit)
                p_exit    = round(intrinsic, 2)
                exit_pnl  = (prem - intrinsic) * 100
                exit_reason = "expiry/held"

        pct_chg = round((S_exit - S) / S * 100, 2)
        rsi_bonus = (row.get("rsi14") is not None and
                     row.get("iv_rank", 0) >= 45 and
                     35 <= (row.get("rsi14") or 99) <= 48)

        trade_results.append({
            **row,
            "prem":              round(prem, 2),
            "tgt_price":         round(tgt_price, 2),
            "stop_price":        round(stop_price, 2),
            "stop_mult":         stop_mult,
            "S_exit":            round(S_exit, 2),
            "p_exit":            p_exit,
            "exit_pnl":          round(exit_pnl, 2),
            "exit_reason":       exit_reason,
            "exit_date":         exit_date_actual,
            "spot_chg_pct":      pct_chg,
            "win":               exit_pnl > 0,
            "rsi_bonus":         rsi_bonus,
        })

    # -- Print trade table -------------------------------------------------
    wins   = [t for t in trade_results if t["win"]]
    losses = [t for t in trade_results if not t["win"]]

    print(f"{'-'*110}")
    print(f"  {'TICKER':<6} {'SCORE':>6} {'SPOT':>8} {'STRIKE':>7} {'OTM%':>5} {'PREM':>5} "
          f"{'IVR':>5} {'RSI':>5} {'SMA':>9} {'BONUS':>6} {'EXIT':>20} {'P&L':>8} {'RESULT'}")
    print(f"{'-'*110}")

    for t in trade_results:
        sma = ("^/^/^" if (t.get("above_sma20") and t.get("above_sma50") and t.get("above_sma200"))
               else ("^/^/-" if (t.get("above_sma20") and t.get("above_sma50"))
               else "MIX"))
        bonus = "RSI+" if t["rsi_bonus"] else ""
        result = "WIN " if t["win"] else "LOSS"
        print(
            f"  {t['ticker']:<6} {t['score']:>6.1f} {t['stock_price']:>8.2f} {t['strike']:>7.0f}"
            f" {t['otm_pct']:>4.1f}% {t['prem']:>5.2f}"
            f" {t['iv_rank']:>4.0f}% {t['rsi14'] or 0:>5.1f}"
            f" {sma:>9} {bonus:>6}"
            f"  {t['exit_reason']:>18}  {t['exit_pnl']:>+7.0f}  {result}"
        )

    print(f"{'-'*110}")

    total_pnl = sum(t["exit_pnl"] for t in trade_results)
    print(f"\n  Passed filter:   {len(candidates)} / {len(UNIVERSE)} tickers")
    print(f"  Win / Loss:      {len(wins)} wins, {len(losses)} losses  "
          f"({len(wins)/max(len(trade_results),1)*100:.0f}% win rate)")
    print(f"  Total P&L:       {total_pnl:+.0f} per-contract (100 shares)")
    if wins:
        print(f"  Avg win:         +{sum(t['exit_pnl'] for t in wins)/len(wins):.0f}")
    if losses:
        print(f"  Avg loss:        {sum(t['exit_pnl'] for t in losses)/len(losses):+.0f}")

    # -- Print rejected tickers ---------------------------------------------
    print(f"\n  FILTERED OUT ({len(rejected)} tickers):")
    for tkr, reason in sorted(rejected, key=lambda x: x[0]):
        print(f"    {tkr:<6}  {reason}")

    # -- Detailed breakdown per trade ---------------------------------------
    if verbose and trade_results:
        print(f"\n{'-'*72}")
        print("  DETAILED TRADE QUALITY\n")
        for t in trade_results:
            print(f"  {t['ticker']}  (score {t['score']:.1f})")
            print(f"    Entry   {t['entry_bar_date']}  spot=${t['stock_price']:.2f}  "
                  f"K={t['strike']:.0f}  ({t['otm_pct']:.1f}% OTM)")
            print(f"    Premium ${t['prem']:.2f}  target=${t['tgt_price']:.2f}  "
                  f"stop=${t['stop_price']:.2f}  ({t['stop_mult']:.0f}x)")
            print(f"    IV rank {t['iv_rank']:.0f}%  RSI {t['rsi14']}  "
                  f"pct_b {t['pct_b']}  vol_ratio {t['vol_ratio']}")
            trend = []
            if t.get("above_sma20")  is True:  trend.append("above SMA-20")
            if t.get("above_sma50")  is True:  trend.append("above SMA-50")
            if t.get("above_sma200") is True:  trend.append("above SMA-200")
            print(f"    Trend:  {', '.join(trend) if trend else 'N/A'}")
            if t["rsi_bonus"]:
                print(f"    ** RSI bonus active: RSI {t['rsi14']} in 35-48 + IVR {t['iv_rank']:.0f}% >= 45")
            print(f"    Exit    {t['exit_date']}  spot=${t['S_exit']:.2f}  "
                  f"({t['spot_chg_pct']:+.2f}%)  "
                  f"put=${t['p_exit']}  [{t['exit_reason']}]")
            print(f"    P&L     {t['exit_pnl']:+.2f}  --> {'WIN' if t['win'] else 'LOSS'}")
            print()

    print(f"{'='*72}\n")
    return trade_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default=None, help="Entry date YYYY-MM-DD (default: last Monday)")
    ap.add_argument("--exit",  default=None, help="Exit date  YYYY-MM-DD (default: last Friday)")
    ap.add_argument("--quiet", action="store_true", help="Skip detailed breakdown")
    args = ap.parse_args()

    today = date.today()  # 2026-06-23

    if args.entry:
        entry_date = date.fromisoformat(args.entry)
    else:
        # Last Monday
        days_since_mon = today.weekday()  # Mon=0
        entry_date = today - timedelta(days=days_since_mon + 7)

    if args.exit:
        exit_date = date.fromisoformat(args.exit)
    else:
        # Last Friday
        exit_date = entry_date + timedelta(days=4)

    scan_week(entry_date, exit_date, verbose=not args.quiet)


if __name__ == "__main__":
    main()

