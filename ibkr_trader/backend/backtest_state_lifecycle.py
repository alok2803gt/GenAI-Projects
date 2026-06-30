#!/usr/bin/env python3
"""
backtest_state_lifecycle.py — State Machine Historical Backtest
===============================================================

Simulates the breakout scanner's 7-state Bollinger Band machine on 3 years of
daily bars to answer five structural questions about how state transition PATHS
predict returns better than the raw signal snapshot alone.

Questions answered:
  Q1  Conviction time   — days in PRE-BREAKOUT before BREAKOUT → better returns?
  Q2  Path quality      — 2-step (PRE-BREAKOUT→BREAKOUT) vs 1-step jump
  Q3  FADING value      — return at fade day vs. holding to 5d (is FADING a useful exit?)
  Q4  EXTENDED fate     — tickers that went EXTENDED: overshot or sustained?
  Q5  Setup re-entry    — after SETUP FAILED, how often does ticker re-enter within 5d?

Outputs (backtest_results/):
  state_lifecycle_signals.csv  — one row per BREAKOUT signal
  state_lifecycle_fading.csv   — one row per FADING event
  state_lifecycle_failed.csv   — one row per SETUP FAILED event
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "backtest_results")
os.makedirs(OUT_DIR, exist_ok=True)

YEARS      = 3
BB_PERIOD  = 20
VOL_MA     = 20      # denominator for vol_ratio
VOL_PCTILE = 90      # vol_90pct percentile threshold
SMA50_PERIOD = 50

TICKERS: list[str] = sorted(set([
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "GLD", "TLT", "ARKK",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SMCI",
    "CRM", "NOW", "ADBE", "ORCL", "SNOW", "PANW", "CRWD", "ZS", "DDOG", "NET",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "V", "MA", "AXP", "TFC",
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "DHR", "ISRG", "VRTX", "GILD", "BMY",
    "HD", "MCD", "SBUX", "NKE", "LOW", "TGT", "COST", "BKNG", "LULU",
    "PG", "KO", "PEP", "WMT",
    "XOM", "CVX", "COP", "SLB", "MPC", "VLO", "OXY",
    "BA", "GE", "CAT", "HON", "RTX", "LMT", "FDX", "UPS", "DE", "UAL",
    "DIS", "CMCSA", "VZ", "T",
    "COIN", "PLTR", "UBER", "RIVN", "ROKU", "HOOD", "SOFI", "PYPL", "IBM",
    "RBLX", "RCL", "ABNB",
]))


# ── State classification (mirrors breakout_scanner.py) ───────────────────────

def classify_state(pct_b: float) -> str:
    if pct_b > 100:  return "EXTENDED"
    if pct_b >= 95:  return "BREAKOUT"
    if pct_b >= 75:  return "PRE-BREAKOUT"
    if pct_b >= 40:  return "NEUTRAL"
    if pct_b >= 25:  return "WEAKENING"
    if pct_b >= 0:   return "PRE-BREAKDOWN"
    return "BREAKDOWN"


# ── Data enrichment ───────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add %B, vol_ratio, state, forward returns to a daily OHLCV dataframe."""
    df = df.copy()
    c, v = df["Close"], df["Volume"]

    sma20  = c.rolling(BB_PERIOD).mean()
    std20  = c.rolling(BB_PERIOD).std()
    upper  = sma20 + 2 * std20
    lower  = sma20 - 2 * std20
    bw     = upper - lower

    df["pct_b"]       = (c - lower) / bw * 100
    df["above_sma20"] = (c > sma20).astype(bool)
    df["above_sma50"] = (c > c.rolling(SMA50_PERIOD).mean()).astype(bool)
    df["vol_ma20"]    = v.rolling(VOL_MA).mean()
    df["vol_ratio"]   = v / df["vol_ma20"]
    df["vol_90pct"]   = df["vol_ratio"].rolling(90).quantile(VOL_PCTILE / 100)
    df["state"]       = df["pct_b"].apply(
        lambda x: classify_state(float(x)) if pd.notna(x) else None
    )

    # Forward returns (percent, not decimal)
    df["ret_1d"] = (c.shift(-1) / c - 1) * 100
    df["ret_3d"] = (c.shift(-3) / c - 1) * 100
    df["ret_5d"] = (c.shift(-5) / c - 1) * 100

    return df


# ── State machine ─────────────────────────────────────────────────────────────

def process_ticker(tk: str, df: pd.DataFrame) -> tuple[list, list, list]:
    """Simulate state machine on enriched daily bars.

    Returns three lists:
      signals     — one dict per BREAKOUT alert event
      fadings     — one dict per FADING event (BREAKOUT/EXTENDED → lower)
      setup_fails — one dict per SETUP FAILED event (PRE-BREAKOUT → dropped)
    """
    signals:     list[dict] = []
    fadings:     list[dict] = []
    setup_fails: list[dict] = []

    if len(df) < 100:
        return signals, fadings, setup_fails

    df = df.reset_index(drop=False)
    if "Date" not in df.columns:
        df = df.rename(columns={"index": "Date"})

    n = len(df)

    # State machine variables
    cur_state     = None
    state_start_i = 0     # row index when cur_state was entered
    days_in_state = 0
    # Ring buffer of (state_name, entry_row_idx) for the last 6 state transitions
    path_hist: list[tuple[str, int]] = []
    # Last BREAKOUT signal tracking (for FADING reference)
    last_sig_i     = None
    last_sig_price = None
    # Track if ticker reached EXTENDED after last signal (for Q4)
    sig_went_extended = False

    for i in range(n):
        row       = df.iloc[i]
        new_state = row.get("state")
        pct_b_val = row.get("pct_b")
        if new_state is None or pd.isna(pct_b_val):
            continue

        pct_b     = float(pct_b_val)
        vol_ratio = float(row["vol_ratio"]) if pd.notna(row["vol_ratio"]) else 0.0
        vol_90pct = float(row["vol_90pct"]) if pd.notna(row["vol_90pct"]) else float("inf")
        sma20_ok  = bool(row["above_sma20"]) if pd.notna(row["above_sma20"]) else False
        sma50_ok  = bool(row["above_sma50"]) if pd.notna(row["above_sma50"]) else False
        date_str  = str(row["Date"])[:10]
        close     = float(row["Close"])
        ret_1d    = float(row["ret_1d"]) if pd.notna(row["ret_1d"]) else None
        ret_3d    = float(row["ret_3d"]) if pd.notna(row["ret_3d"]) else None
        ret_5d    = float(row["ret_5d"]) if pd.notna(row["ret_5d"]) else None

        prev_state = cur_state

        # ── State transition ─────────────────────────────────────────────────
        if new_state != cur_state:

            # FADING: leaving BREAKOUT/EXTENDED zone (Q3, Q4)
            if prev_state in ("BREAKOUT", "EXTENDED") and new_state not in ("BREAKOUT", "EXTENDED"):
                if last_sig_i is not None and last_sig_price:
                    ret_at_fade = (close / last_sig_price - 1) * 100
                    ret_5d_hold = None
                    fi = last_sig_i + 5
                    if fi < n:
                        ret_5d_hold = (float(df.iloc[fi]["Close"]) / last_sig_price - 1) * 100
                    fadings.append({
                        "ticker":               tk,
                        "signal_date":          str(df.iloc[last_sig_i]["Date"])[:10],
                        "fading_date":          date_str,
                        "days_held_in_bo":      i - last_sig_i,
                        "went_extended":        sig_went_extended,
                        "return_at_fade_pct":   round(ret_at_fade, 3),
                        "ret_5d_from_signal_pct": round(ret_5d_hold, 3) if ret_5d_hold is not None else None,
                        "fade_beat_hold5d":     bool(ret_at_fade > ret_5d_hold) if ret_5d_hold is not None else None,
                    })
                last_sig_i         = None
                last_sig_price     = None
                sig_went_extended  = False

            # SETUP FAILED: PRE-BREAKOUT dropped without reaching BREAKOUT (Q5)
            if (prev_state == "PRE-BREAKOUT"
                    and new_state not in ("BREAKOUT", "EXTENDED", "PRE-BREAKOUT")):
                days_in_pre = i - state_start_i   # days spent in PRE-BREAKOUT
                re_entry_days = None
                for j in range(1, 6):
                    fi = i + j
                    if fi < n and df.iloc[fi]["state"] in ("PRE-BREAKOUT", "BREAKOUT", "EXTENDED"):
                        re_entry_days = j
                        break
                setup_fails.append({
                    "ticker":          tk,
                    "failed_date":     date_str,
                    "days_in_pre_bo":  days_in_pre,
                    "re_entered_5d":   re_entry_days is not None,
                    "re_entry_days":   re_entry_days,
                })

            # Commit previous state to path history
            if prev_state is not None:
                path_hist = (path_hist + [(prev_state, state_start_i)])[-6:]

            # PRE-BREAKOUT tracking reset when leaving bullish zone
            if new_state not in ("PRE-BREAKOUT", "BREAKOUT", "EXTENDED"):
                pass  # pre_bo tracked via path_hist now

            cur_state     = new_state
            state_start_i = i
            days_in_state = 1
        else:
            days_in_state += 1
            # Track if ticker reaches EXTENDED after signal (Q4)
            if new_state == "EXTENDED" and last_sig_i is not None:
                sig_went_extended = True

        # ── BREAKOUT signal (first entry day only, volume + trend confirms) ──
        if (new_state == "BREAKOUT"
                and days_in_state == 1       # only on entry day
                and vol_ratio >= vol_90pct
                and sma20_ok and sma50_ok):

            # Conviction: days spent in PRE-BREAKOUT before this BREAKOUT
            days_in_pre = 0
            for ph_state, ph_start in reversed(path_hist):
                if ph_state == "PRE-BREAKOUT":
                    days_in_pre = state_start_i - ph_start
                    break
                if ph_state not in ("BREAKOUT", "EXTENDED"):
                    break  # reached a non-bullish state — stop searching

            # Reconstruct readable path (deduplicate consecutive, last 4 transitions)
            recent = [p[0] for p in path_hist[-4:]] + [new_state]
            deduped: list[str] = []
            for s in recent:
                if not deduped or s != deduped[-1]:
                    deduped.append(s)
            path_str = "->".join(deduped)

            # 2-step flag: immediately preceded by PRE-BREAKOUT
            is_2step = prev_state == "PRE-BREAKOUT"

            # EXTENDED flag: pct_b > 100 right at signal
            went_extended_now = pct_b > 100

            signals.append({
                "ticker":         tk,
                "date":           date_str,
                "pct_b":          round(pct_b, 1),
                "vol_ratio":      round(vol_ratio, 2),
                "prev_state":     prev_state,
                "days_in_pre_bo": days_in_pre,
                "is_2step":       is_2step,
                "path":           path_str,
                "went_extended":  went_extended_now,
                "ret_1d_pct":     round(ret_1d, 3) if ret_1d is not None else None,
                "ret_3d_pct":     round(ret_3d, 3) if ret_3d is not None else None,
                "ret_5d_pct":     round(ret_5d, 3) if ret_5d is not None else None,
                "is_win_1d":      int(ret_1d > 0) if ret_1d is not None else None,
                "is_win_5d":      int(ret_5d > 0) if ret_5d is not None else None,
            })
            last_sig_i        = i
            last_sig_price    = close
            sig_went_extended = went_extended_now

    return signals, fadings, setup_fails


# ── Analysis ──────────────────────────────────────────────────────────────────

def hdr(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def pct(n, d) -> str:
    return f"{n/d*100:.1f}%" if d else "n/a"


def q1_conviction_time(df: pd.DataFrame) -> None:
    hdr("Q1 — CONVICTION TIME: days in PRE-BREAKOUT before signal")
    df = df.dropna(subset=["ret_5d_pct"])
    bins   = [-1, 0, 1, 3, 100]
    labels = ["0d (jumped)", "1d", "2-3d", "4d+"]
    df["conv_bucket"] = pd.cut(df["days_in_pre_bo"], bins=bins, labels=labels)
    g = df.groupby("conv_bucket", observed=True).agg(
        count=("ret_5d_pct", "count"),
        win_rate=("is_win_5d", "mean"),
        avg_ret_5d=("ret_5d_pct", "mean"),
        avg_ret_1d=("ret_1d_pct", "mean"),
        med_ret_5d=("ret_5d_pct", "median"),
    )
    g["win_rate"] = g["win_rate"] * 100
    print(f"\n{'Bucket':<14} {'Count':>6} {'WinRate%':>9} {'Avg5d%':>9} {'Med5d%':>9} {'Avg1d%':>8}")
    print("-" * 57)
    for label, row in g.iterrows():
        print(f"{str(label):<14} {row['count']:>6} {row['win_rate']:>8.1f}% "
              f"{row['avg_ret_5d']:>8.2f}% {row['med_ret_5d']:>8.2f}% {row['avg_ret_1d']:>7.2f}%")

    # Highlight best bucket
    best = g["avg_ret_5d"].idxmax()
    print(f"\nBEST conviction bucket by avg 5d return: {best}")
    worst = g["avg_ret_5d"].idxmin()
    print(f"WORST conviction bucket by avg 5d return: {worst}")


def q2_path_quality(df: pd.DataFrame) -> None:
    hdr("Q2 — PATH QUALITY: 2-step vs 1-step into BREAKOUT")
    df = df.dropna(subset=["ret_5d_pct"])
    g = df.groupby("is_2step").agg(
        count=("ret_5d_pct", "count"),
        win_rate=("is_win_5d", "mean"),
        avg_ret_5d=("ret_5d_pct", "mean"),
        avg_ret_1d=("ret_1d_pct", "mean"),
        med_ret_5d=("ret_5d_pct", "median"),
    )
    g["win_rate"] = g["win_rate"] * 100
    print(f"\n{'Path type':<22} {'Count':>6} {'WinRate%':>9} {'Avg5d%':>9} {'Med5d%':>9} {'Avg1d%':>8}")
    print("-" * 65)
    for is2, row in g.iterrows():
        label = "2-step (via PRE-BO)" if is2 else "1-step jump (skip PRE-BO)"
        print(f"{label:<22} {row['count']:>6} {row['win_rate']:>8.1f}% "
              f"{row['avg_ret_5d']:>8.2f}% {row['med_ret_5d']:>8.2f}% {row['avg_ret_1d']:>7.2f}%")
    if len(g) == 2:
        delta = g.loc[True, "avg_ret_5d"] - g.loc[False, "avg_ret_5d"]
        print(f"\n2-step advantage over 1-step: {delta:+.2f}% avg 5d return")


def q3_fading_value(fdf: pd.DataFrame) -> None:
    hdr("Q3 — FADING VALUE: exit at fade vs. holding to 5d")
    fdf = fdf.dropna(subset=["return_at_fade_pct", "ret_5d_from_signal_pct"])
    if fdf.empty:
        print("  No fading events with complete data.")
        return

    n_total     = len(fdf)
    n_fade_wins = (fdf["fade_beat_hold5d"] == True).sum()
    avg_at_fade = fdf["return_at_fade_pct"].mean()
    avg_5d_hold = fdf["ret_5d_from_signal_pct"].mean()
    med_at_fade = fdf["return_at_fade_pct"].median()
    med_5d_hold = fdf["ret_5d_from_signal_pct"].median()

    print(f"\n  Total FADING events:             {n_total}")
    print(f"  Avg return at fade day:          {avg_at_fade:+.2f}%")
    print(f"  Avg return at signal+5d:         {avg_5d_hold:+.2f}%")
    print(f"  Median return at fade day:       {med_at_fade:+.2f}%")
    print(f"  Median return at signal+5d:      {med_5d_hold:+.2f}%")
    print(f"  FADING beat holding 5d:          {pct(n_fade_wins, n_total)} ({n_fade_wins}/{n_total})")

    # By how many days they held the position in BREAKOUT before fading
    fdf["hold_bucket"] = pd.cut(fdf["days_held_in_bo"], [-1, 1, 3, 7, 100],
                                labels=["1d", "2-3d", "4-7d", "8d+"])
    g = fdf.groupby("hold_bucket", observed=True).agg(
        count=("return_at_fade_pct", "count"),
        avg_fade_ret=("return_at_fade_pct", "mean"),
        avg_hold_ret=("ret_5d_from_signal_pct", "mean"),
        pct_fade_wins=("fade_beat_hold5d", "mean"),
    )
    g["pct_fade_wins"] = g["pct_fade_wins"] * 100
    print(f"\n  By hold duration:")
    print(f"  {'Held':>6}  {'Count':>5}  {'AvgFade%':>9}  {'AvgHold5d%':>11}  {'FadeWin%':>9}")
    print("  " + "-" * 47)
    for label, row in g.iterrows():
        print(f"  {str(label):>6}  {row['count']:>5}  {row['avg_fade_ret']:>8.2f}%  "
              f"{row['avg_hold_ret']:>10.2f}%  {row['pct_fade_wins']:>8.1f}%")

    if avg_at_fade > avg_5d_hold:
        print(f"\nVERDICT: FADING IS a useful exit signal ({avg_at_fade:+.2f}% vs hold {avg_5d_hold:+.2f}%)")
    else:
        print(f"\nVERDICT: FADING is NOT better than holding ({avg_at_fade:+.2f}% vs hold {avg_5d_hold:+.2f}%)")


def q4_extended_fate(sdf: pd.DataFrame, fdf: pd.DataFrame) -> None:
    hdr("Q4 — EXTENDED FATE: overshot or sustained?")
    sdf = sdf.dropna(subset=["ret_5d_pct"])
    g = sdf.groupby("went_extended").agg(
        count=("ret_5d_pct", "count"),
        win_rate_5d=("is_win_5d", "mean"),
        avg_ret_5d=("ret_5d_pct", "mean"),
        avg_ret_1d=("ret_1d_pct", "mean"),
        med_ret_5d=("ret_5d_pct", "median"),
    )
    g["win_rate_5d"] = g["win_rate_5d"] * 100
    print(f"\n{'Signal type':<22} {'Count':>6} {'WinRate%':>9} {'Avg5d%':>9} {'Med5d%':>9} {'Avg1d%':>8}")
    print("-" * 65)
    for ext, row in g.iterrows():
        label = "EXTENDED at signal (%B>100)" if ext else "Normal BREAKOUT (%B 95-100)"
        print(f"{label:<22}... wait, {'ext' if ext else 'not ext'}: "
              f"{row['count']:>4}  {row['win_rate_5d']:>7.1f}%  "
              f"{row['avg_ret_5d']:>7.2f}%  {row['med_ret_5d']:>7.2f}%  {row['avg_ret_1d']:>6.2f}%")

    # Also look at FADING events: did going EXTENDED before FADING protect returns?
    fdf2 = fdf.dropna(subset=["return_at_fade_pct"])
    if not fdf2.empty:
        g2 = fdf2.groupby("went_extended").agg(
            count=("return_at_fade_pct", "count"),
            avg_ret_at_fade=("return_at_fade_pct", "mean"),
        )
        print(f"\n  FADING events by whether ticker went EXTENDED:")
        for ext, row in g2.iterrows():
            label = "Did go EXTENDED" if ext else "Never went EXTENDED"
            print(f"    {label}: n={row['count']}, avg ret at fade = {row['avg_ret_at_fade']:+.2f}%")


def q5_setup_reentry(fdf: pd.DataFrame) -> None:
    hdr("Q5 — SETUP FAILED RE-ENTRY: how often does it try again within 5d?")
    if fdf.empty:
        print("  No SETUP FAILED events.")
        return

    n_total    = len(fdf)
    n_reenter  = fdf["re_entered_5d"].sum()
    print(f"\n  Total SETUP FAILED events:       {n_total}")
    print(f"  Re-entered within 5d:            {pct(n_reenter, n_total)} ({n_reenter}/{n_total})")

    g = fdf.groupby("re_entered_5d").agg(
        count=("days_in_pre_bo", "count"),
        avg_days_in_pre=("days_in_pre_bo", "mean"),
    )
    for reenter, row in g.iterrows():
        label = "Re-entered" if reenter else "Did NOT re-enter"
        print(f"    {label}: n={row['count']}, avg days spent in PRE-BO before failure: {row['avg_days_in_pre']:.1f}d")

    # By days in PRE-BREAKOUT before failure: does conviction matter for re-entry?
    bins   = [-1, 0, 2, 5, 100]
    labels = ["0d", "1-2d", "3-5d", "6d+"]
    fdf = fdf.copy()
    fdf["pre_bo_bucket"] = pd.cut(fdf["days_in_pre_bo"], bins=bins, labels=labels)
    g2 = fdf.groupby("pre_bo_bucket", observed=True).agg(
        count=("re_entered_5d", "count"),
        reentry_rate=("re_entered_5d", "mean"),
        avg_reentry_days=("re_entry_days", "mean"),
    )
    g2["reentry_rate"] = g2["reentry_rate"] * 100
    print(f"\n  Re-entry rate by days spent in PRE-BREAKOUT before failure:")
    print(f"  {'Pre-BO days':<12} {'Failures':>8} {'ReEntry%':>9} {'AvgDaysToRe':>13}")
    print("  " + "-" * 44)
    for label, row in g2.iterrows():
        avg_days = f"{row['avg_reentry_days']:.1f}d" if not pd.isna(row["avg_reentry_days"]) else "n/a"
        print(f"  {str(label):<12} {row['count']:>8} {row['reentry_rate']:>8.1f}% {avg_days:>13}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backtest() -> None:
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=int(YEARS * 365.25) + 60)   # +60 for warm-up

    print(f"Backtest: state lifecycle across {len(TICKERS)} tickers "
          f"({start_date.date()} to {end_date.date()})")
    print("Downloading historical data in batches...")

    all_signals:     list[dict] = []
    all_fadings:     list[dict] = []
    all_setup_fails: list[dict] = []

    batch_size = 40
    batches    = [TICKERS[i:i+batch_size] for i in range(0, len(TICKERS), batch_size)]

    for b_idx, batch in enumerate(batches, 1):
        print(f"  Batch {b_idx}/{len(batches)}: {len(batch)} tickers...", end=" ", flush=True)
        try:
            raw = yf.download(
                batch,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=True,
            )
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

        if raw.empty:
            print("no data")
            continue

        is_multi = isinstance(raw.columns, pd.MultiIndex)
        n_signals = 0

        for tk in batch:
            try:
                if is_multi:
                    df = raw.xs(tk, axis=1, level=1).dropna(how="all") if tk in raw.columns.get_level_values(1) else pd.DataFrame()
                else:
                    df = raw.copy() if len(batch) == 1 else pd.DataFrame()

                if df.empty or len(df) < 100:
                    continue

                df = df.reset_index()
                df = enrich(df)

                sigs, fades, fails = process_ticker(tk, df)
                all_signals.extend(sigs)
                all_fadings.extend(fades)
                all_setup_fails.extend(fails)
                n_signals += len(sigs)

            except Exception as exc:
                pass   # skip individual ticker errors silently

        print(f"{n_signals} signals")

    if not all_signals:
        print("No signals found. Check data download.")
        return

    sdf = pd.DataFrame(all_signals)
    fdf = pd.DataFrame(all_fadings)
    xff = pd.DataFrame(all_setup_fails)

    # Trim to backtest window (exclude warm-up rows)
    cutoff = (end_date - timedelta(days=int(YEARS * 365.25))).strftime("%Y-%m-%d")
    sdf = sdf[sdf["date"] >= cutoff].copy()
    fdf = fdf[fdf["signal_date"] >= cutoff].copy() if not fdf.empty else fdf
    xff = xff[xff["failed_date"] >= cutoff].copy() if not xff.empty else xff

    # Save CSVs
    sdf.to_csv(os.path.join(OUT_DIR, "state_lifecycle_signals.csv"), index=False)
    fdf.to_csv(os.path.join(OUT_DIR, "state_lifecycle_fading.csv"),  index=False)
    xff.to_csv(os.path.join(OUT_DIR, "state_lifecycle_failed.csv"),  index=False)

    print(f"\nTotal signals:       {len(sdf)}")
    print(f"Total fading events: {len(fdf)}")
    print(f"Total setup fails:   {len(xff)}")
    print(f"Date range:          {sdf['date'].min()} to {sdf['date'].max()}")

    # ── Answer all 5 questions ───────────────────────────────────────────────
    q1_conviction_time(sdf.copy())
    q2_path_quality(sdf.copy())
    q3_fading_value(fdf.copy())
    q4_extended_fate(sdf.copy(), fdf.copy())
    q5_setup_reentry(xff.copy())

    print(f"\n{'='*65}")
    print("  DONE — results saved to backtest_results/state_lifecycle_*.csv")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_backtest()
