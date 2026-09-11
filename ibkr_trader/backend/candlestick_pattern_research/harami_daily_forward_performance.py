"""
Live forward-return tracker for the ONE validated candlestick finding:
bullish harami + downtrend context, enter next open, hold 5 trading days
(harami_backtest.py / RESEARCH_LOG.md -- 5y real backtest: +1.073% vs a
downtrend-matched baseline of +0.486%, Welch p=0.00007).

harami_scanner.py (IBKR-HaramiScanner, weekday 4:10pm ET) fires an
alert-only signal for this rule and logs a `harami_scanner_alert` entry to
oversight_log.jsonl. That log records the setup but never scores it. This
script does the scoring: it replays every real fired signal against REAL
subsequent daily bars and tracks the forward return at 3/5/10-trading-day
holds against two baselines --

  1. matched (downtrend-context) baseline: that SAME ticker's mean forward
     return, entering next-open, over every trailing day it was ALSO below
     both its SMA20 and SMA50 (the exact "what this name normally does in a
     downtrend" comparison the backtest's +0.486% came from), and
  2. plain baseline: the ticker's unconditional close-to-close forward
     return over the same trailing window.

Mirrors harami_1m_live_performance.py's discipline (incremental persist,
only recompute what's new or still "pending", safe to re-run weekly).
Entry/exit convention is copied exactly from harami_backtest.py:
  signal completes at bar i -> entry = Open[i+1] -> exit = Close[i+1+hold].

Usage: python harami_daily_forward_performance.py
"""
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
OVERSIGHT_LOG = BACKEND_DIR / "oversight_log.jsonl"
OUT_CSV = HERE / "harami_daily_performance_log.csv"

HOLDS = [3, 5, 10]                 # trading-day holds to track (5 is the validated one)
BASELINE_TRAILING_BARS = 504      # ~2 trading years of history for the matched baseline
MIN_N_FOR_CONCLUSION = 20

# harami_scanner.py writes summary as one line joined by " | ":
#   "<candle> Bullish Harami + downtrend: ISRG (2026-09-09) | Prior day: 361.31 -> 350.16 (bearish)
#    | Today: 350.21 -> 353.24 (bullish, inside prior body)
#    | Close $353.24 vs SMA20 $377.15 / SMA50 $378.87 (both below -- ...) | Backtested plan ..."
_RE = re.compile(
    r"Bullish Harami \+ downtrend:\s*(?P<ticker>[A-Z][A-Z.\-]*)\s*\((?P<date>\d{4}-\d{2}-\d{2})\)"
    r".*?Prior day:\s*(?P<po>[\d.]+)\s*->\s*(?P<pc>[\d.]+)"
    r".*?Today:\s*(?P<to>[\d.]+)\s*->\s*(?P<tc>[\d.]+)"
    r".*?SMA20\s*\$(?P<sma20>[\d.]+)\s*/\s*SMA50\s*\$(?P<sma50>[\d.]+)",
    re.DOTALL,
)


def load_signals():
    sigs = {}
    if not OVERSIGHT_LOG.exists():
        return []
    with open(OVERSIGHT_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("category") != "harami_scanner_alert":
                continue
            m = _RE.search(e.get("summary", ""))
            if not m:
                continue
            key = (m.group("ticker"), m.group("date"))
            sigs[key] = {                       # dedupe: keep last occurrence
                "ticker": m.group("ticker"),
                "signal_date": m.group("date"),
                "prior_open": float(m.group("po")), "prior_close": float(m.group("pc")),
                "today_open": float(m.group("to")), "today_close": float(m.group("tc")),
                "sma20": float(m.group("sma20")), "sma50": float(m.group("sma50")),
                "logged_at": e.get("time", ""),
            }
    return list(sigs.values())


def load_existing():
    if not OUT_CSV.exists():
        return {}
    rows = {}
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["ticker"], r["signal_date"])] = r
    return rows


def fetch_daily(ticker, cache):
    if ticker in cache:
        return cache[ticker]
    try:
        h = yf.Ticker(ticker).history(period="3y", auto_adjust=False)
        if h.empty:
            cache[ticker] = None
        else:
            h.index = pd.to_datetime([d.strftime("%Y-%m-%d") for d in h.index])
            h = h.dropna(subset=["Open", "Close"])
            cache[ticker] = h
    except Exception as exc:
        print(f"  [{ticker}] yfinance fetch failed: {exc}")
        cache[ticker] = None
    return cache[ticker]


def signal_row_index(hist, signal_date):
    """Index of the signal (harami) bar, or None if not present yet."""
    want = pd.Timestamp(signal_date)
    matches = np.where(hist.index == want)[0]
    return int(matches[0]) if len(matches) else None


def realised_return(hist, sig_idx, hold, today):
    """Enter Open[sig_idx+1], exit Close[sig_idx+1+hold] -- exactly harami_backtest.py.
    Returns (pct or None, 'ok'|'pending'|'nodata')."""
    entry_idx = sig_idx + 1
    exit_idx = entry_idx + hold
    if entry_idx >= len(hist):
        return None, "pending"                     # entry day hasn't happened
    if exit_idx >= len(hist):
        return None, "pending"                     # hold not fully elapsed
    if hist.index[exit_idx].date() > today:
        return None, "pending"
    entry_open = float(hist.iloc[entry_idx]["Open"])
    exit_close = float(hist.iloc[exit_idx]["Close"])
    if entry_open <= 0:
        return None, "nodata"
    return round((exit_close / entry_open - 1) * 100, 4), "ok"


def baselines(hist, sig_idx, hold):
    """As-of the signal bar, over the trailing window:
       matched  = mean fwd return on days the ticker was < SMA20 AND < SMA50
       plain    = mean unconditional fwd return
    Same enter-next-open / exit-close+hold convention."""
    lo = max(51, sig_idx - BASELINE_TRAILING_BARS)
    o = hist["Open"].values
    c = hist["Close"].values
    s20 = hist["Close"].rolling(20).mean().values
    s50 = hist["Close"].rolling(50).mean().values
    matched, plain = [], []
    for i in range(lo, sig_idx):                    # strictly before the signal (no lookahead)
        ei, xi = i + 1, i + 1 + hold
        if xi >= len(c) or o[ei] <= 0:
            continue
        r = (c[xi] / o[ei] - 1) * 100
        plain.append(r)
        if c[i] < s20[i] and c[i] < s50[i]:
            matched.append(r)
    mb = round(float(np.mean(matched)), 4) if len(matched) >= 5 else None
    pb = round(float(np.mean(plain)), 4) if len(plain) >= 5 else None
    return mb, len(matched), pb, len(plain)


def main():
    signals = load_signals()
    print(f"{len(signals)} real harami_scanner_alert signal(s) in oversight_log.jsonl")
    if not signals:
        print("Nothing fired yet -- nothing to score.")
        return

    existing = load_existing()
    cache = {}
    today = datetime.now(timezone.utc).date()
    out_rows = []

    for sig in sorted(signals, key=lambda s: (s["signal_date"], s["ticker"])):
        key = (sig["ticker"], sig["signal_date"])
        row = dict(existing.get(key, {}))
        row.update({
            "signal_date": sig["signal_date"], "ticker": sig["ticker"],
            "signal_open": sig["today_open"], "signal_close": sig["today_close"],
            "sma20": sig["sma20"], "sma50": sig["sma50"], "logged_at": sig["logged_at"],
        })

        need = any(row.get(f"ret_{h}d") in (None, "", "pending") for h in HOLDS) or \
               any(not row.get(f"mbase_{h}d") for h in HOLDS)
        if need:
            hist = fetch_daily(sig["ticker"], cache)
            if hist is None:
                for h in HOLDS:
                    row.setdefault(f"ret_{h}d", "pending")
                    row.setdefault(f"mbase_{h}d", "")
                    row.setdefault(f"pbase_{h}d", "")
                out_rows.append(row)
                continue
            si = signal_row_index(hist, sig["signal_date"])
            row["entry_date"] = (hist.index[si + 1].strftime("%Y-%m-%d")
                                 if si is not None and si + 1 < len(hist) else "pending")
            for h in HOLDS:
                if row.get(f"ret_{h}d") not in (None, "", "pending"):
                    pass  # already resolved on a prior run
                elif si is None:
                    row[f"ret_{h}d"] = "pending"
                else:
                    val, st = realised_return(hist, si, h, today)
                    row[f"ret_{h}d"] = val if st == "ok" else st
                if not row.get(f"mbase_{h}d") and si is not None:
                    mb, mn, pb, pn = baselines(hist, si, h)
                    row[f"mbase_{h}d"] = "" if mb is None else mb
                    row[f"mbase_{h}d_n"] = mn
                    row[f"pbase_{h}d"] = "" if pb is None else pb
        for h in HOLDS:
            row.setdefault(f"ret_{h}d", "pending")
            row.setdefault(f"mbase_{h}d", "")
            row.setdefault(f"mbase_{h}d_n", "")
            row.setdefault(f"pbase_{h}d", "")
        out_rows.append(row)

    fields = (["signal_date", "ticker", "entry_date", "signal_open", "signal_close",
               "sma20", "sma50", "logged_at"]
              + [f"ret_{h}d" for h in HOLDS]
              + [f"mbase_{h}d" for h in HOLDS] + [f"mbase_{h}d_n" for h in HOLDS]
              + [f"pbase_{h}d" for h in HOLDS])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {len(out_rows)} row(s) to {OUT_CSV}")

    print("\n=== Forward performance (only holds that have fully elapsed) ===")
    print(f"    (validated backtest, 5d: signal mean +1.073% vs matched baseline +0.486%, p=0.00007)\n")
    any_elapsed = False
    for h in HOLDS:
        vals, mbs, pbs = [], [], []
        for r in out_rows:
            v = r.get(f"ret_{h}d")
            if v in (None, "", "pending", "nodata"):
                continue
            vals.append(float(v))
            if r.get(f"mbase_{h}d") not in (None, ""):
                mbs.append(float(r[f"mbase_{h}d"]))
            if r.get(f"pbase_{h}d") not in (None, ""):
                pbs.append(float(r[f"pbase_{h}d"]))
        tag = "  <- validated hold" if h == 5 else ""
        if not vals:
            print(f"  {h:>2}d hold: no signal has completed a full {h}-day hold yet{tag}")
            continue
        any_elapsed = True
        n = len(vals)
        mean = sum(vals) / n
        win = sum(1 for v in vals if v > 0) / n * 100
        mb = f"{sum(mbs)/len(mbs):+.3f}%" if mbs else "n/a"
        pb = f"{sum(pbs)/len(pbs):+.3f}%" if pbs else "n/a"
        excess = f"{mean - sum(mbs)/len(mbs):+.3f}%" if mbs else "n/a"
        flag = "" if n >= MIN_N_FOR_CONCLUSION else f"   [N<{MIN_N_FOR_CONCLUSION}: reference only, not conclusive]"
        print(f"  {h:>2}d hold  N={n:<3} mean={mean:+.3f}%  win={win:.0f}%  | matched base {mb}  plain base {pb}  | excess vs matched {excess}{tag}{flag}")

    pend = [r for r in out_rows if r.get("ret_5d") in (None, "", "pending")]
    if pend:
        print(f"\n  {len(pend)} signal(s) still mid-hold (5d not elapsed):")
        for r in pend:
            print(f"    {r['signal_date']}  {r['ticker']:6s}  entry {r.get('entry_date','?')}  "
                  f"signal close {r.get('signal_close')}")
    if not any_elapsed:
        print("\n  Nothing has a completed hold yet -- re-run in a few trading days; the sample grows each run.")


if __name__ == "__main__":
    main()
