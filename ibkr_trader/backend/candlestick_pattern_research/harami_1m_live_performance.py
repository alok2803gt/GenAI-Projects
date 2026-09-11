"""
Live-fired 1-min bullish harami alert performance tracker.

Every time _check_harami_1m() (main.py) fires a real alert on the Signal
Trader chart, it logs a harami_1m_detected entry to oversight_log.jsonl in
addition to the Telegram push -- but that log is plain JSONL mixed with
every other strategy's decisions, not something you can eyeball for a real
answer to "does this actually work." This script pulls every real fire,
looks up REAL subsequent 1-min price action via yfinance, and tracks
forward returns at several horizons against a matched baseline (this same
ticker's own unconditional forward return on the same day, at the same
horizon -- not zero) -- the same discipline this account's other research
(squeeze momentum, daily-bar harami) already holds itself to.

IMPORTANT: this is explicitly NOT the validated daily-bar/5-day-hold
research (candlestick_pattern_research/harami_backtest.py, p=0.00007) --
that was tested on daily bars with a downtrend-context filter. This tracks
the BARE 1-min pattern with no filter, which was deployed purely for
visibility (the user's own request: "any time there is a bullish harami
candle set alarm"), not because it was expected to have an edge. This
script exists to find out, with real data, whether it does.

Pulls extended-hours data (prepost=True) to match the IBKR 1-min pipe this
alert is built on (ONE_MIN_TICKERS uses useRTH=False, so real fires can
and do happen pre/post market -- a regular-hours-only yfinance pull would
silently have gaps for those).

Persists incrementally to harami_1m_performance_log.csv (only computes
what's new, or was still "pending" -- horizon not yet elapsed -- last
run) since yfinance's ~7-day 1-min lookback window would otherwise make
older signals unrecoverable. Safe to re-run any time, e.g. weekly, to see
the sample grow.

Usage: python harami_1m_live_performance.py
"""
import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
OVERSIGHT_LOG = BACKEND_DIR / "oversight_log.jsonl"
OUT_CSV = HERE / "harami_1m_performance_log.csv"

ET = timezone(timedelta(hours=-4))  # fixed EDT offset, matches harami_scanner.py's own convention
HORIZONS_MIN = [5, 15, 30, 60]
MIN_N_FOR_CONCLUSION = 20  # below this, report the number but don't imply it means anything

_SUMMARY_RE = re.compile(
    r"1-min Bullish Harami: (?P<ticker>[A-Z]+) at (?P<time>\S+)\n"
    r"Prior candle: (?P<prior_open>[\d.]+) -> (?P<prior_close>[\d.]+) \(bearish\)\n"
    r"This candle: (?P<curr_open>[\d.]+) -> (?P<curr_close>[\d.]+) \(bullish"
)


def load_signals():
    """Parse every real harami_1m_detected entry from the durable oversight log."""
    signals = []
    if not OVERSIGHT_LOG.exists():
        return signals
    with open(OVERSIGHT_LOG, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("category") != "harami_1m_detected":
                continue
            m = _SUMMARY_RE.search(entry.get("summary", ""))
            if not m:
                continue
            signals.append({
                "signal_time": m.group("time"),
                "ticker": m.group("ticker"),
                "prior_open": float(m.group("prior_open")),
                "prior_close": float(m.group("prior_close")),
                "curr_open": float(m.group("curr_open")),
                "entry_price": float(m.group("curr_close")),
            })
    return signals


def load_existing_csv():
    if not OUT_CSV.exists():
        return {}
    rows = {}
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["signal_time"], row["ticker"])] = row
    return rows


def fetch_1m_bars(ticker, cache):
    if ticker in cache:
        return cache[ticker]
    try:
        hist = yf.Ticker(ticker).history(period="7d", interval="1m", prepost=True)
        if hist.empty:
            cache[ticker] = None
            return None
        hist = hist.tz_convert(ET) if hist.index.tz is not None else hist.tz_localize("UTC").tz_convert(ET)
        cache[ticker] = hist
    except Exception as exc:
        print(f"  [{ticker}] yfinance fetch failed: {exc}")
        cache[ticker] = None
    return cache[ticker]


def forward_return(hist, entry_time, entry_price, minutes):
    target = entry_time + timedelta(minutes=minutes)
    future = hist[hist.index >= target]
    if future.empty:
        return None  # horizon hasn't elapsed yet, or fell outside yfinance's window
    px = float(future.iloc[0]["Close"])
    return round((px / entry_price - 1) * 100, 4)


def baseline_return(hist, day_str, minutes):
    """Matched baseline: this SAME ticker's own unconditional mean forward
    return across every 1-min bar on the SAME day, at the SAME horizon --
    not a comparison against zero."""
    day_bars = hist[hist.index.strftime("%Y-%m-%d") == day_str]
    if len(day_bars) < minutes + 5:
        return None
    closes = day_bars["Close"].values
    rets = [(closes[i + minutes] / closes[i] - 1) * 100 for i in range(len(closes) - minutes)]
    return round(sum(rets) / len(rets), 4) if rets else None


def main():
    signals = load_signals()
    print(f"{len(signals)} real harami_1m_detected signal(s) in oversight_log.jsonl")
    if not signals:
        print("Nothing fired yet -- nothing to analyze.")
        return

    existing = load_existing_csv()
    yf_cache, baseline_cache = {}, {}
    rows = []
    now = datetime.now(ET)

    for sig in signals:
        key = (sig["signal_time"], sig["ticker"])
        entry_time = datetime.fromisoformat(sig["signal_time"])
        day_str = entry_time.strftime("%Y-%m-%d")

        row = dict(existing.get(key, {}))
        row.update({
            "signal_time": sig["signal_time"], "ticker": sig["ticker"],
            "prior_open": sig["prior_open"], "prior_close": sig["prior_close"],
            "curr_open": sig["curr_open"], "entry_price": sig["entry_price"],
        })

        still_pending = any(row.get(f"fwd_return_{m}m") in (None, "", "pending") for m in HORIZONS_MIN)
        if still_pending:
            hist = fetch_1m_bars(sig["ticker"], yf_cache)
            bkey = (sig["ticker"], day_str)
            for m in HORIZONS_MIN:
                field = f"fwd_return_{m}m"
                if row.get(field) not in (None, "", "pending"):
                    continue  # already resolved on a prior run
                if hist is None:
                    row[field] = "pending"
                    continue
                if now < entry_time + timedelta(minutes=m):
                    row[field] = "pending"
                    continue
                r = forward_return(hist, entry_time, sig["entry_price"], m)
                row[field] = r if r is not None else "pending"

                bfield = f"baseline_{m}m"
                if not row.get(bfield):
                    if bkey not in baseline_cache:
                        baseline_cache[bkey] = {}
                    if m not in baseline_cache[bkey]:
                        baseline_cache[bkey][m] = baseline_return(hist, day_str, m)
                    b = baseline_cache[bkey][m]
                    row[bfield] = b if b is not None else ""
        for m in HORIZONS_MIN:
            row.setdefault(f"fwd_return_{m}m", "pending")
            row.setdefault(f"baseline_{m}m", "")

        rows.append(row)

    fieldnames = (["signal_time", "ticker", "prior_open", "prior_close", "curr_open", "entry_price"]
                  + [f"fwd_return_{m}m" for m in HORIZONS_MIN]
                  + [f"baseline_{m}m" for m in HORIZONS_MIN])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {len(rows)} row(s) to {OUT_CSV}")

    print("\n=== Aggregate (only counting horizons that have actually elapsed) ===")
    for m in HORIZONS_MIN:
        field, bfield = f"fwd_return_{m}m", f"baseline_{m}m"
        vals = [float(r[field]) for r in rows if r.get(field) not in (None, "", "pending")]
        bvals = [float(r[bfield]) for r in rows if r.get(bfield) not in (None, "", "pending")]
        if not vals:
            print(f"  {m:>3}min: no elapsed signals yet")
            continue
        n = len(vals)
        mean_ret = sum(vals) / n
        win_rate = sum(1 for v in vals if v > 0) / n * 100
        base_str = f"vs matched baseline {sum(bvals)/len(bvals):+.3f}%" if bvals else "(baseline n/a)"
        flag = "" if n >= MIN_N_FOR_CONCLUSION else f"  [N<{MIN_N_FOR_CONCLUSION} -- too small to conclude anything, reference only]"
        print(f"  {m:>3}min  N={n:<4} mean={mean_ret:+.3f}%  win_rate={win_rate:.1f}%  {base_str}{flag}")


if __name__ == "__main__":
    main()
