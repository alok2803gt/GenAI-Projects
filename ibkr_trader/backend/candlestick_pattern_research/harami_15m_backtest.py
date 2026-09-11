"""
15-minute-candle version of the validated daily harami+downtrend finding.

Same universe (112 tickers, breakout_research/universe_5y_ohlcv.pkl), same
pattern definition and same bare-vs-downtrend-context test discipline as
harami_backtest.py (find_harami_signals/backtest_ticker are reused directly
-- they're timeframe-agnostic, just operate on whatever OHLC bars they're
given). "SMA20/SMA50" here means 20/50 FIFTEEN-MINUTE bars (~5h / ~12.5h of
trading), not 20/50 days -- an intraday-scaled version of the same
"is price below its own recent trend" context.

Data: Polygon REST 15-min aggs (this account's flat-file S3 credentials
don't cover us_stocks_sip/, only us_options_opra/ -- confirmed 403 on
2026-09-10 -- so this uses the REST API instead). Free/basic tier: 2-year
lookback, ~4-4.5k bars/page with next_url pagination, and a real rate limit
(HTTP 429 after ~3 calls back-to-back -- confirmed empirically) -- paced at
one call per ~13s with exponential backoff on 429. Each ticker's raw bars
are cached to data_cache_15m/<ticker>.json so a re-run or a widened test
doesn't re-pay the download.

Bars include extended hours (Polygon's default) -- filtered to regular
trading hours (09:30-16:00 ET) before backtesting, since that's what
"15-min candle" trading normally means and extended-hours bars are thin/
gappy in a way that would distort the pattern definition (body-size
percentile, SMA) with low-volume noise.

Holds are tracked in BARS, not calendar time, and CAN cross a session
boundary (e.g. a hold of 8 bars near the close spills into the next day's
open) since the RTH frame is concatenated day-to-day with no gap markers --
same convention harami_backtest.py uses on daily bars (which always cross
a session by definition). If a headline number here looks promising,
add a same-session-only cut before trusting it.

Usage: python harami_15m_backtest.py [--tickers AAPL,MSFT,...] [--max-tickers N] [--years Y]
"""
import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from harami_backtest import find_harami_signals, backtest_ticker, BODY_LOOKBACK  # noqa: E402

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
CACHE_DIR = HERE / "data_cache_15m"
CACHE_DIR.mkdir(exist_ok=True)
OUT_SUMMARY = HERE / "harami_15m_backtest_summary.csv"
OUT_PER_TICKER = HERE / "harami_15m_per_ticker.csv"

ET = ZoneInfo("America/New_York")
HOLD_BARS = [1, 2, 4, 8, 13, 26]     # 15m, 30m, 1h, 2h, ~3.25h, ~1 RTH day
LOOKBACK_YEARS = 1                    # default; this key allows up to 2 (--years to override)
CALL_PACE_S = 13.5                    # ~5 req/min free-tier pace, with margin
MAX_RETRIES = 6


def _cfg():
    with open(BACKEND_DIR / "scanner_config.json") as f:
        return json.load(f)


def _get_json(url, api_key):
    full = url + ("&" if "?" in url else "?") + "apiKey=" + api_key
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(full, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(60, 5 * (attempt + 1))
                print(f"    429, backing off {wait}s (attempt {attempt+1}/{MAX_RETRIES})", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"gave up after {MAX_RETRIES} retries: {url}")


def fetch_15m_raw(ticker, api_key, years=None):
    """Paginated fetch, cached to disk. Returns list of {t,o,h,l,c,v} dicts."""
    years = years or LOOKBACK_YEARS
    cache_file = CACHE_DIR / f"{ticker}_{years}y.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if data.get("bars"):
                return data["bars"]
        except Exception:
            pass

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=int(365.25 * years))
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/15/minute/"
           f"{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=50000")
    bars = []
    page = 0
    while url:
        d = _get_json(url, api_key)
        time.sleep(CALL_PACE_S)
        res = d.get("results") or []
        bars.extend(res)
        page += 1
        url = d.get("next_url")
    cache_file.write_text(json.dumps({"ticker": ticker, "fetched_at": datetime.now(timezone.utc).isoformat(),
                                      "n_pages": page, "bars": bars}))
    print(f"  [{ticker}] fetched {len(bars)} raw bars across {page} page(s)", flush=True)
    return bars


def to_rth_df(bars):
    if not bars:
        return None
    df = pd.DataFrame(bars)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df[["ts", "Open", "High", "Low", "Close", "Volume"]].sort_values("ts")
    tod = df["ts"].dt.time
    rth = df[(tod >= pd.Timestamp("09:30").time()) & (tod < pd.Timestamp("16:00").time())]
    rth = rth.drop_duplicates(subset="ts").reset_index(drop=True)
    return rth if len(rth) > BODY_LOOKBACK + 60 else None


def load_universe(limit=None, only=None):
    import pickle
    u = pickle.load(open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb"))
    tickers = sorted(u.keys())
    if only:
        tickers = [t for t in tickers if t in only]
    if limit:
        tickers = tickers[:limit]
    return tickers


def baseline_returns_15m(all_dfs, hold):
    rets = []
    for ticker, df in all_dfs.items():
        c = df["Close"].reset_index(drop=True)
        fwd = (c.shift(-hold) / c - 1) * 100
        rets.append(fwd.dropna())
    return pd.concat(rets) if rets else pd.Series(dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None, help="comma-separated subset")
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--years", type=float, default=LOOKBACK_YEARS)
    args = ap.parse_args()

    only = set(args.tickers.split(",")) if args.tickers else None
    tickers = load_universe(limit=args.max_tickers, only=only)
    print(f"Universe: {len(tickers)} tickers, {args.years}y lookback, 15-min RTH bars\n", flush=True)

    cfg = _cfg()
    api_key = cfg["polygon_api_key"]

    dfs = {}
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t}", flush=True)
        try:
            raw = fetch_15m_raw(t, api_key, years=args.years)
            df = to_rth_df(raw)
            if df is not None:
                dfs[t] = df
            else:
                print(f"  [{t}] insufficient RTH bars, skipping", flush=True)
        except Exception as exc:
            print(f"  [{t}] FAILED: {exc}", flush=True)

    print(f"\nLoaded usable 15-min RTH data for {len(dfs)}/{len(tickers)} tickers.\n", flush=True)
    if not dfs:
        print("Nothing to backtest.", flush=True)
        return

    per_ticker_rows = []
    summary_rows = []
    for hold in HOLD_BARS:
        agg = {"bare": [], "downtrend": []}
        for ticker, df in dfs.items():
            d = df.reset_index(drop=True)
            for label, req_dt in (("bare", False), ("downtrend", True)):
                try:
                    trades = backtest_ticker(d, hold, req_dt)
                except Exception:
                    trades = []
                agg[label].extend(trades)
                if label == "downtrend":
                    if trades:
                        arr = np.array(trades)
                        per_ticker_rows.append({
                            "hold_bars": hold, "ticker": ticker, "n": len(arr),
                            "mean_pct": round(float(arr.mean()), 4),
                            "win_pct": round(float((arr > 0).mean() * 100), 2),
                        })
        base = baseline_returns_15m(dfs, hold)
        for label in ("bare", "downtrend"):
            r = pd.Series(agg[label])
            if r.empty:
                continue
            # Welch t-test vs baseline (scipy if available, else manual)
            try:
                from scipy import stats
                _, p = stats.ttest_ind(r, base, equal_var=False)
            except Exception:
                p = float("nan")
            summary_rows.append({
                "hold_bars": hold, "variant": label, "n": len(r),
                "mean_pct": round(float(r.mean()), 4), "median_pct": round(float(r.median()), 4),
                "win_pct": round(float((r > 0).mean() * 100), 2),
                "baseline_n": len(base), "baseline_mean_pct": round(float(base.mean()), 4),
                "welch_p": round(float(p), 6) if p == p else None,
            })

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(per_ticker_rows).to_csv(OUT_PER_TICKER, index=False)

    print(f"{'='*100}", flush=True)
    print(f"{'hold':>6} {'variant':>10} {'n':>8} {'mean%':>9} {'median%':>9} {'win%':>7}   "
          f"{'base_n':>8} {'base_mean%':>11} {'welch_p':>10}")
    for row in summary_rows:
        print(f"{row['hold_bars']:>6} {row['variant']:>10} {row['n']:>8} {row['mean_pct']:>9.3f} "
              f"{row['median_pct']:>9.3f} {row['win_pct']:>6.1f}%   {row['baseline_n']:>8} "
              f"{row['baseline_mean_pct']:>11.3f} {('%.2g' % row['welch_p']) if row['welch_p'] is not None else '-':>10}")
    print(f"\nWrote {OUT_SUMMARY.name} and {OUT_PER_TICKER.name}", flush=True)


if __name__ == "__main__":
    main()
