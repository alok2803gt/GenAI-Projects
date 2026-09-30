"""
Daily implied-volatility archiver -- real historical IV for every ticker in
the shared universe (same list as calc_gex_vex.py / safe-income-screener),
pulled directly from IBKR via whatToShow="OPTION_IMPLIED_VOLATILITY" on the
underlying STOCK contract.

Confirmed live 2026-09-16 (QQQ): this returns a genuine daily OHLC-style IV
series with ZERO options-chain scanning -- one reqHistoricalData call per
ticker against the stock contract, no strikes/rights to walk. That's why
this is its own script rather than folded into calc_gex_vex.py: GEX/VEX
needs the full option chain (the ~2hr full-universe cost), IV history needs
none of that, so bolting it on would make every GEX run pay for IV work it
doesn't need. Standalone, this is fast enough to run for the full ~112-
ticker universe in a couple of minutes.

Two modes:
  --backfill "3 M"   one-time historical seed using any IBKR duration string
                     (e.g. "3 M", "6 M", "1 Y") -- gets real history immediately
                     instead of waiting weeks for the daily mode to build it up.
  (default)          today's single daily bar; meant to run once after close
                     so the day's IV bar is final, not mid-session.

Idempotent: skips any (ticker, date) pair already present in iv_history.jsonl,
so re-running (a retry, a backfill overlapping with daily runs) never creates
duplicate rows -- same reasoning as calc_gex_vex.py's incremental-write fix,
just via dedup instead of merge-on-write since this is append-only per-day data.

Usage: python archive_iv_daily.py [--tickers SPY,QQQ] [--backfill "3 M"]
No --tickers = full universe.
"""
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from ib_insync import IB, Stock

ET = timezone(timedelta(hours=-4))
_SKILL_DIR = Path.home() / ".claude" / "skills" / "gex-vex-calculator"
sys.path.insert(0, str(_SKILL_DIR))
from calc_gex_vex import UNIVERSE  # noqa: E402

HISTORY_FILE = "iv_history.jsonl"
TWS_PORT = 7496
CLIENT_ID = 1851  # dedicated -- avoids colliding with any other live clientId in this codebase


def load_existing_keys():
    path = Path(HISTORY_FILE)
    if not path.exists():
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                keys.add((row["ticker"], row["date"]))
            except Exception:
                continue
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None, help="comma-separated subset; default = full universe")
    ap.add_argument("--backfill", default=None,
                     help='IBKR duration string for a one-time historical seed, e.g. "3 M"')
    args = ap.parse_args()

    tickers = args.tickers.split(",") if args.tickers else list(UNIVERSE.keys())
    duration = args.backfill or "1 D"
    existing = load_existing_keys()

    print(f"Archiving IV history for {len(tickers)} tickers (duration={duration}, "
          f"{len(existing)} (ticker,date) rows already on file)...")

    ib = IB()
    ib.errorEvent += lambda *a: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)

    written = skipped = failed = 0
    with open(HISTORY_FILE, "a") as f:
        for i, ticker in enumerate(tickers):
            try:
                stk = Stock(ticker, "SMART", "USD")
                ib.qualifyContracts(stk)
                bars = ib.reqHistoricalData(
                    stk, endDateTime="", durationStr=duration, barSizeSetting="1 day",
                    whatToShow="OPTION_IMPLIED_VOLATILITY", useRTH=True, formatDate=1, timeout=30,
                )
                new_here = 0
                for b in bars:
                    key = (ticker, str(b.date))
                    if key in existing:
                        skipped += 1
                        continue
                    entry = {
                        "ticker": ticker, "date": str(b.date),
                        "iv_close": round(b.close, 4), "iv_high": round(b.high, 4), "iv_low": round(b.low, 4),
                        "archived_at": datetime.now(ET).isoformat(),
                    }
                    f.write(json.dumps(entry) + "\n")
                    f.flush()  # incremental -- a crash/timeout partway through the universe loses nothing already written
                    existing.add(key)
                    written += 1
                    new_here += 1
                print(f"  [{i+1}/{len(tickers)}] {ticker}: {len(bars)} bar(s), {new_here} new")
            except Exception as e:
                failed += 1
                print(f"  [{i+1}/{len(tickers)}] {ticker}: FAILED ({e})")

    ib.disconnect()
    print(f"\nWrote {written} new IV bars ({skipped} already on file, {failed} tickers failed) to {HISTORY_FILE}")


if __name__ == "__main__":
    main()
