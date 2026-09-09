"""
CEO asked how to leverage order-book data. Full Level 2 depth (multiple
price levels/sizes) is NOT historically retrievable via any tool this
account has -- confirmed live 2026-09-07 for GEX, and the same applies to
depth snapshots generally. But real historical TOP-OF-BOOK bid/ask IS
retrievable via Alpaca's StockQuotesRequest, confirmed live back to at
least 2026-04-28 (the earliest date in the existing candidate dataset) --
this is genuinely backtestable, unlike GEX.

This tests the single most actionable order-book-derived number: the real
bid-ask spread at the exact moment each candidate confirmed. The earlier
slippage investigation (2026-09-06) found real spread-widening was the
actual driver of the worst exit fills (VEEV: real $2.14 spread, GNRC:
real $1.46) -- this asks the same question on the ENTRY side: does a wide
spread AT CONFIRMATION predict a worse trade, and would filtering on it
improve real results?

Method: for each of the real baseline-confirmed candidates (0.35%+RVOL,
full population), fetch the REAL Alpaca quote at the exact confirmation
minute, compute real spread as % of price, then test whether filtering
out wide-spread confirmations changes real win rate/avg return.
"""
import csv
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest

import sys
sys.path.insert(0, r"C:\Projects\GenAI-Projects\ibkr_trader\backend")
from alpaca_0dte_common import load_config

HERE = Path(__file__).parent
CONFIRM_PCT = 0.35
RVOL_THRESHOLD = 1.2
CONFIRM_WINDOW_MIN = 60
TRAIL_PCT = 0.2
ET_OFFSET_HOURS = 4  # EDT; real UTC offset for market-hours minute reconstruction

cfg = load_config()
client = StockHistoricalDataClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"])


def load_all_bars() -> dict:
    with open(HERE / "minute_bars.json") as f:
        bars = json.load(f)
    p = HERE / "minute_bars_incremental.json"
    if p.exists():
        with open(p) as f:
            bars.update(json.load(f))
    return bars


def simulate(bars, avg_daily_vol):
    day_open = bars[0]["open"]
    confirm_price = day_open * (1 + CONFIRM_PCT / 100)
    watch_end = min(CONFIRM_WINDOW_MIN, len(bars))
    cum_vol = 0.0
    for i in range(0, watch_end):
        b = bars[i]
        cum_vol += b["volume"]
        expected_by_now = avg_daily_vol * (i + 1) / 390.0
        rvol = cum_vol / expected_by_now if expected_by_now > 0 else 0.0
        if b["close"] >= confirm_price and rvol >= RVOL_THRESHOLD:
            entry_idx, entry_px = i, b["close"]
            running_high = entry_px
            for b2 in bars[i + 1:]:
                running_high = max(running_high, b2["high"])
                stop_px = running_high * (1 - TRAIL_PCT / 100)
                if b2["low"] <= stop_px:
                    return {"entry_minute": i, "entry_px": entry_px, "ret_pct": (stop_px / entry_px - 1) * 100}
            exit_px = bars[-1]["close"]
            return {"entry_minute": i, "entry_px": entry_px, "ret_pct": (exit_px / entry_px - 1) * 100}
    return None


def real_spread_at(ticker: str, date_str: str, entry_minute: int) -> float | None:
    """Real Alpaca bid-ask spread (% of mid) at the confirmation minute."""
    y, m, d = date_str.split("-")
    session_open_et = datetime(int(y), int(m), int(d), 9, 30, tzinfo=timezone.utc) + timedelta(hours=ET_OFFSET_HOURS)
    ts = session_open_et + timedelta(minutes=entry_minute)
    try:
        req = StockQuotesRequest(symbol_or_symbols=[ticker], start=ts, end=ts + timedelta(seconds=30), limit=5)
        quotes = client.get_stock_quotes(req)
        df = quotes.df
        if df.empty:
            return None
        row = df.iloc[0]
        bid, ask = float(row["bid_price"]), float(row["ask_price"])
        if bid <= 0 or ask <= 0:
            return None
        mid = (bid + ask) / 2
        return (ask - bid) / mid * 100
    except Exception:
        return None


def _agg(results):
    if not results:
        return {"n": 0}
    rets = [r["ret_pct"] for r in results]
    wins = [r for r in rets if r > 0]
    return {"n": len(results), "win_rate_pct": round(len(wins) / len(results) * 100, 1),
            "avg_ret_pct": round(sum(rets) / len(results), 4)}


def main():
    bar_data = load_all_bars()
    rows = list(csv.DictReader(open(HERE / "candidates.csv")))
    avgvol = json.loads((HERE / "confirm_volume_avgvol_cache.json").read_text())

    confirmed = []
    for r in rows:
        key = f"{r['ticker']}:{r['date']}"
        bars = bar_data.get(key)
        av = avgvol.get(r["ticker"])
        if not bars or len(bars) < 65 or av is None:
            continue
        res = simulate(bars, av)
        if res is not None:
            confirmed.append({"ticker": r["ticker"], "date": r["date"], **res})

    print(f"Real confirmed candidates: {len(confirmed)}")
    print("Fetching real spread at each confirmation minute (real Alpaca quotes)...")
    for i, c in enumerate(confirmed):
        c["spread_pct"] = real_spread_at(c["ticker"], c["date"], c["entry_minute"])
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(confirmed)}")
        time.sleep(0.05)

    n_missing = sum(1 for c in confirmed if c["spread_pct"] is None)
    print(f"Missing real spread data for {n_missing}/{len(confirmed)} candidates")

    with_spread = [c for c in confirmed if c["spread_pct"] is not None]
    with_spread.sort(key=lambda c: c["spread_pct"])
    spreads = [c["spread_pct"] for c in with_spread]
    print(f"\nReal spread distribution: min={min(spreads):.4f}% p25={spreads[len(spreads)//4]:.4f}% "
          f"median={spreads[len(spreads)//2]:.4f}% p75={spreads[3*len(spreads)//4]:.4f}% max={max(spreads):.4f}%")

    agg_all = _agg(with_spread)
    print(f"\nALL (with real spread data): n={agg_all['n']} win={agg_all['win_rate_pct']}% avg={agg_all['avg_ret_pct']}%")

    for pctile, label in [(0.25, "tightest quartile"), (0.5, "tightest half"), (0.75, "widest quartile excluded")]:
        cutoff_idx = int(len(with_spread) * pctile)
        subset = with_spread[:cutoff_idx] if pctile < 0.75 else with_spread[:int(len(with_spread) * 0.75)]
        agg = _agg(subset)
        cutoff_val = spreads[cutoff_idx - 1] if cutoff_idx > 0 else 0
        print(f"  {label} (spread<={cutoff_val:.4f}%): n={agg['n']} win={agg['win_rate_pct']}% avg={agg['avg_ret_pct']}%")

    widest_quartile = with_spread[int(len(with_spread) * 0.75):]
    agg_wide = _agg(widest_quartile)
    print(f"  WIDEST quartile only: n={agg_wide['n']} win={agg_wide['win_rate_pct']}% avg={agg_wide['avg_ret_pct']}%")

    out = {"all": agg_all, "widest_quartile": agg_wide,
           "spread_distribution": {"min": min(spreads), "median": spreads[len(spreads)//2], "max": max(spreads)}}
    (HERE / "spread_at_confirm_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote spread_at_confirm_results.json")

    # Per-candidate data for robustness checks -- not saved in the first run
    with open(HERE / "spread_at_confirm_per_candidate.csv", "w", newline="") as f:
        import csv as csv_mod
        w = csv_mod.DictWriter(f, fieldnames=["ticker", "date", "entry_minute", "entry_px", "ret_pct", "spread_pct"])
        w.writeheader()
        for c in confirmed:
            w.writerow({k: c.get(k) for k in ["ticker", "date", "entry_minute", "entry_px", "ret_pct", "spread_pct"]})
    print(f"Wrote spread_at_confirm_per_candidate.csv ({len(confirmed)} rows)")


if __name__ == "__main__":
    main()
