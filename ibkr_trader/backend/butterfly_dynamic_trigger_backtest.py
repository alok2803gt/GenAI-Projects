"""
Event-triggered entry, real data -- replaces the fixed-clock-time design
(9:35/9:45/10:00 snapshots, butterfly_entry_time_backtest.py) with the
better-motivated version: treat "near prior-day S/R" as the actual ENTRY
TRIGGER, not a filter checked at one arbitrary moment. Scans the real
underlying price path (Polygon 1-min bars, dense/complete -- unlike the
option tape, no forward-fill needed for this scan) from market open
through a cutoff, fires on the FIRST minute price comes within 0.3% of
the prior real trading day's high or low, and enters there -- whatever
real clock time that turns out to be, day to day.

Window scanned: 09:30 to 11:00 ET. If the trigger never fires in that
window, no trade that day (same "skip" outcome as the fixed-time design's
non-qualifying days, just decided by the live path instead of one snapshot).
Wing_step=4 throughout (the already-validated width). Real UW intraday
tape supplies the option entry prices at whatever minute the trigger
fires; real close-of-day intrinsic value is the exit, unchanged from
every other backtest in this series.

Output: butterfly_dynamic_trigger_rows.csv
"""
import sys
import io
from datetime import date, datetime, timedelta, timezone

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import yfinance as yf

import butterfly_0dte_backtest_v2 as v2

WING_STEP = 4
SCAN_START = (9, 30)
SCAN_END = (11, 0)
SR_THRESHOLD = 0.003


def et_to_utc(d, hh, mm):
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=timezone.utc) + timedelta(hours=4)


def get_minute_bars(ticker, d):
    data = v2._polygon_get(f"/v2/aggs/ticker/{ticker}/range/1/minute/{d.isoformat()}/{d.isoformat()}",
                            {"adjusted": "true", "sort": "asc", "limit": 500})
    if not data or data.get("status") not in ("OK", "OK "):
        return []
    return data.get("results") or []


def find_trigger(bars, d, prior_high, prior_low):
    start_utc = et_to_utc(d, *SCAN_START)
    end_utc = et_to_utc(d, *SCAN_END)
    for row in bars:
        ts = datetime.fromtimestamp(row["t"] / 1000, tz=timezone.utc)
        if ts < start_utc or ts > end_utc:
            continue
        px = float(row["c"])
        dist_high = abs(prior_high - px) / px
        dist_low = abs(px - prior_low) / px
        if dist_high < SR_THRESHOLD or dist_low < SR_THRESHOLD:
            return ts, px
    return None, None


def get_option_price_near(occ_ticker, d, target_utc):
    uw_id = occ_ticker[2:] if occ_ticker.startswith("O:") else occ_ticker
    data = v2._uw_get(f"/api/option-contract/{uw_id}/intraday", {"date": d.isoformat()})
    if not data:
        return None
    rows = data.get("data") or []
    best, best_diff = None, None
    for row in rows:
        try:
            ts = datetime.fromisoformat(row["start_time"].replace("Z", "+00:00"))
        except Exception:
            continue
        diff = abs((ts - target_utc).total_seconds())
        if diff <= v2.ENTRY_SEARCH_WINDOW_MIN * 60 and (best_diff is None or diff < best_diff):
            best, best_diff = row, diff
    return float(best["close"]) if best else None


def main():
    earliest = v2.uw_earliest_available_date()
    rows = []

    for ticker in v2.ETF_UNIVERSE:
        print(f"\n=== {ticker} ===", flush=True)
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=False, actions=False)
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        trading_days = [d for d in hist.index.date if earliest <= d < date.today()]
        close_map = dict(zip(hist.index.date, hist["Close"].values))

        n_no_trigger = 0
        for i, d in enumerate(trading_days, 1):
            if i % 15 == 0:
                print(f"  [{i}/{len(trading_days)}] {d} (no_trigger so far: {n_no_trigger})", flush=True)
            prior = hist[hist.index.date < d]
            if prior.empty:
                continue
            prior_high, prior_low = float(prior.iloc[-1]["High"]), float(prior.iloc[-1]["Low"])

            bars = get_minute_bars(ticker, d)
            if not bars:
                continue
            trigger_ts, trigger_px = find_trigger(bars, d, prior_high, prior_low)
            if trigger_ts is None:
                n_no_trigger += 1
                continue

            call_map = v2.get_contract_grid_0dte(ticker, d)
            if len(call_map) < 2 * WING_STEP + 1:
                continue
            strikes_sorted = sorted(call_map.keys())
            k2 = min(strikes_sorted, key=lambda s: abs(s - trigger_px))
            k2_pos = strikes_sorted.index(k2)
            lo_pos, hi_pos = k2_pos - WING_STEP, k2_pos + WING_STEP
            if lo_pos < 0 or hi_pos >= len(strikes_sorted):
                continue
            k1, k3 = strikes_sorted[lo_pos], strikes_sorted[hi_pos]

            p1 = get_option_price_near(call_map[k1], d, trigger_ts)
            p2 = get_option_price_near(call_map[k2], d, trigger_ts)
            p3 = get_option_price_near(call_map[k3], d, trigger_ts)
            if p1 is None or p2 is None or p3 is None:
                continue
            entry_debit = p1 + p3 - 2 * p2
            if entry_debit <= 0.005:
                continue

            close_spot = float(close_map[d])
            exit_value = max(0.0, close_spot - k1) - 2 * max(0.0, close_spot - k2) + max(0.0, close_spot - k3)
            payoff = exit_value - entry_debit

            trigger_et = trigger_ts - timedelta(hours=4)
            rows.append(dict(
                ticker=ticker, entry_date=d.isoformat(), trigger_time_et=trigger_et.strftime("%H:%M"),
                trigger_minute_of_day=trigger_et.hour * 60 + trigger_et.minute,
                trigger_spot=trigger_px, entry_debit=entry_debit,
                payoff_dollar=payoff * 100, win=payoff > 0,
            ))
        print(f"  {ticker}: {n_no_trigger}/{len(trading_days)} days never triggered in the 09:30-11:00 window")

    df = pd.DataFrame(rows)
    df.to_csv("butterfly_dynamic_trigger_rows.csv", index=False)
    print(f"\nSaved {len(df)} triggered-and-priced rows")
    if len(df) == 0:
        return

    print("\n=== ALL TRIGGERED DAYS ===")
    print(f"  n={len(df)} mean=${df.payoff_dollar.mean():+.2f} median=${df.payoff_dollar.median():+.2f} "
          f"win_rate={df.win.mean():.1%}")

    print("\n=== By ticker ===")
    for ticker in v2.ETF_UNIVERSE:
        g = df[df.ticker == ticker]
        if len(g) == 0:
            continue
        print(f"  {ticker}: n={len(g)} mean=${g.payoff_dollar.mean():+.2f} win_rate={g.win.mean():.1%}")

    print("\n=== Trigger time distribution ===")
    print(df.trigger_time_et.value_counts().sort_index().to_string())

    print("\n=== Early trigger (<=10:00) vs late (>10:00) ===")
    early = df[df.trigger_minute_of_day <= 10 * 60]
    late = df[df.trigger_minute_of_day > 10 * 60]
    print(f"  early (n={len(early)}): mean=${early.payoff_dollar.mean():+.2f} win_rate={early.win.mean():.1%}")
    print(f"  late  (n={len(late)}): mean=${late.payoff_dollar.mean():+.2f} win_rate={late.win.mean():.1%}")


if __name__ == "__main__":
    main()
