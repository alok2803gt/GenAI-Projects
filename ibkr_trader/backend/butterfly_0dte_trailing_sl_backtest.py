"""
0DTE long butterfly backtest WITH a trailing stop-loss on the position's
real intraday mark, layered on top of butterfly_0dte_backtest_v2.py's
already-validated real entry (9:45 ET, real UW intraday tape) / real exit
(intrinsic value at the real close) methodology -- same 90-real-trading-day
window (Unusual Whales' real plan limit), same SPY/QQQ/IWM universe, same
real strike selection. This does NOT re-run or change that baseline; it
adds a real intraday mark-to-market path so a trailing stop can actually
be tested against real data, not simulated.

How the intraday path is built: each leg's FULL real 1-min intraday tape
(already fetched once per contract, same API this account's plan
supports) is pulled ONCE, then sampled at 15-minute checkpoints from
10:00 to 15:30 ET. Real option trades are SPARSE (a liquid SPY 0DTE
contract might see ~10-15 real trades across a whole day, thinner for
deep-OTM wings) -- each checkpoint forward-fills from the most recent
real trade at or before that time (or the 9:45 entry price if nothing has
traded yet). This is a real, disclosed approximation: a "mark" between
real trades is a stale price, not a guaranteed executable quote -- the
same class of approximation as using closing prices elsewhere in this
account's backtests, just intraday instead of daily.

Trailing-stop rule tested: track the running PEAK butterfly value since
entry; if the value drops to peak * (1 - trail_pct), exit at that
checkpoint's real (or forward-filled) mark. If never triggered, exit at
the real close via the exact same intrinsic-value formula as v2 -- so the
no-trigger case is identical to v2's baseline, not a different estimate.

Output: butterfly_0dte_trailing_sl_rows.csv
"""
import sys
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import butterfly_0dte_backtest_v2 as v2

TRAIL_PCTS = [0.25, 0.40, 0.60]
CHECKPOINT_MINUTES = list(range(10 * 60, 15 * 60 + 31, 15))  # 10:00 to 15:30 ET, every 15min, as minutes-since-midnight
ENTRY_MINUTE = 9 * 60 + 45  # 9:45 ET


def get_full_intraday_tape(occ_ticker: str, d: date) -> list:
    """Full real 1-min tape for one contract/day (all real trades, not just
    the one closest to 9:45 -- v2's get_morning_option_price() only kept
    the single closest match; this keeps everything for the intraday path)."""
    uw_id = occ_ticker[2:] if occ_ticker.startswith("O:") else occ_ticker
    data = v2._uw_get(f"/api/option-contract/{uw_id}/intraday", {"date": d.isoformat()})
    if not data:
        return []
    rows = []
    for row in data.get("data") or []:
        try:
            ts = datetime.fromisoformat(row["start_time"].replace("Z", "+00:00"))
        except Exception:
            continue
        rows.append((ts, float(row["close"])))
    rows.sort(key=lambda x: x[0])
    return rows


def et_minutes_to_utc(d: date, minute_of_day: int) -> datetime:
    """ET (EDT, UTC-4) minute-of-day -> real UTC datetime for that date."""
    hh, mm = divmod(minute_of_day, 60)
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=timezone.utc) + timedelta(hours=4)


def forward_fill_price(tape: list, at_time: datetime, fallback: float) -> float:
    price = fallback
    for ts, px in tape:
        if ts <= at_time:
            price = px
        else:
            break
    return price


@dataclass
class Row:
    ticker: str
    wing_step: int
    entry_date: str
    status: str
    reason: str = ""
    entry_debit: float = np.nan
    close_exit_value: float = np.nan
    close_payoff: float = np.nan
    peak_value: float = np.nan
    peak_minute: float = np.nan


def process_day_ticker(ticker: str, d: date, close_spot: float, rows: list):
    morning_spot = v2.get_morning_underlying_price(ticker, d)
    if morning_spot is None:
        return
    call_map = v2.get_contract_grid_0dte(ticker, d)
    if len(call_map) < 9:
        return
    strikes_sorted = sorted(call_map.keys())
    k2 = min(strikes_sorted, key=lambda s: abs(s - morning_spot))
    k2_pos = strikes_sorted.index(k2)

    for wing_step in [2, 4]:
        base = dict(ticker=ticker, wing_step=wing_step, entry_date=d.isoformat())
        lo_pos, hi_pos = k2_pos - wing_step, k2_pos + wing_step
        if lo_pos < 0 or hi_pos >= len(strikes_sorted):
            rows.append(Row(status="skip", reason="wing_out_of_range", **base).__dict__)
            continue
        k1, k3 = strikes_sorted[lo_pos], strikes_sorted[hi_pos]
        occ_map = {"k1": call_map[k1], "k2": call_map[k2], "k3": call_map[k3]}

        tapes = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {name: ex.submit(get_full_intraday_tape, occ, d) for name, occ in occ_map.items()}
            for name, fut in futs.items():
                tapes[name] = fut.result()

        entry_prices = {}
        for name in ("k1", "k2", "k3"):
            entry_target = et_minutes_to_utc(d, ENTRY_MINUTE)
            px = None
            best_diff = None
            for ts, p in tapes[name]:
                diff = abs((ts - entry_target).total_seconds())
                if diff <= v2.ENTRY_SEARCH_WINDOW_MIN * 60 and (best_diff is None or diff < best_diff):
                    px, best_diff = p, diff
            entry_prices[name] = px
        if any(entry_prices[k] is None for k in ("k1", "k2", "k3")):
            rows.append(Row(status="skip", reason="no_morning_option_data", **base).__dict__)
            continue
        entry_debit = entry_prices["k1"] + entry_prices["k3"] - 2 * entry_prices["k2"]
        if entry_debit <= 0.005:
            rows.append(Row(status="skip", reason="nonpositive_debit", entry_debit=entry_debit, **base).__dict__)
            continue

        # Build the real (forward-filled) intraday mark path.
        path = []
        for minute in CHECKPOINT_MINUTES:
            at_time = et_minutes_to_utc(d, minute)
            p1 = forward_fill_price(tapes["k1"], at_time, entry_prices["k1"])
            p2 = forward_fill_price(tapes["k2"], at_time, entry_prices["k2"])
            p3 = forward_fill_price(tapes["k3"], at_time, entry_prices["k3"])
            value = p1 + p3 - 2 * p2
            path.append((minute, value))

        close_exit_value = (max(0.0, close_spot - k1) - 2 * max(0.0, close_spot - k2)
                             + max(0.0, close_spot - k3))
        close_payoff = close_exit_value - entry_debit
        peak_value = max([v for _, v in path] + [entry_debit])
        peak_minute = next((m for m, v in path if v == peak_value), ENTRY_MINUTE)

        row = Row(status="ok", entry_debit=entry_debit, close_exit_value=close_exit_value,
                  close_payoff=close_payoff, peak_value=peak_value, peak_minute=peak_minute, **base)
        row_dict = row.__dict__.copy()
        for tp in TRAIL_PCTS:
            stop_level = peak_value * (1 - tp)
            triggered_value = None
            for minute, value in path:
                running_peak = max([v for m, v in path if m <= minute] + [entry_debit])
                if value <= running_peak * (1 - tp):
                    triggered_value = value
                    break
            if triggered_value is not None:
                payoff = triggered_value - entry_debit
            else:
                payoff = close_payoff
            row_dict[f"trail_{int(tp*100)}_payoff"] = payoff
            row_dict[f"trail_{int(tp*100)}_triggered"] = triggered_value is not None
        rows.append(row_dict)


def main():
    rows = []
    for ticker in ["SPY", "QQQ", "IWM"]:
        print(f"\n=== {ticker} ===", flush=True)
        earliest = v2.uw_earliest_available_date()
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=False, actions=False)
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        trading_days = [d for d in hist.index.date if earliest <= d < date.today()]
        close_map = dict(zip(hist.index.date, hist["Close"].values))

        for i, d in enumerate(trading_days, 1):
            if i % 10 == 0:
                print(f"  [{i}/{len(trading_days)}] {d}", flush=True)
            try:
                process_day_ticker(ticker, d, float(close_map[d]), rows)
            except Exception as e:
                print(f"  [ERROR] {ticker} {d}: {e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("butterfly_0dte_trailing_sl_rows.csv", index=False)
    print(f"\nSaved {len(df)} rows")
    ok = df[df.status == "ok"].copy()
    print(f"Fully priced: n={len(ok)}")
    if len(ok) == 0:
        return

    for wing_step, grp in ok.groupby("wing_step"):
        print(f"\n=== wing_step={wing_step} (n={len(grp)}) ===")
        baseline_payoff = grp["close_payoff"] * 100
        print(f"  BASELINE (hold to close, no stop): mean=${baseline_payoff.mean():.2f}  "
              f"median=${baseline_payoff.median():.2f}  win_rate={(baseline_payoff>0).mean():.1%}  "
              f"min=${baseline_payoff.min():.2f}")
        for tp in TRAIL_PCTS:
            col = f"trail_{int(tp*100)}_payoff"
            trig_col = f"trail_{int(tp*100)}_triggered"
            pay = grp[col] * 100
            trig_rate = grp[trig_col].mean()
            print(f"  TRAIL {int(tp*100)}%: mean=${pay.mean():.2f}  median=${pay.median():.2f}  "
                  f"win_rate={(pay>0).mean():.1%}  min=${pay.min():.2f}  triggered={trig_rate:.1%} of days")


if __name__ == "__main__":
    main()
