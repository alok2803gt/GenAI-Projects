"""
Live scanner for the Memory/Storage sector-catalyst laggard-catchup setup
(real research: sector_catalyst_research/RESEARCH_LOG.md, motivated by the
CEO's real 2026-09-03 MU $1000/$1020 call spread, paid $71, MU ran
958->1017 the next day, ~10x). Validated finding: when >=2 of
MU/WDC/STX/SNDK pop >=3% same day, the peer(s) that HAVEN'T popped yet show
a real, backtested tail-move edge over the next 1-2 days -- a ~4% OTM /
2%-wider short-DTE call debit spread on the laggard was solidly profitable
across a realistic vol-assumption range, while the same structure bought
on a random day was NOT (see phase2_pnl_test.py). Backtest used the SIGNAL
DAY's CLOSE as entry -- this scanner runs INTRADAY so there's still time
to act before that close, rather than alerting after the fact.

DELIBERATELY MANUAL, per CEO's 2026-09-08 choice (AskUserQuestion): this
sends a Telegram alert with a candidate setup and REAL live quotes for
review -- it does NOT place any order. Does not generalize beyond
Memory/Storage -- 9 of 14 other candidate sector baskets tested showed no
real edge (see phase1_multi_basket_sweep.py); only add a basket here after
it separately clears the same two-phase validation.

State file (sector_catalyst_scanner_state.json) dedupes alerts so the same
basket-hot-day doesn't re-fire every poll once triggered.
"""
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yfinance as yf

from butterfly_babysitter_common import telegram
from alpaca_0dte_common import load_config

HERE = Path(__file__).parent
STATE_FILE = HERE / "sector_catalyst_scanner_state.json"
ET = timezone(timedelta(hours=-4))

BASKETS = {
    "Memory/Storage": ["MU", "WDC", "STX", "SNDK"],
}

POP_RET_THRESH = 0.03
MIN_MOVERS = 2
POLL_INTERVAL_S = 900  # 15 min -- frequent enough to catch an intraday pop with time to act,
                         # not so frequent it hammers yfinance or produces noisy re-checks
LONG_OTM_PCT = 0.04     # matches the validated backtest grid's best-balanced cell
SHORT_OTM_EXTRA_PCT = 0.02


def now_et():
    return datetime.now(ET)


def is_market_hours(ts):
    if ts.weekday() >= 5:
        return False
    return ts.replace(hour=9, minute=35) <= ts <= ts.replace(hour=15, minute=50)


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"alerted_dates": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def prior_close_and_vol_baseline(ticker):
    hist = yf.Ticker(ticker).history(period="30d", interval="1d")
    if len(hist) < 21:
        return None, None
    hist = hist.iloc[:-1]  # exclude today's still-forming bar
    prior_close = hist["Close"].iloc[-1]
    vol_avg20 = hist["Volume"].tail(20).mean()
    return prior_close, vol_avg20


def live_snapshot(ticker):
    t = yf.Ticker(ticker)
    fi = t.fast_info
    price = fi.get("lastPrice") or fi.get("last_price")
    return price


def suggest_spread(ticker, spot):
    long_k = round(spot * (1 + LONG_OTM_PCT), 0)
    short_k = round(long_k * (1 + SHORT_OTM_EXTRA_PCT), 0)
    try:
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return long_k, short_k, None, None
        expiry = expiries[0]  # nearest -- for a 0-2 DTE signal, first listed expiry is what we want
        chain = t.option_chain(expiry).calls
        long_row = chain.iloc[(chain["strike"] - long_k).abs().argsort()[:1]]
        short_row = chain.iloc[(chain["strike"] - short_k).abs().argsort()[:1]]
        long_bid_ask = (float(long_row["bid"].iloc[0]), float(long_row["ask"].iloc[0]))
        short_bid_ask = (float(short_row["bid"].iloc[0]), float(short_row["ask"].iloc[0]))
        return float(long_row["strike"].iloc[0]), float(short_row["strike"].iloc[0]), long_bid_ask, short_bid_ask
    except Exception as exc:
        print(f"  (could not pull live option chain for {ticker}: {exc})")
        return long_k, short_k, None, None


def check_basket(name, tickers, state, cfg):
    today_str = now_et().strftime("%Y-%m-%d")
    already = state["alerted_dates"].get(name) == today_str

    movers, laggards = [], []
    for t in tickers:
        prior_close, _ = prior_close_and_vol_baseline(t)
        price = live_snapshot(t)
        if prior_close is None or price is None:
            continue
        ret = price / prior_close - 1
        (movers if ret >= POP_RET_THRESH else laggards).append((t, ret, price))

    print(f"[{now_et().strftime('%H:%M:%S')}] {name}: movers={[(t, f'{r*100:.1f}%') for t,r,_ in movers]}  "
          f"laggards={[t for t,_,_ in laggards]}")

    if len(movers) < MIN_MOVERS or not laggards or already:
        return

    lines = [f"\U0001F4E1 Sector-catalyst setup: {name}",
             f"Popped today ({len(movers)}): " + ", ".join(f"{t} {r*100:+.1f}%" for t, r, _ in movers),
             f"Laggard candidate(s) (haven't moved yet -- real backtested edge on THESE, not the movers):"]
    for t, ret, spot in laggards:
        long_k, short_k, long_ba, short_ba = suggest_spread(t, spot)
        lines.append(f"  {t} @ ${spot:.2f} ({ret*100:+.1f}% today) -- suggested call debit spread "
                     f"{long_k:.0f}/{short_k:.0f} (nearest expiry)")
        if long_ba and short_ba:
            debit_bid = round(long_ba[0] - short_ba[1], 2)
            debit_ask = round(long_ba[1] - short_ba[0], 2)
            lines.append(f"    live quotes: long {long_k:.0f}C bid/ask ${long_ba[0]:.2f}/${long_ba[1]:.2f}, "
                         f"short {short_k:.0f}C bid/ask ${short_ba[0]:.2f}/${short_ba[1]:.2f} "
                         f"-> spread cost ~${debit_bid:.2f}-${debit_ask:.2f}")
        else:
            lines.append(f"    (live option quotes unavailable -- verify real price in broker before acting)")
    lines.append("Backtested edge is real but modeled (Black-Scholes, no historical options data) -- "
                 "manual review only, this does not place any order.")
    text = "\n".join(lines)
    print(text)
    telegram(cfg, text, high_priority=False)

    entry = {
        "time": datetime.now(timezone.utc).isoformat(), "actor": "trader",
        "category": "sector_catalyst_alert", "summary": text.replace("\n", " | "),
        "rationale": "Memory/Storage laggard-catchup signal fired live (validated research, 2026-09-08).",
        "outcome": "Alert sent, manual review only, no order placed.", "pnl_impact": None,
    }
    with open(HERE / "oversight_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    state["alerted_dates"][name] = today_str
    save_state(state)


def main():
    cfg = load_config()
    print("Sector-catalyst scanner starting. Baskets:", list(BASKETS.keys()))
    while True:
        now = now_et()
        if is_market_hours(now):
            state = load_state()
            for name, tickers in BASKETS.items():
                try:
                    check_basket(name, tickers, state, cfg)
                except Exception as exc:
                    print(f"  ERROR checking {name}: {exc}")
        else:
            print(f"[{now.strftime('%H:%M:%S')}] outside market hours, sleeping")
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
