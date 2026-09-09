"""
Live, alert-only daily scanner for the ONE validated finding from
candlestick_pattern_research/: bullish harami + downtrend context (price
below SMA20 AND SMA50), held 5 trading days. Real backtest (5y, 112-ticker
universe): mean +1.073% vs a downtrend-matched baseline of +0.486%
(Welch t-test p=0.00007), and the direction was positive in 76/111 tickers
(binomial test p=0.000062 vs the 50% pure-chance null) -- broad-based, not
a few lucky names. 10-day and 20-day holds were tested too and were
directionally consistent but materially weaker (p=0.033, p=0.050) --
this scanner deliberately only alerts the 5-day version, the one that
actually cleared a real bar.

One-shot daily run (NOT a continuous poller like sector_catalyst_scanner.py)
-- the pattern can only be confirmed once today's daily candle is final, so
this is scheduled once per weekday shortly after the close, matching
goog_condor_monday_entry.py's one-shot convention rather than
sector_catalyst_scanner.py's intraday-polling one.

Alert-only, matches the CEO's own stated preference (2026-09-08,
AskUserQuestion): flags a real setup with real price levels and the
backtested plan for manual review. Places no orders.
"""
import json
import pickle
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yfinance as yf

_BACKEND_DIR = str(Path(__file__).parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)  # butterfly_babysitter_common/alpaca_0dte_common live one level up

from butterfly_babysitter_common import telegram
from alpaca_0dte_common import load_config

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
STATE_FILE = HERE / "harami_scanner_state.json"
ET = timezone(timedelta(hours=-4))


def _load_universe():
    """Loads the REAL backtested universe from the same pickle file the
    research used, rather than a hand-typed list -- a manually-retyped
    ticker list is a real, easy way to silently drift from what was
    actually validated (caught 2026-09-09: a first hand-typed attempt at
    this list was already wrong -- missing GLD/META, and included 4
    tickers -- ADI/AMGN/ADP/AON -- that were never in the real universe
    at all)."""
    with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
        return sorted(pickle.load(f).keys())


UNIVERSE = _load_universe()
BODY_LOOKBACK = 60
HOLD_DAYS = 5


def now_et():
    return datetime.now(ET)


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"alerted": {}}   # {ticker: last_alerted_date}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def check_ticker(ticker):
    hist = yf.Ticker(ticker).history(period="4mo", interval="1d")
    if len(hist) < BODY_LOOKBACK + 5:
        return None
    hist = hist.tz_localize(None) if hist.index.tz is None else hist.tz_convert(None)

    today = hist.iloc[-1]
    prior = hist.iloc[-2]
    today_open, today_close = today["Open"], today["Close"]
    prior_open, prior_close = prior["Open"], prior["Close"]

    prior_bearish = prior_close < prior_open
    today_bullish = today_close > today_open
    contained = (today_open > prior_close) and (today_close < prior_open)
    if not (prior_bearish and today_bullish and contained):
        return None

    bodies = (hist["Close"] - hist["Open"]).abs()
    body_median = bodies.iloc[-(BODY_LOOKBACK + 1):-1].median()
    prior_body = abs(prior_close - prior_open)
    if prior_body < body_median:
        return None

    sma20 = hist["Close"].tail(20).mean()
    sma50 = hist["Close"].tail(50).mean()
    if not (today_close < sma20 and today_close < sma50):
        return None

    return {
        "ticker": ticker, "date": hist.index[-1].strftime("%Y-%m-%d"),
        "prior_open": round(prior_open, 2), "prior_close": round(prior_close, 2),
        "today_open": round(today_open, 2), "today_close": round(today_close, 2),
        "sma20": round(sma20, 2), "sma50": round(sma50, 2),
    }


def main():
    et = now_et()
    if et.weekday() >= 5:
        print("Weekend, nothing to do.")
        return
    today_str = et.strftime("%Y-%m-%d")
    state = load_state()
    cfg = load_config()

    hits = []
    for ticker in UNIVERSE:
        if state["alerted"].get(ticker) == today_str:
            continue
        try:
            hit = check_ticker(ticker)
        except Exception as exc:
            print(f"  {ticker}: error {exc}")
            continue
        if hit:
            hits.append(hit)
            state["alerted"][ticker] = today_str

    print(f"Scanned {len(UNIVERSE)} tickers, {len(hits)} real bullish-harami+downtrend hit(s) today.")
    for h in hits:
        text = (
            f"\U0001F56F️ Bullish Harami + downtrend: {h['ticker']} ({h['date']})\n"
            f"Prior day: {h['prior_open']:.2f} -> {h['prior_close']:.2f} (bearish)\n"
            f"Today: {h['today_open']:.2f} -> {h['today_close']:.2f} (bullish, inside prior body)\n"
            f"Close ${h['today_close']:.2f} vs SMA20 ${h['sma20']:.2f} / SMA50 ${h['sma50']:.2f} (both below -- real downtrend context)\n"
            f"Backtested plan (5y, real, p=0.00007 vs matched baseline): enter at tomorrow's open, "
            f"hold {HOLD_DAYS} trading days. Manual review only, no order placed."
        )
        print(text)
        telegram(cfg, text)
        entry = {
            "time": datetime.now(timezone.utc).isoformat(), "actor": "trader",
            "category": "harami_scanner_alert", "summary": text.replace("\n", " | "),
            "rationale": "Bullish harami + downtrend context fired live (validated research, 2026-09-09).",
            "outcome": "Alert sent, manual review only, no order placed.", "pnl_impact": None,
        }
        with open(BACKEND_DIR / "oversight_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    save_state(state)


if __name__ == "__main__":
    main()
