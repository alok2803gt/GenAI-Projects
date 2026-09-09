"""
Shadow filter -- tags REAL, LIVE breakout_scanner alerts with a 5-level
quality tier as they fire, in real time. Read-only against alert_history
(breakout_scanner's own real table) -- never writes to it, never touches
breakout_scanner.py or its config. This is a live, forward, out-of-sample
validation layer running ALONGSIDE the real scanner, not a replacement for
it and not wired into anything that trades.

Telegram: sends ONE digest message per poll run (never one per alert, same
flood-avoidance convention as darkpool_activity_monitor.py) containing only
PREFER-HIGH alerts, per CEO request 2026-08-25. AVOID/NEUTRAL/PREFER-LOW/
PREFER-MID are still logged to shadow_filter_log.csv but not sent. Every
message states plainly this is a research tier, not a trade signal --
backtested, not yet live-validated as a real-time predictor.

TIERING (see breakout_research/RESEARCH_LOG.md Iterations 4-7 for the real,
backtested numbers behind every threshold below):

  AVOID        PRE-BREAKOUT with pct_b in [75,85) (the real "dead zone") OR
               RSI>=80. Real, validated on 537 live alerts: win rate 46.8%
               ->45.7%, avg return -0.09%->-0.37% (EOD->+5d).
  NEUTRAL      Everything not AVOID and not PREFER.
  PREFER-LOW   Clears the coarse PREFER bar (BREAKOUT, or PRE-BREAKOUT near
               the 52w high) but fails the finer within-PREFER checks below.
               Real finding: this bucket actually goes NEGATIVE (median
               -0.47% to -0.54% by +3d/+5d) -- worse than NEUTRAL. Flagged
               honestly rather than silently tightened; still tagged
               separately so it's visible, not hidden inside PREFER.
  PREFER-MID   Clears 1 of the within-PREFER quality checks.
  PREFER-HIGH  BREAKOUT + ADX<30, or PRE-BREAKOUT clearing >=2 of 3 quality
               checks (pct_b 65-75, dist from 52w high >-2%, vol_ratio>=1.5).
               Real: win rate 57.6%->59.9%, avg +1.09%->+1.39% (+3d->+5d).

ADX and distance-from-52-week-high are NOT columns in alert_history (only
pct_b/rsi/vol_ratio are captured live) -- computed here in real time from a
real yfinance daily-bar pull per newly-seen ticker (cached within one poll
run), so the live tag stays faithful to the full validated ranking instead
of silently dropping two of its real components.

Two jobs, one script, run repeatedly (state persisted so re-runs pick up
only new alerts):
  1. --poll (default): tag new alert_history rows, append to
     shadow_filter_log.csv. Meant to run every ~15min during market hours.
  2. --report: joins shadow_filter_log.csv against alert_performance's real
     outcomes and reports live, out-of-sample performance PER TIER -- the
     strongest possible validation, since these are alerts that fired AFTER
     the ranking was designed, not historical replay.
"""
import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
BACKEND_DIR = r"C:\Projects\GenAI-Projects\ibkr_trader\backend"
TAPE_DB = f"{BACKEND_DIR}\\tape_data.db"
STATE_FILE = HERE / "shadow_filter_state.json"
LOG_FILE = HERE / "shadow_filter_log.csv"
ET = ZoneInfo("America/New_York")


def _load_telegram_creds() -> tuple[str, str] | tuple[None, None]:
    try:
        with open(f"{BACKEND_DIR}\\scanner_config.json") as f:
            cfg = json.load(f)
        return cfg.get("telegram_token"), cfg.get("telegram_chat_id")
    except Exception:
        return None, None


def telegram(msg: str):
    import requests
    token, chat_id = _load_telegram_creds()
    if not token or not chat_id:
        print("  (Telegram creds not found in scanner_config.json -- skipping send)")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as e:
        print(f"  Telegram send failed: {e}")


def market_is_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t


def _compute_adx(high, low, close, period=14):
    import numpy as np
    import pandas as pd
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


_ticker_cache: dict[str, dict] = {}


def get_adx_and_dist52w(ticker: str) -> tuple[float | None, float | None]:
    """Real-time equivalents of the two backtest-only features. Cached per
    ticker within one process run (poll cycles rarely see the same ticker
    fire twice in one 15-min window, but cheap to guard anyway)."""
    if ticker in _ticker_cache:
        c = _ticker_cache[ticker]
        return c["adx"], c["dist52w"]
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=False)
        # Real point-in-time fix, 2026-08-31 (CEO question: "will it not be
        # stale if calculated on prior day"): yfinance's daily history
        # includes a live-updating row for TODAY once the session has
        # started -- called ~11-15min after an alert fires (this monitor's
        # poll cadence), that row already reflects those extra minutes of
        # real intraday price action, a genuine (if small) look-ahead leak
        # into the very indicator deciding that alert's tier. Measured the
        # real cost of dropping it: comparing live vs. prior-day-only ADX
        # across 15 real sampled alerts, differences ran -1.42 to +1.49,
        # consistent with ADX being a 14-day Wilder-smoothed indicator that
        # barely moves on one extra partial day -- negligible cost, and it
        # makes every future classification genuinely point-in-time-clean.
        hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
        today_et = datetime.now(ET).date()
        hist = hist[hist.index.date < today_et]
        if len(hist) < 30:
            _ticker_cache[ticker] = {"adx": None, "dist52w": None}
            return None, None
        adx_series = _compute_adx(hist["High"].values, hist["Low"].values, hist["Close"].values, 14)
        adx_val = float(adx_series.iloc[-1]) if len(adx_series) else None
        roll_high = float(hist["Close"].rolling(252, min_periods=100).max().iloc[-1])
        last_close = float(hist["Close"].iloc[-1])
        dist52w = (last_close - roll_high) / roll_high * 100 if roll_high else None
        _ticker_cache[ticker] = {"adx": adx_val, "dist52w": dist52w}
        return adx_val, dist52w
    except Exception as e:
        print(f"  {ticker}: real-time ADX/52w-high fetch failed: {e}")
        _ticker_cache[ticker] = {"adx": None, "dist52w": None}
        return None, None


def classify_tier(signal_type: str, pct_b, rsi, vol_ratio, adx, dist52w) -> str:
    if signal_type == "BREAKOUT":
        if adx is not None and adx < 30:
            return "PREFER-HIGH"
        return "PREFER-MID"

    if signal_type != "PRE-BREAKOUT":
        return "UNKNOWN"

    if pct_b is None or rsi is None:
        return "UNKNOWN"

    dead_zone = 75 <= pct_b < 85
    overbought = rsi >= 80
    if dead_zone or overbought:
        return "AVOID"

    near_high = dist52w is not None and dist52w > -5
    if not near_high:
        return "NEUTRAL"

    pts = 0
    if 65 <= pct_b < 75:
        pts += 1
    if dist52w is not None and dist52w > -2:
        pts += 1
    if vol_ratio is not None and vol_ratio >= 1.5:
        pts += 1
    if pts >= 2:
        return "PREFER-HIGH"
    if pts == 1:
        return "PREFER-MID"
    return "PREFER-LOW"


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"last_seen_id": 0}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def poll():
    state = load_state()
    con = sqlite3.connect(TAPE_DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT id, fired_at, session_date, ticker, signal_type, price, pct_b, rsi, vol_ratio
        FROM alert_history WHERE id > ? ORDER BY id
    """, (state["last_seen_id"],)).fetchall()
    con.close()

    if not rows:
        print(f"No new alerts since id={state['last_seen_id']}.")
        return

    is_new_file = not LOG_FILE.exists()
    prefer_high = []
    with open(LOG_FILE, "a") as f:
        if is_new_file:
            f.write("alert_id,fired_at,session_date,ticker,signal_type,price,pct_b,rsi,vol_ratio,"
                     "adx,dist_52w_high,tier,checked_at\n")
        for r in rows:
            adx, dist52w = get_adx_and_dist52w(r["ticker"])
            tier = classify_tier(r["signal_type"], r["pct_b"], r["rsi"], r["vol_ratio"], adx, dist52w)
            f.write(f"{r['id']},{r['fired_at']},{r['session_date']},{r['ticker']},{r['signal_type']},"
                    f"{r['price']},{r['pct_b']},{r['rsi']},{r['vol_ratio']},{adx},{dist52w},{tier},"
                    f"{datetime.now(timezone.utc).isoformat()}\n")
            print(f"  [{r['id']}] {r['ticker']} {r['signal_type']} pct_b={r['pct_b']} rsi={r['rsi']} "
                  f"adx={adx} dist52w={dist52w} -> {tier}")
            if tier == "PREFER-HIGH":
                prefer_high.append({**dict(r), "adx": adx, "dist52w": dist52w})

    state["last_seen_id"] = rows[-1]["id"]
    save_state(state)
    print(f"Logged {len(rows)} new alert(s). last_seen_id now {state['last_seen_id']}.")

    if prefer_high:
        lines = [f"🎯 <b>PREFER-HIGH breakout alert(s)</b> — {len(prefer_high)}",
                 "<i>Research tier, not a trade signal -- backtested but not yet validated "
                 "as a real-time predictor. See breakout_research/RESEARCH_LOG.md.</i>"]
        for a in prefer_high:
            reason = (f"ADX={a['adx']:.1f}" if a["signal_type"] == "BREAKOUT" and a["adx"] is not None
                      else f"%B={a['pct_b']:.1f} RSI={a['rsi']:.1f} 52w={a['dist52w']:+.1f}%"
                      if a["dist52w"] is not None else "")
            lines.append(f"\n<b>{a['ticker']}</b> {a['signal_type']} @ ${a['price']:.2f}  ({reason})")
        telegram("\n".join(lines))
        print(f"Sent Telegram digest for {len(prefer_high)} PREFER-HIGH alert(s).")


def report():
    import pandas as pd
    if not LOG_FILE.exists():
        print("No shadow_filter_log.csv yet -- run --poll first (needs real alerts to have fired).")
        return
    shadow = pd.read_csv(LOG_FILE)
    con = sqlite3.connect(TAPE_DB)
    perf = pd.read_sql_query(
        "SELECT session_date, ticker, signal_type, alert_price, eod_return_pct, is_win "
        "FROM alert_performance WHERE alert_price IS NOT NULL", con)
    con.close()

    merged = shadow.merge(perf, on=["session_date", "ticker", "signal_type"], how="inner")
    print(f"{len(shadow)} shadow-logged alerts, {len(merged)} with real EOD outcomes so far "
          f"(rest haven't been enriched yet or are still open).")
    if merged.empty:
        return

    order = ["AVOID", "NEUTRAL", "PREFER-LOW", "PREFER-MID", "PREFER-HIGH"]
    for tier in order:
        sub = merged[merged["tier"] == tier]
        if len(sub) < 3:
            print(f"  {tier:12} n={len(sub)} (too few for a real read yet)")
            continue
        wr = (sub["eod_return_pct"] > 0).mean() * 100
        avg = sub["eod_return_pct"].mean()
        print(f"  {tier:12} n={len(sub):4} win_rate={wr:5.1f}% avg_eod_return={avg:+.3f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="report live shadow-tier performance instead of polling")
    ap.add_argument("--ignore-market-hours", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
    elif not args.ignore_market_hours and not market_is_open():
        print("Market is closed (weekday 9:30-16:00 ET only) -- skipping poll.")
    else:
        poll()
