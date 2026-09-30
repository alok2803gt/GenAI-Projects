"""
SHADOW / PARALLEL bullish-harami + downtrend scanner on IBKR daily bars.

Built 2026-09-20 to answer one question before anyone touches the live
pipeline: does harami_scanner.py's yfinance data source (degraded on this Mac
-- curl_cffi can't load on macOS 12, so yfinance falls back to plain `requests`
and is exposed to Yahoo rate-limiting) produce the SAME signals as IBKR's own
daily bars? The detection rule below is a line-for-line copy of
harami_scanner.check_ticker's; only the data source differs.

PRIMARY MODE (--primary, promoted 2026-09-23): writes the
`harami_scanner_alert` entries harami_daily_trader.py reads, i.e. IBKR bars
now drive the live signal. Promoted because yfinance lost an ENTIRE session:
on 2026-09-22 it returned no data for 0/112 tickers (UPS's real harami was
missed silently) while IBKR scanned 112/112 clean, and on the one day both
had data they agreed exactly. harami_scanner.py now runs with --shadow as
the cross-check, writing no alerts.

Without --primary this stays shadow-only: NO orders, NO `harami_scanner_alert`
entries (writing them from two scanners would double-trade every signal);
results go to harami_scanner_ibkr_shadow.jsonl only.

Modes:
  (default)        live run after the close (scheduled 4:12pm ET weekdays):
                   scan the universe on IBKR bars, compare against what the
                   yfinance scanner alerted today, log + one Telegram line.
  --replay N       historical parity check: run BOTH data sources over the last
                   N trading days and report where they disagree. Immediate
                   evidence without waiting N live days.
  --report         summarize agreement across all live shadow runs so far.

Data note: IBKR "TRADES" daily bars vs yfinance auto-adjusted bars agree on
closes; opens can differ by a cent or so (yfinance uses the official opening
print, IBKR the first trade), and the rule has an `open > prior_close` edge,
so rare near-threshold disagreements are expected -- that is what the
replay/report modes measure.
"""
import argparse
import json
import pickle
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
sys.path.insert(0, str(BACKEND_DIR))
from harami_daily_trader import _is_trading_day  # noqa: E402  (shared NYSE holiday calendar)

ET = ZoneInfo("America/New_York")
TWS_PORT = 7496
CLIENT_ID = 2452
BODY_LOOKBACK = 60
SHADOW_LOG = HERE / "harami_scanner_ibkr_shadow.jsonl"
OVERSIGHT_LOG = BACKEND_DIR / "oversight_log.jsonl"
_RE = re.compile(r"(?:Bullish Harami \+ downtrend|Inside Day Reversal):\s*(?P<ticker>[A-Z][A-Z.\-]*)\s*\((?P<date>\d{4}-\d{2}-\d{2})\)")


def load_universe():
    with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
        return sorted(pickle.load(f).keys())


def detect(hist: pd.DataFrame):
    """Same rule as harami_scanner.check_ticker, applied to the LAST row of
    `hist` (columns Open/High/Low/Close, ascending date index)."""
    if len(hist) < BODY_LOOKBACK + 5:
        return None
    today, prior = hist.iloc[-1], hist.iloc[-2]
    today_open, today_close = today["Open"], today["Close"]
    prior_open, prior_close = prior["Open"], prior["Close"]

    prior_bearish = prior_close < prior_open
    today_bullish = today_close > today_open
    contained = (today_open > prior_close) and (today_close < prior_open)
    if not (prior_bearish and today_bullish and contained):
        return None

    bodies = (hist["Close"] - hist["Open"]).abs()
    body_median = bodies.iloc[-(BODY_LOOKBACK + 1):-1].median()
    if abs(prior_close - prior_open) < body_median:
        return None

    sma20 = hist["Close"].tail(20).mean()
    sma50 = hist["Close"].tail(50).mean()
    if not (today_close < sma20 and today_close < sma50):
        return None
    return {"date": hist.index[-1].strftime("%Y-%m-%d"),
            "prior_open": round(float(prior_open), 2), "prior_close": round(float(prior_close), 2),
            "today_open": round(float(today_open), 2), "today_close": round(float(today_close), 2),
            "sma20": round(float(sma20), 2), "sma50": round(float(sma50), 2)}


def ibkr_daily(ib, ticker, duration):
    from ib_insync import Stock
    c = Stock(ticker, "SMART", "USD")
    if not ib.qualifyContracts(c):
        raise RuntimeError("contract did not qualify")
    bars = ib.reqHistoricalData(c, endDateTime="", durationStr=duration, barSizeSetting="1 day",
                                whatToShow="TRADES", useRTH=True, formatDate=1)
    if not bars:
        raise RuntimeError("no bars returned")
    df = pd.DataFrame([{"Open": b.open, "High": b.high, "Low": b.low, "Close": b.close} for b in bars],
                      index=pd.to_datetime([str(b.date) for b in bars]))
    return df


def connect():
    from ib_insync import IB
    ib = IB()
    ib.RequestTimeout = 45
    ib.errorEvent += lambda *a: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    return ib


def yfinance_alerts_for(date_str):
    """Tickers the yfinance scanner saw for `date_str`. Since 2026-09-23 that
    scanner runs as a cross-check and records its hits to
    harami_scanner_yf_shadow.jsonl; older days are read from its alert entries
    in oversight_log.jsonl."""
    yf_shadow = HERE / "harami_scanner_yf_shadow.jsonl"
    if yf_shadow.exists():
        for line in yf_shadow.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("date") == date_str:
                return set(r.get("hits", []))
    out = set()
    try:
        with open(OVERSIGHT_LOG, encoding="utf-8") as f:
            for line in f:
                if '"harami_scanner_alert"' not in line:
                    continue
                m = _RE.search(json.loads(line).get("summary", ""))
                if m and m.group("date") == date_str:
                    out.add(m.group("ticker"))
    except Exception as e:
        print(f"could not read yfinance alerts: {e}")
        return None
    return out


def telegram(text):
    import requests
    try:
        from telegram_alert_gate import alert_enabled
        if not alert_enabled("harami_trader"):
            return
    except Exception:
        pass
    try:
        cfg = json.loads((BACKEND_DIR / "scanner_config.json").read_text())
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      data={"chat_id": cfg["telegram_chat_id"], "text": text}, timeout=10)
    except Exception as e:
        print(f"Telegram send failed: {e}")


def run_live(force=False, primary=False):
    et = datetime.now(ET)
    today = et.date()
    if not _is_trading_day(today):
        print(f"{today} is not a trading day, nothing to do.")
        return
    today_str = today.isoformat()
    if not force and SHADOW_LOG.exists():
        for line in SHADOW_LOG.read_text().splitlines():
            if line and json.loads(line).get("date") == today_str:
                print(f"already recorded {today_str}; use --force to re-run.")
                return

    universe = load_universe()
    try:
        ib = connect()
    except Exception as e:
        msg = f"harami IBKR-shadow scanner: could not connect to TWS ({e}) -- no result for {today_str}."
        print(msg)
        telegram("⚠️ " + msg)
        sys.exit(1)

    hits, errors, stale = [], {}, []
    try:
        for t in universe:
            try:
                df = ibkr_daily(ib, t, "6 M")
            except Exception as e:
                errors[t] = f"{type(e).__name__}: {e}"
                continue
            if df.index[-1].date() != today:
                stale.append(t)     # no bar for today yet -> do NOT evaluate yesterday's as today's
                continue
            hit = detect(df)
            if hit:
                hits.append({"ticker": t, **hit})
            ib.sleep(0.5)
    finally:
        ib.disconnect()

    if primary:                       # write the alerts harami_daily_trader.py reads
        for h in hits:
            text = (
                f"\U0001F56F\ufe0f Inside Day Reversal: {h['ticker']} ({today_str})\n"
                f"Prior day: {h['prior_open']:.2f} -> {h['prior_close']:.2f} (bearish)\n"
                f"Today: {h['today_open']:.2f} -> {h['today_close']:.2f} (bullish, inside prior body)\n"
                f"Close ${h['today_close']:.2f} vs SMA20 ${h['sma20']:.2f} / SMA50 ${h['sma50']:.2f} "
                f"(both below -- real downtrend context)\n"
                f"Backtested plan (5y, real, p=0.00007 vs matched baseline): enter at tomorrow's open, "
                f"hold 5 trading days. Source: IBKR daily bars. Manual review only, no order placed."
            )
            print(text)
            telegram(text)
            entry = {
                "time": datetime.now(timezone.utc).isoformat(), "actor": "trader",
                "category": "harami_scanner_alert", "summary": text.replace("\n", " | "),
                "rationale": "Bullish harami + downtrend fired on IBKR daily bars (primary source since 2026-09-23).",
                "outcome": "Alert sent, manual review only, no order placed.", "pnl_impact": None,
            }
            with open(OVERSIGHT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    ibkr_set = {h["ticker"] for h in hits}
    yf_set = yfinance_alerts_for(today_str)
    rec = {"time": datetime.now(timezone.utc).isoformat(), "date": today_str, "scanned": len(universe),
           "n_errors": len(errors), "errors": errors, "n_stale": len(stale), "stale": stale,
           "ibkr_hits": sorted(ibkr_set), "yf_hits": sorted(yf_set) if yf_set is not None else None,
           "only_ibkr": sorted(ibkr_set - yf_set) if yf_set is not None else None,
           "only_yf": sorted(yf_set - ibkr_set) if yf_set is not None else None, "detail": hits}
    with open(SHADOW_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    if yf_set is None:
        verdict = "yfinance result unavailable"
    elif ibkr_set == yf_set:
        verdict = "MATCH"
    else:
        verdict = f"DIFF (IBKR-only {sorted(ibkr_set - yf_set)}, yfinance-only {sorted(yf_set - ibkr_set)})"
    role = "PRIMARY" if primary else "shadow"
    text = (f"🕯️ Inside Day Reversal IBKR-{role} {today_str}: IBKR {sorted(ibkr_set) or 'none'} | "
            f"yfinance {sorted(yf_set) if yf_set is not None else '?'} -> {verdict} "
            f"(scanned {len(universe) - len(errors) - len(stale)}/{len(universe)}, "
            f"{len(errors)} errors, {len(stale)} stale). "
            f"{'Alerts written for the entry job.' if primary else 'Shadow only, no orders.'}")
    print(text)
    telegram(text)


def run_replay(n_days):
    import yfinance as yf
    universe = load_universe()
    ib = connect()
    ibkr_data, yf_data, ibkr_err, yf_err = {}, {}, {}, {}
    try:
        for i, t in enumerate(universe, 1):
            try:
                ibkr_data[t] = ibkr_daily(ib, t, "1 Y")
            except Exception as e:
                ibkr_err[t] = str(e)
            try:
                # auto_adjust=False: the live scanner sees prices as of THAT day; the default (dividend-adjusted as of today) would make older
                # replay dates differ from what it saw, inflating the disagreement count with a pure artifact.
                h = yf.Ticker(t).history(period="1y", interval="1d", auto_adjust=False)
                # tz_localize(None) keeps the exchange-local wall date (midnight); tz_convert(None)
                # would shift to UTC (04:00) and never line up with IBKR's midnight dates.
                h.index = pd.to_datetime(h.index).tz_localize(None)
                yf_data[t] = h
            except Exception as e:
                yf_err[t] = str(e)
            ib.sleep(0.5)
            if i % 20 == 0:
                print(f"  fetched {i}/{len(universe)}", flush=True)
    finally:
        ib.disconnect()
    print(f"IBKR ok {len(ibkr_data)} err {len(ibkr_err)} | yfinance ok {len(yf_data)} err {len(yf_err)}")

    both = sorted(set(ibkr_data) & set(yf_data))
    # evaluation dates = last n trading days present in BOTH sources (using SPY's index as the calendar)
    cal = [d for d in ibkr_data["SPY"].index if d in yf_data["SPY"].index][-n_days:]
    per_day, dis = {}, []
    for d in cal:
        a, b = set(), set()
        for t in both:
            for src, store in ((a, ibkr_data), (b, yf_data)):
                h = store[t]
                h = h[h.index <= d]
                if len(h) and h.index[-1] == d and detect(h.tail(85)):
                    src.add(t)
        per_day[d.strftime("%Y-%m-%d")] = (sorted(a), sorted(b))
        if a != b:
            dis.append((d.strftime("%Y-%m-%d"), sorted(a - b), sorted(b - a)))
    tot_a = sum(len(v[0]) for v in per_day.values())
    tot_b = sum(len(v[1]) for v in per_day.values())
    agree = sum(len(set(v[0]) & set(v[1])) for v in per_day.values())
    print(f"\nReplay over {len(cal)} trading days x {len(both)} tickers: "
          f"IBKR signals {tot_a}, yfinance signals {tot_b}, agreed {agree}")
    print(f"Days with any disagreement: {len(dis)}/{len(cal)}")
    for d, only_i, only_y in dis:
        print(f"  {d}: IBKR-only {only_i}  yfinance-only {only_y}")
    (HERE / "harami_scanner_ibkr_replay.json").write_text(json.dumps(
        {"days": len(cal), "tickers": len(both), "per_day": per_day, "disagreements": dis,
         "ibkr_errors": ibkr_err, "yf_errors": yf_err}, indent=2))


def run_report():
    if not SHADOW_LOG.exists():
        print("no live shadow runs recorded yet.")
        return
    rows = [json.loads(l) for l in SHADOW_LOG.read_text().splitlines() if l]
    seen = {}
    for r in rows:
        seen[r["date"]] = r            # last run per date wins
    print(f"{'date':<12}{'IBKR':<28}{'yfinance':<28}{'verdict'}")
    match = 0
    comparable = 0
    for d, r in sorted(seen.items()):
        if r["yf_hits"] is None:
            v = "n/a"
        else:
            comparable += 1
            ok = r["ibkr_hits"] == r["yf_hits"]
            match += ok
            v = "MATCH" if ok else f"DIFF only_ibkr={r['only_ibkr']} only_yf={r['only_yf']}"
        print(f"{d:<12}{str(r['ibkr_hits']):<28}{str(r['yf_hits']):<28}{v}   errors={r['n_errors']} stale={r['n_stale']}")
    print(f"\n{match}/{comparable} comparable days matched exactly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, metavar="N")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--primary", action="store_true",
                    help="write the harami_scanner_alert entries the entry job reads (IBKR is the live source)")
    args = ap.parse_args()
    if args.replay:
        run_replay(args.replay)
    elif args.report:
        run_report()
    else:
        run_live(force=args.force, primary=args.primary)
