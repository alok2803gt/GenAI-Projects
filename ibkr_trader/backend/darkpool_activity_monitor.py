"""
Dark-Pool Activity Monitor -- polls Unusual Whales' real dark-pool trade
tape (unusual_whales_client.py) and sends a Telegram alert for any NEW,
non-canceled print above a premium threshold in this account's curated
watch universe. Meant to run on a repeating Task Scheduler trigger (every
~10-15 min during market hours), same architecture pattern as
pdd_condor_babysitter.py / goog_residual_put_monitor.py -- one poll-and-
alert pass per invocation, dedup state persisted between runs.

READ-ONLY. Never places an order, never touches live trading config or
any strategy's position/risk state. Purely informational, same epistemic
status as darkpool-levels-calculator's "wall" concept (see that skill's
SKILL.md): a real, observed print is not a directional signal on its own
-- UW's /darkpool/recent endpoint reports size/price/premium only, no
buy/sell aggressor side. A big block could be accumulation, distribution,
or neutral crossing/block-facilitation. Every alert says this explicitly.

Dark-pool activity is the master list here -- breakout_scanner.py's
current per-ticker state (persisted daily to ticker_states_{date}.json,
same directory) is cross-referenced ONLY to flag/reorder, never to filter:
a dark-pool print on a ticker NOT currently BREAKOUT/PRE-BREAKOUT/EXTENDED
still shows in the digest, just without the flag. Two independent, both
real signals shown together -- this does not imply one confirms the
other; no backtest exists for that combination (the one related idea that
WAS tested, F11 same-day-aggressive-dark-pool-print-predicts-continuation,
came back with no edge, per darkpool-levels-calculator's SKILL.md -- this
is a different, weaker claim: "these two independently-real things both
happened," not "this predicts that").

Threshold: RELATIVE, not a flat dollar figure -- changed 2026-08-24 after
a real, CEO-flagged gap: a flat $5M print means wildly different things
per ticker. Checked with real 20-day avg-dollar-volume data (same metric
breakout_scanner.py's F9 gate already uses): $5M is 0.0145% of MU/SPY's
ADV but 1.64% of LULU's -- a ~113x difference in relative significance
for the identical dollar figure. Now: alert threshold per ticker =
max(MIN_PREMIUM_FLOOR, REL_PREMIUM_PCT% of that ticker's own 20-day ADV),
so a mega-cap needs a genuinely large print (~$150-200M range for
MU/SPY-scale names) while a smaller curated-universe name can trigger on
something much smaller in absolute terms but still a real, meaningful
chunk of its typical day. ADV is cached daily (ADV_CACHE_FILE) --
recomputing 112 real yfinance calls every 15-min poll would be wasteful;
refreshed once per day, first run of the day pays the ~1-2min cost.

Server-side API filtering still uses a flat, LOW floor (MIN_PREMIUM_FLOOR)
since Unusual Whales' /darkpool/recent has no per-ticker filter -- the
real per-ticker relative threshold is applied client-side after fetching.
Checked empirically that this floor still leaves the 200-row fetch window
covering well over 20 minutes of real time (confirmed 117.8min at a $1M
floor during moderate after-hours activity 2026-08-24) -- `limit=200` on
the API is a ROW-count window, not a time window, so a too-low floor
could in principle shrink that below the LOOKBACK_MINUTES cutoff during a
genuinely heavy session; not yet verified under real regular-hours peak
load, worth rechecking if alerts start looking sparse during a busy day.

Calibrated against real unusual_whales_client.recent_trades() data on
2026-08-24: ordinary individual dark-pool prints run roughly $100K-$700K
premium in normal market flow (ambient noise floor, well under even the
$1M server-side floor). Activity is also extremely BURSTY, not steady: one
15-min window alone had 28 prints >=$5M landing within about 2 minutes of
each other (almost certainly one basket/index-rebalance execution
reported ticker-by-ticker, not 28 independent signals), versus 1-3 per
window in calmer periods. Two real design consequences of that finding:
(1) alerts are a single digest message per run, never one message per
print, so a burst like that doesn't flood Telegram; (2) only prints with
executed_at inside the last LOOKBACK_MINUTES are ever considered -- without
a real time cutoff, the very first run against an empty dedup-state file
would backfill-alert on everything in a stale multi-hour window.
"""
from __future__ import annotations

import json
import sys

# Console prints include emoji tags (breakout flags); Windows' default
# console codepage (cp1252) can't encode them and raises UnicodeEncodeError.
# Doesn't affect the real Telegram send (UTF-8 JSON payload, unrelated to
# console codepage) -- only crashed manual/--dry-run console output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # reconfigure() needs Python 3.7+; harmless no-op otherwise
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BACKEND_DIR = Path(r"C:\Projects\GenAI-Projects\ibkr_trader\backend")
sys.path.insert(0, str(BACKEND_DIR))

from unusual_whales_client import UnusualWhalesClient  # noqa: E402

ET = ZoneInfo("America/New_York")
STATE_FILE = BACKEND_DIR / "darkpool_activity_state.json"
ADV_CACHE_FILE = BACKEND_DIR / "darkpool_adv_cache.json"
MIN_PREMIUM_FLOOR = 1_000_000.0   # server-side API filter AND absolute per-ticker floor
REL_PREMIUM_PCT = 0.5              # alert threshold = this % of the ticker's own 20-day ADV
DEFAULT_ADV_FALLBACK = 500_000_000.0  # used only if a ticker is somehow missing from the cache
LOOKBACK_MINUTES = 20  # margin above the 15-min poll interval this is scheduled at
MAX_SEEN_IDS = 2000  # bound state file growth; oldest dropped first

# Same curated universe darkpool-levels-calculator / safe-income-screener /
# gex-vex-calculator already use -- duplicated here for self-containment,
# same convention as every other standalone script in this account.
UNIVERSE = {
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF", "XLK": "ETF", "XLF": "ETF",
    "XLE": "ETF", "XLV": "ETF", "XLI": "ETF", "GLD": "ETF", "TLT": "ETF", "ARKK": "ETF",
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech", "AMZN": "Tech",
    "META": "Tech", "TSLA": "Tech", "NFLX": "Tech",
    "AMD": "Semis", "INTC": "Semis", "QCOM": "Semis", "AVGO": "Semis", "TXN": "Semis",
    "MU": "Semis", "AMAT": "Semis", "LRCX": "Semis", "KLAC": "Semis", "MRVL": "Semis", "SMCI": "Semis",
    "CRM": "Software", "NOW": "Software", "ADBE": "Software", "ORCL": "Software",
    "SNOW": "Software", "PANW": "Software", "CRWD": "Software", "ZS": "Software",
    "DDOG": "Software", "NET": "Software",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "BLK": "Financials", "SCHW": "Financials",
    "V": "Financials", "MA": "Financials", "AXP": "Financials", "TFC": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare", "DHR": "Healthcare",
    "ISRG": "Healthcare", "VRTX": "Healthcare", "GILD": "Healthcare", "BMY": "Healthcare",
    "HD": "ConsumerDisc", "MCD": "ConsumerDisc", "SBUX": "ConsumerDisc", "NKE": "ConsumerDisc",
    "LOW": "ConsumerDisc", "TGT": "ConsumerDisc", "COST": "ConsumerDisc", "BKNG": "ConsumerDisc",
    "LULU": "ConsumerDisc",
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "WMT": "Staples",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "MPC": "Energy",
    "VLO": "Energy", "OXY": "Energy",
    "BA": "Industrials", "GE": "Industrials", "CAT": "Industrials", "HON": "Industrials",
    "RTX": "Industrials", "LMT": "Industrials", "FDX": "Industrials", "UPS": "Industrials",
    "DE": "Industrials", "UAL": "Industrials",
    "DIS": "Comms", "CMCSA": "Comms", "VZ": "Comms", "T": "Comms",
    "COIN": "Growth", "PLTR": "Growth", "UBER": "Growth", "RIVN": "Growth", "ROKU": "Growth",
    "HOOD": "Growth", "SOFI": "Growth", "PYPL": "Growth", "XYZ": "Growth", "IBM": "Growth",
    "RBLX": "Growth", "RCL": "Growth", "ABNB": "Growth",
}


def _compute_adv() -> dict[str, float]:
    """20-day avg dollar volume per ticker (avg volume x last close), real
    yfinance data -- same methodology as breakout_scanner.py's F9 liquidity
    gate and this session's rank_universe_liquidity.py. Takes ~1-2min for
    the full curated universe; only called when the daily cache is stale."""
    import yfinance as yf
    adv: dict[str, float] = {}
    for i, tk in enumerate(UNIVERSE, 1):
        try:
            hist = yf.Ticker(tk).history(period="1mo", interval="1d", auto_adjust=False)
            if len(hist) < 5:
                continue
            avg_vol = hist["Volume"].tail(20).mean()
            last_close = float(hist["Close"].iloc[-1])
            adv[tk] = float(avg_vol * last_close)
        except Exception as e:
            print(f"  [{i}/{len(UNIVERSE)}] {tk}: ADV fetch failed: {e}")
    return adv


def load_or_refresh_adv_cache() -> dict[str, float]:
    """Refreshes once per real calendar day; every other run just reads the
    cached values -- recomputing 112 yfinance calls on every 15-min poll
    would be wasteful and slow the whole run down."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    try:
        with open(ADV_CACHE_FILE) as f:
            cache = json.load(f)
        if cache.get("date") == today and cache.get("adv"):
            return cache["adv"]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    print("ADV cache stale/missing -- recomputing (real yfinance data, ~1-2min)...")
    adv = _compute_adv()
    with open(ADV_CACHE_FILE, "w") as f:
        json.dump({"date": today, "adv": adv}, f, indent=2)
    return adv


def threshold_for(ticker: str, adv: dict[str, float]) -> float:
    return max(MIN_PREMIUM_FLOOR, REL_PREMIUM_PCT / 100 * adv.get(ticker, DEFAULT_ADV_FALLBACK))


# Same "bullish" state set breakout_scanner.py's own _transition_alert() uses
# internally -- reusing its convention rather than inventing a new grouping.
BREAKOUT_FLAG_STATES = {"BREAKOUT": "🚨 BREAKOUT", "EXTENDED": "⚠️ EXTENDED", "PRE-BREAKOUT": "⚡ PRE-BREAKOUT"}


def load_breakout_states() -> dict[str, str]:
    """Today's per-ticker state from breakout_scanner.py's own persisted
    ledger (ticker_states_{date}.json, same directory -- written after
    every scan cycle, see breakout_scanner.py's #13 persistence comment).
    Returns {} if the scanner hasn't run today or the file is missing/
    unreadable -- this is optional, additive cross-reference, same "never
    block on a missing optional cache" convention as GEX/VEX and the
    darkpool price-levels wall. Only tickers in BREAKOUT_FLAG_STATES are
    returned; NEUTRAL/WEAKENING/PRE-BREAKDOWN/BREAKDOWN are not flagged."""
    path = BACKEND_DIR / f"ticker_states_{datetime.now(ET).strftime('%Y-%m-%d')}.json"
    try:
        with open(path) as f:
            states = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {tk: BREAKOUT_FLAG_STATES[v["state"]]
            for tk, v in states.items() if v.get("state") in BREAKOUT_FLAG_STATES}


def market_is_open() -> bool:
    """Weekday-only, NOT time-of-day-gated (changed 2026-08-24 -- see module
    docstring). This is a read-only alert monitor, not an order-placer, so
    the 9:30-16:00 execution-risk reasoning behind safe_income_auto.py's
    otherwise-identically-named guard doesn't apply here: confirmed live
    that real dark-pool prints keep landing seconds apart well into
    extended hours (checked 7:25 PM ET, 3.5h after close). Keeps the
    weekday check only because exchanges are fully closed weekends -- same
    convention as portfolio_oversight_check.py / cro_risk_check.py's
    24/7-with-in-script-weekday-skip pattern. Name kept for the module's
    existing call sites even though it no longer means 'regular session'.
    """
    return datetime.now(ET).weekday() < 5


def load_config() -> dict:
    with open(BACKEND_DIR / "scanner_config.json") as f:
        return json.load(f)


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"seen_tracking_ids": []}


def save_state(state: dict):
    # Cap growth -- keep only the most recent MAX_SEEN_IDS.
    state["seen_tracking_ids"] = state["seen_tracking_ids"][-MAX_SEEN_IDS:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


TELEGRAM_MAX_CHARS = 3800  # real limit is 4096; margin for the HTML tags/entities overhead


def _chunk_message(msg: str, max_chars: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Split on line boundaries (never mid-tag/mid-line) so a long digest
    -- e.g. a multi-hour backfill with 100 tickers -- doesn't get silently
    truncated or rejected by Telegram's real 4096-char message limit."""
    lines = msg.split("\n")
    chunks, current = [], ""
    for line in lines:
        candidate = current + ("\n" if current else "") + line
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def telegram(cfg: dict, msg: str):
    import requests
    chunks = _chunk_message(msg)
    for i, chunk in enumerate(chunks, 1):
        text = chunk if len(chunks) == 1 else f"{chunk}\n\n<i>({i}/{len(chunks)})</i>"
        try:
            requests.post(
                f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                json={"chat_id": cfg["telegram_chat_id"], "text": text, "parse_mode": "HTML"},
                timeout=8,
            )
        except Exception as e:
            print(f"Telegram send failed (chunk {i}/{len(chunks)}): {e}")


def log_oversight(summary: str, outcome: str, rationale: str):
    entry = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(),
        "actor": "trader",
        "category": "darkpool_activity_monitor",
        "summary": summary,
        "rationale": rationale,
        "outcome": outcome,
        "pnl_impact": None,
    }
    with open(BACKEND_DIR / "oversight_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def format_digest(trades: list[dict], rel_pct: float, breakout_states: dict[str, str] | None = None) -> str:
    """One consolidated message for the whole run -- never one message per
    print (see module docstring: a single basket-execution event can
    report dozens of near-simultaneous prints across many tickers).

    Dark pool is the master list -- every qualifying print shows regardless
    of breakout_scanner state. Tickers breakout_scanner currently has flagged
    (BREAKOUT/PRE-BREAKOUT/EXTENDED) get a visible tag AND sort to the top,
    so the overlap is easy to spot without hiding anything else."""
    breakout_states = breakout_states or {}
    lines = [f"🌑 <b>Dark pool activity</b> — {len(trades)} print(s) ≥ {rel_pct}% of each ticker's own 20d ADV"]
    by_ticker: dict[str, list[dict]] = {}
    for t in trades:
        by_ticker.setdefault(t["ticker"], []).append(t)

    flagged  = [tk for tk in by_ticker if tk in breakout_states]
    unflagged = [tk for tk in by_ticker if tk not in breakout_states]
    if flagged:
        lines.append(f"\n<b>⚡ Also flagged by breakout scanner today ({len(flagged)}):</b>")

    def _sort_key(tk):
        return -sum(float(x.get("premium", 0) or 0) for x in by_ticker[tk])

    for group_i, group in enumerate((sorted(flagged, key=_sort_key), sorted(unflagged, key=_sort_key))):
        if group_i == 1 and flagged and unflagged:
            lines.append("\n<b>Dark pool only:</b>")
        for ticker in group:
            rows = by_ticker[ticker]
            sector = UNIVERSE.get(ticker, "?")
            total = sum(float(r.get("premium", 0) or 0) for r in rows)
            tag = f" {breakout_states[ticker]}" if ticker in breakout_states else ""
            if len(rows) == 1:
                r = rows[0]
                lines.append(f"\n<b>{ticker}</b>{tag} ({sector}): {int(r.get('size', 0)):,} sh @ "
                             f"${float(r.get('price', 0)):,.2f} = ${total:,.0f}")
            else:
                lines.append(f"\n<b>{ticker}</b>{tag} ({sector}): {len(rows)} prints, ${total:,.0f} combined")

    lines.append("\n<i>Observed prints only -- no buy/sell side in this data. Not a "
                  "validated directional signal. Breakout-scanner flag is a separate, "
                  "independently-real signal shown alongside -- not a confirmation of "
                  "the print, no combined backtest exists.</i>")
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-premium-floor", type=float, default=MIN_PREMIUM_FLOOR,
                     help="server-side API filter AND absolute per-ticker floor (default $1M)")
    ap.add_argument("--rel-pct", type=float, default=REL_PREMIUM_PCT,
                     help="per-ticker alert threshold as %% of that ticker's own 20-day ADV (default 0.5)")
    ap.add_argument("--lookback-minutes", type=int, default=LOOKBACK_MINUTES)
    ap.add_argument("--backfill-hours", type=float, default=None,
                     help="One-off wide-window mode: fetch per-ticker via ticker_trades() (up to 500 rows EACH, "
                          "not a shared 200-row market-wide budget) instead of the normal recent_trades() call, "
                          "so a real multi-hour/full-day catch-up is actually possible. Takes ~30-60s (paced "
                          "0.3s/ticker across the universe). Not meant for the recurring 15-min schedule -- use "
                          "for a one-time backfill (e.g. the first run after activating this monitor).")
    ap.add_argument("--tickers", help="comma-separated subset override (default: full curated universe)")
    ap.add_argument("--ignore-market-hours", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print what would be alerted, send nothing")
    args = ap.parse_args()

    if not args.ignore_market_hours and not market_is_open():
        print("Market is closed (weekday 9:30-16:00 ET only) -- skipping this run.")
        return

    watch = set(t.strip().upper() for t in args.tickers.split(",")) if args.tickers else set(UNIVERSE.keys())
    cfg = load_config()
    state = load_state()
    seen = set(state["seen_tracking_ids"])
    adv = load_or_refresh_adv_cache()

    client = UnusualWhalesClient()
    if args.backfill_hours is not None:
        import time as _time
        trades = []
        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        watch_list = sorted(watch)
        for i, tk in enumerate(watch_list, 1):
            try:
                rows = client.ticker_trades(tk, date=today_str, limit=500, min_premium=args.min_premium_floor)
                trades.extend(rows)
            except Exception as e:
                print(f"  [{i}/{len(watch_list)}] {tk}: backfill fetch failed: {e}")
            _time.sleep(0.3)
        print(f"Backfill: fetched {len(trades)} raw prints across {len(watch_list)} tickers for {today_str}.")
    else:
        try:
            trades = client.recent_trades(limit=200, min_premium=args.min_premium_floor,
                                           order_by="executed_at", order="desc")
        except Exception as e:
            log_oversight(f"Dark pool activity poll failed: {e}", "error", "market-wide recent_trades() call failed")
            print(f"Poll failed: {e}")
            return

    lookback_desc = f"{args.backfill_hours}h backfill" if args.backfill_hours is not None else f"{args.lookback_minutes}min"
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=args.backfill_hours) if args.backfill_hours is not None
              else datetime.now(timezone.utc) - timedelta(minutes=args.lookback_minutes))
    # Oldest-first so the digest reads chronologically. Relative, per-ticker
    # threshold applied here (client-side) -- the API only got the flat floor.
    candidates = [t for t in reversed(trades)
                  if t.get("ticker") in watch and not t.get("canceled")
                  and t.get("tracking_id") not in seen
                  and _parse_ts(t["executed_at"]) >= cutoff
                  and float(t.get("premium", 0) or 0) >= threshold_for(t["ticker"], adv)]

    breakout_states = load_breakout_states()

    if args.dry_run:
        print(f"Would alert on {len(candidates)} new print(s) in the last {lookback_desc} "
              f"(of {len(trades)} fetched >= ${args.min_premium_floor:,.0f} floor, "
              f"{args.rel_pct}% of each ticker's own ADV). "
              f"{len(breakout_states)} tickers flagged by breakout_scanner today:")
        for t in candidates:
            tag = f" [{breakout_states[t['ticker']]}]" if t["ticker"] in breakout_states else ""
            thresh = threshold_for(t["ticker"], adv)
            print(f"  {t['ticker']}{tag}  ${float(t.get('premium', 0)):,.0f}  "
                  f"(threshold ${thresh:,.0f})  {t.get('executed_at')}")
        return

    # Mark every fetched tracking_id as seen regardless of the time window --
    # otherwise a print that ages out of the lookback before its first
    # qualifying poll would never be seen at all AND never get marked seen,
    # harmless either way since it's already outside the alert window, but
    # this keeps the seen-set an honest record of everything this run observed.
    for t in trades:
        if t.get("tracking_id") is not None:
            seen.add(t["tracking_id"])
    state["seen_tracking_ids"] = list(seen)
    save_state(state)

    if candidates:
        telegram(cfg, format_digest(candidates, args.rel_pct, breakout_states))
        tickers_hit = sorted({t["ticker"] for t in candidates})
        overlap = [tk for tk in tickers_hit if tk in breakout_states]
        log_oversight(
            f"Dark pool activity: {len(candidates)} new print(s) >= {args.rel_pct}% of each ticker's own "
            f"ADV (floor ${args.min_premium_floor:,.0f}) in last "
            f"{lookback_desc}, alerted as one digest: {', '.join(tickers_hit)}"
            + (f" -- {len(overlap)} also flagged by breakout_scanner: {', '.join(overlap)}" if overlap else ""),
            "alerted",
            f"Polled {len(trades)} market-wide prints above threshold, {len(candidates)} within the "
            f"lookback window and new after dedup.",
        )
        print(f"Alerted on {len(candidates)} new print(s): {', '.join(tickers_hit)}"
              + (f" ({len(overlap)} breakout-flagged)" if overlap else ""))
    else:
        print(f"No new qualifying prints this run ({len(trades)} fetched, 0 within "
              f"{args.lookback_minutes}min lookback after dedup/universe filter).")


if __name__ == "__main__":
    main()
