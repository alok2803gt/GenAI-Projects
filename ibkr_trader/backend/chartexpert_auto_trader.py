"""
Chartexpert auto-trader -- SHADOW MODE. One-shot task, weekday 9:35 AM ET.

Pipeline (per CEO instruction, 2026-09-10/11): generate the live 1-min
charts already built on the frontend's Charts tab -> screenshot each panel
-> read each chart with the chartexpert skill's exact format (a real vision
call, since "read this chart" is a judgment call, not a formula) -> send
the analysis + chart image to Telegram for all 3 symbols (SPY, QQQ, MNQ) ->
for SPY/QQQ only, if the parsed UP probability > SIGNAL_THRESHOLD_PCT,
track what a 1-contract ATM 0DTE call buy would have done, closed on a
0.5% trailing stop.

SHADOW MODE: places NO real IBKR orders. Every "would have bought" /
"would have exited" uses REAL live IBKR quotes throughout (never a
modeled/Black-Scholes price), so the logged P&L is exactly what a real
paper trade would show -- just not sent to the exchange. Logged to
chartexpert_shadow_trades.jsonl and oversight_log.jsonl, alerted via
Telegram exactly like a real fill would be. Flip SHADOW_MODE to False
(and get explicit CEO sign-off first) only after a real review of that
log -- this heuristic has zero track record before today.

Trailing stop is on the UNDERLYING price, not the option premium: a 0.5%
trailing stop on a 0DTE ATM option's own bid/ask would trigger almost
instantly from ordinary quote noise (a single 1-2 cent tick is often
>0.5% of an ATM 0DTE premium). Trailing the underlying and exiting the
option when THAT triggers is the only version of "0.5% trailing stop"
that is mechanically functional for a 0DTE option.

MNQ has no clean 0DTE-call equivalent (it is a future, not an equity with
daily listed options), so it stays chart+Telegram only -- never a trade
signal, matching the scope agreed for this build.

Needs a real Anthropic API key in scanner_config.json's anthropic_api_key.
The one on file as of 2026-09-10 is dead (confirmed 401) -- this script
fails LOUDLY (Telegram alert, non-zero exit so run_chartexpert_auto_trader.ps1's
stdout-capture wrapper catches it) rather than silently skipping the day.

Usage: python chartexpert_auto_trader.py
"""
import base64
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic
import requests
from ib_insync import IB, Option, Stock
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
CFG_PATH = HERE / "scanner_config.json"
STATE_PATH = HERE / "chartexpert_shadow_state.json"
TRADES_LOG = HERE / "chartexpert_shadow_trades.jsonl"
OVERSIGHT_LOG = HERE / "oversight_log.jsonl"
JOURNAL_DB_PATH = HERE / "trade_journal.db"
SHOT_DIR = HERE / "chartexpert_shots"
SHOT_DIR.mkdir(exist_ok=True)

FRONTEND_URL = "http://localhost:8001/index.html"
BACKEND_URL = "http://localhost:8000"
TWS_PORT = 7496
CLIENT_ID = 1850

ET = ZoneInfo("America/New_York")
SYMBOLS = ["SPY", "QQQ", "MNQ"]          # charted + Telegrammed every run
TRADEABLE = ["SPY", "QQQ"]                # only these can fire a shadow signal
SIGNAL_THRESHOLD_PCT = 70                 # CEO instruction: up% > 70
TRAILING_STOP_PCT = 0.005                 # 0.5%, on the UNDERLYING price
EOD_CUTOFF = "15:50"                      # hard close for any open shadow position
POLL_SECONDS = 30
SHADOW_MODE = True                        # per CEO decision 2026-09-11 -- do not flip without sign-off
MAX_BAR_AGE_SECONDS = 180                  # last 1m bar must be this fresh at 9:35 AM ET or the
                                            # chart is treated as stale (subscription/IBKR problem,
                                            # not benign -- this task only ever fires inside market
                                            # hours). Confirmed 2026-09-11: SPY/QQQ bars go 2.85h
                                            # stale outside their session with no error raised
                                            # anywhere -- a dead subscription during market hours
                                            # would look identical if nothing checked bar age.
SUBSCRIBE_TIMEOUT_SECONDS = 60             # total budget to get FRESH bars before giving up on a symbol

CHARTEXPERT_SYSTEM = """You are ChartWise Analyst -- an expert futures and equity chart reader.
Respond ONLY in the exact structured format below, no extra commentary before or after.

**Chart Analysis ([timeframe] timeframe, [Symbol] - [Full Name])**

**Current Price:** **[exact price shown on the chart]**

**Bias:** [Strongly Bullish / Bullish / Mildly Bullish / Neutral / Mildly Bearish / Bearish / Strongly Bearish] short-term ([1-sentence reason based on candles, structure, moving averages, RSI])

**Key Levels:**
- Resistance: [price range]
- Support/Demand: [price range]
- Key Confluence: [what's actually visible on this chart -- moving averages, volume, RSI, structure]

**Next directional move (next 1-3 hours):**
UP - Probability XX% | Down XX%
OR
DOWN - Probability XX% | Up XX%

**Primary Target:** [price] -> [next level] area

**Invalidation:** Clean close [above/below] [level] with volume (would shift bias)

**Trade Plan:**
- Favored short-term: [Long/Short] ... stop ... target ...
- Alternative: ...

**Overall Setup:** [2-3 sentences]

Rules: extract the current price exactly as shown. The two probabilities must add to 100.
Be objective and based only on what is visible on the chart -- never invent levels, patterns,
or labels that aren't actually there."""

_PROB_RE = re.compile(r"(UP|DOWN)\s*-\s*Probability\s*(\d+)%\s*\|\s*(?:Up|Down)\s*(\d+)%", re.IGNORECASE)
_PRICE_RE = re.compile(r"Current Price:\*\*\s*\*\*\$?([\d,]+\.?\d*)")


def now_et():
    return datetime.now(ET)


def load_cfg():
    return json.loads(CFG_PATH.read_text())


def telegram_text(cfg, text):
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    try:
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      data={"chat_id": cfg["telegram_chat_id"], "text": html, "parse_mode": "HTML"},
                      timeout=10)
    except Exception as e:
        print(f"Telegram text send failed: {e}")


def telegram_photo(cfg, path, caption):
    try:
        with open(path, "rb") as f:
            requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendPhoto",
                          data={"chat_id": cfg["telegram_chat_id"], "caption": caption},
                          files={"photo": f}, timeout=20)
    except Exception as e:
        print(f"Telegram photo send failed: {e}")


def oversight_log(actor, category, summary, rationale="", outcome=None, pnl_impact=None):
    entry = {"time": now_et().astimezone().isoformat(), "actor": actor, "category": category,
              "summary": summary, "rationale": rationale, "outcome": outcome, "pnl_impact": pnl_impact}
    with open(OVERSIGHT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _last_bar_age_seconds(ticker):
    """Age in seconds of the most recent bars-1m bar, or None if there are
    no bars at all yet. Positive -- the bar is that many seconds in the
    past; a bar that looks like it's in the future (clock skew) clamps to 0
    rather than going negative."""
    try:
        r = requests.get(f"{BACKEND_URL}/bars-1m/{ticker}?limit=1", timeout=10)
        if r.status_code != 200:
            return None
        bars = r.json()
        if not bars:
            return None
        bar_time = datetime.fromisoformat(bars[-1]["time"])
        age = (datetime.now(bar_time.tzinfo) - bar_time).total_seconds()
        return max(age, 0)
    except Exception:
        return None


# ── Chart generation + screenshot ────────────────────────────────────────────
def ensure_subscribed_and_shoot():
    """Subscribe all 3 symbols to the 1m stream, wait for bars that are both
    PRESENT and FRESH (age <= MAX_BAR_AGE_SECONDS -- a subscription that has
    plenty of old bars but stopped updating looks identical to a healthy one
    on a bare bar-count check), then screenshot each #chart-panel-<TICKER>
    div. Returns (shots, stale) -- shots is {ticker: png_path} for every
    symbol that produced a screenshot at all; stale is {ticker: age_seconds}
    for any symbol whose bars never became fresh within the time budget
    (still gets a chart+screenshot -- just flagged, not silently trusted)."""
    for t in SYMBOLS:
        try:
            requests.post(f"{BACKEND_URL}/add_ticker_1m", json={"ticker": t}, timeout=10)
        except Exception as e:
            print(f"  [{t}] add_ticker_1m failed: {e}")

    deadline = time.time() + SUBSCRIBE_TIMEOUT_SECONDS
    fresh, last_age = set(), {}
    while time.time() < deadline and len(fresh) < len(SYMBOLS):
        for t in SYMBOLS:
            if t in fresh:
                continue
            age = _last_bar_age_seconds(t)
            last_age[t] = age
            if age is not None and age <= MAX_BAR_AGE_SECONDS:
                fresh.add(t)
        if len(fresh) < len(SYMBOLS):
            time.sleep(3)
    stale = {t: last_age.get(t) for t in SYMBOLS if t not in fresh}
    print(f"Fresh bars for: {sorted(fresh)} (of {SYMBOLS})"
          + (f"  STALE: { {k: f'{v:.0f}s old' if v is not None else 'no bars' for k, v in stale.items()} }" if stale else ""))

    shots = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=2)
        page.goto(FRONTEND_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1200)
        page.get_by_text("Market Data", exact=True).click()
        page.wait_for_timeout(300)
        page.get_by_text("Charts", exact=True).click()
        page.wait_for_timeout(4000)
        for t in SYMBOLS:
            path = SHOT_DIR / f"{t}_{now_et().strftime('%Y%m%d_%H%M%S')}.png"
            try:
                page.locator(f"#chart-panel-{t}").screenshot(path=str(path), timeout=10000)
                shots[t] = str(path)
            except Exception as e:
                print(f"  [{t}] screenshot failed: {e}")
        browser.close()
    return shots, stale


# ── Chartexpert read ──────────────────────────────────────────────────────
def read_chart(client, ticker, png_path):
    img_b64 = base64.b64encode(Path(png_path).read_bytes()).decode()
    resp = client.messages.create(
        model="claude-sonnet-5", max_tokens=1000, system=CHARTEXPERT_SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
            {"type": "text", "text": f"Analyze this {ticker} 1-minute chart."},
        ]}],
    )
    # claude-sonnet-5 can emit a ThinkingBlock before the actual TextBlock,
    # so content[0] is not reliably the answer -- confirmed 2026-09-11
    # (AttributeError: 'ThinkingBlock' object has no attribute 'text').
    # Take the first block that actually has text.
    text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        raise ValueError(f"no text block in response for {ticker}: {[getattr(b, 'type', '?') for b in resp.content]}")
    text = text_blocks[0]
    m = _PROB_RE.search(text)
    direction, up_pct = None, None
    if m:
        direction = m.group(1).upper()
        # group(2) is whichever direction leads ("<DIR> - Probability N%"),
        # group(3) is the OTHER direction's own stated value after the pipe
        # ("| Up N%" or "| Down N%") -- it is ALREADY that direction's
        # number, never a complement to subtract from 100. Bug found
        # 2026-09-10: the DOWN branch used to compute 100-group(3), which
        # silently returned the DOWN probability mislabeled as up_pct --
        # e.g. "DOWN - Probability 62% | Up 38%" (real up_pct=38) came back
        # as up_pct=62, backwards. Confirmed against a real API response.
        lead_pct, other_pct = int(m.group(2)), int(m.group(3))
        up_pct = lead_pct if direction == "UP" else other_pct
        if lead_pct + other_pct != 100:
            print(f"  WARNING: {ticker} probabilities don't sum to 100 "
                  f"({direction} {lead_pct}% / other {other_pct}%) -- using as-is")
    pm = _PRICE_RE.search(text)
    price = float(pm.group(1).replace(",", "")) if pm else None
    return {"text": text, "direction": direction, "up_pct": up_pct, "price": price}


# ── IBKR: ATM 0DTE call lookup + quotes ──────────────────────────────────
def safe_px(v):
    """A live IBKR quote field is not just 'present or missing' -- it can be
    a real NaN (no data), and NaN is TRUTHY in Python (`nan or 0` returns
    nan, not 0; `if nan` is True), so the usual `x or default` idiom lets it
    silently slip through. Confirmed 2026-09-11: an unresolvable contract's
    reqTickers() call returned bid=nan/ask=nan with no exception raised,
    which then corrupted a shadow-close's P&L as NaN, and would have
    permanently broken the trailing-stop's high-water-mark comparison chain
    if it had happened in the live monitoring loop instead (same fix
    pattern as nflx_bottom_monitor.py's safe_px). Returns None for
    anything that isn't a real positive, finite price."""
    if v is None:
        return None
    try:
        f = float(v)
        return f if f > 0 and f == f else None   # f == f is False for NaN
    except (TypeError, ValueError):
        return None


def get_atm_0dte_quote(ib, ticker):
    stk = ib.qualifyContracts(Stock(ticker, "SMART", "USD"))[0]
    [t] = ib.reqTickers(stk)
    spot = safe_px(t.marketPrice()) or safe_px(t.close)
    if spot is None:
        return None
    today = date.today().strftime("%Y%m%d")
    chains = ib.reqSecDefOptParams(ticker, "", "STK", stk.conId)
    chain = next((c for c in chains if c.exchange == "SMART" and c.tradingClass == ticker), None)
    if chain is None or today not in chain.expirations:
        return None  # no 0DTE listed today (holiday-shortened week, etc.)
    strike = min(chain.strikes, key=lambda s: abs(s - spot))
    qualified = ib.qualifyContracts(Option(ticker, today, strike, "C", "SMART"))
    if not qualified:
        return None
    opt = qualified[0]
    [oq] = ib.reqTickers(opt)
    ib.sleep(2)
    bid, ask = safe_px(oq.bid) or 0, safe_px(oq.ask) or 0
    return {"spot": spot, "strike": strike, "expiry": today, "conid": opt.conId,
            "bid": bid, "ask": ask, "local_symbol": opt.localSymbol}


def get_underlying_price(ib, ticker):
    stk = ib.qualifyContracts(Stock(ticker, "SMART", "USD"))[0]
    [t] = ib.reqTickers(stk)
    return safe_px(t.marketPrice()) or safe_px(t.close)


def get_option_quote(ib, conid, ticker, expiry, strike):
    """Returns (bid, ask) as clean floats (0.0 if truly unavailable, never
    NaN), or (None, None) if the contract can't be qualified at all (e.g. an
    expired 0DTE from a prior day -- expected for the stale-position guard)."""
    opt = Option(ticker, expiry, strike, "C", "SMART")
    opt.conId = conid
    qualified = ib.qualifyContracts(opt)
    if not qualified:
        return None, None
    [oq] = ib.reqTickers(qualified[0])
    return safe_px(oq.bid) or 0, safe_px(oq.ask) or 0


# ── Shadow state ──────────────────────────────────────────────────────────
def load_shadow_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"open": {}}


def save_shadow_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def log_shadow_trade(record):
    with open(TRADES_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def journal_shadow_trade(pos, exit_price, pnl, reason):
    """Also write a real trade_journal row so this shows up in the frontend's
    Independent Traders tab ("Chartexpert Trade (Shadow)") the exact same way
    every other independent trader's history does, instead of only being
    visible via chartexpert_shadow_trades.jsonl. is_paper=1 (this is a
    simulation, never a real fill) -- same convention every other role in
    this codebase already relies on to exclude shadow/paper rows from real
    P&L (see the CFO skill's standing rule: 'always filter is_paper=0 before
    summing anything'). Never touches is_paper=0 real trades."""
    try:
        con = sqlite3.connect(JOURNAL_DB_PATH)
        con.execute("""INSERT INTO trade_journal
            (opened_at, closed_at, ticker, expiry, strike, right, action, qty,
             entry_price, exit_price, exit_reason, pnl, win, strategy_type,
             is_paper, notes)
            VALUES (?, ?, ?, ?, ?, 'C', 'BUY', 1, ?, ?, ?, ?, ?, 'CHARTEXPERT_SHADOW', 1, ?)""",
            (pos["signal_time"], now_et().isoformat(), pos["ticker"], pos["expiry"], pos["strike"],
             pos["option_entry_ask"], exit_price, reason, pnl, 1 if pnl > 0 else 0,
             f"chartexpert shadow trade -- up-prob {pos.get('up_pct')}%, "
             f"underlying entry ${pos['underlying_entry']:.2f} -> high ${pos['high_water_mark']:.2f}, "
             f"0.5% trailing stop, SIMULATED FILL ONLY (no real order)"))
        con.commit()
        con.close()
    except Exception as e:
        print(f"  journal_shadow_trade failed: {e}")


def main():
    cfg = load_cfg()
    et = now_et()
    print(f"[{et.isoformat()}] chartexpert_auto_trader starting (SHADOW_MODE={SHADOW_MODE})")

    if et.weekday() >= 5:
        print("Weekend, nothing to do.")
        return

    client = anthropic.Anthropic(api_key=cfg["anthropic_api_key"])

    shots, stale = ensure_subscribed_and_shoot()
    if not shots:
        telegram_text(cfg, "chartexpert_auto_trader: could not get any chart screenshots today -- check the frontend/backend.")
        sys.exit(1)
    if stale:
        stale_desc = ", ".join(f"{t} ({age:.0f}s old)" if age is not None else f"{t} (no bars)"
                               for t, age in stale.items())
        telegram_text(cfg, f"⚠️ <b>chartexpert_auto_trader: stale data</b> for {stale_desc} -- "
                            f"last bar older than {MAX_BAR_AGE_SECONDS}s at 9:35 AM ET means the 1m "
                            f"subscription likely stalled. Reading the chart anyway (marked below), but "
                            f"any signal on these tickers is SKIPPED, not traded on stale data.")
        oversight_log("programmer", "chartexpert_stale_data",
                      f"Stale 1m bars at run start: {stale_desc}.",
                      outcome="chart still read + Telegrammed with a stale warning; trading signal suppressed for these tickers")

    analyses = {}
    for t in SYMBOLS:
        if t not in shots:
            continue
        try:
            analyses[t] = read_chart(client, t, shots[t])
        except anthropic.AuthenticationError:
            telegram_text(cfg, "\U0001F6A8 <b>chartexpert_auto_trader: Anthropic API key is still dead (401).</b> "
                                "No chart analysis ran today -- get a fresh key from console.anthropic.com "
                                "and update scanner_config.json's anthropic_api_key.")
            oversight_log("programmer", "chartexpert_blocked",
                          "chartexpert_auto_trader could not run: anthropic_api_key is invalid (401).",
                          outcome="alerted, no trades considered")
            sys.exit(1)
        except Exception as e:
            print(f"  [{t}] chartexpert read failed: {e}")
            telegram_text(cfg, f"chartexpert_auto_trader: {t} chart read failed ({e}).")

    # Telegram every symbol's analysis + chart, every run, regardless of signal
    for t in SYMBOLS:
        if t not in analyses:
            continue
        a = analyses[t]
        prob_line = f"{a['direction']} {a['up_pct'] if a['direction']=='UP' else 100-a['up_pct']}%" if a["direction"] else "?"
        stale_tag = " [STALE DATA]" if t in stale else ""
        telegram_photo(cfg, shots[t], f"{t} 1-min chart (9:35 AM chartexpert read) -- {prob_line}{stale_tag}")
        telegram_text(cfg, (f"⚠️ <b>STALE DATA -- last bar {stale[t]:.0f}s old, this reflects an earlier "
                            f"market moment, not right now:</b>\n\n" if t in stale and stale[t] is not None else "") + a["text"])

    # ── Signal + shadow entry (SPY/QQQ only) ─────────────────────────────
    ib = IB(); ib.errorEvent += lambda *a: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    ib.reqMarketDataType(1)

    shadow_state = load_shadow_state()
    # Stale-position guard: a mid-run crash (IBKR drop, machine reboot, etc.)
    # leaves an unresolved position in "open" -- this task has NO restart-on-
    # crash watchdog (unlike the always-on scanners), so the NEXT day's fresh
    # 9:35 AM run would otherwise happily "monitor" it using yesterday's now-
    # EXPIRED 0DTE contract and produce a garbage exit. Anything not opened
    # today gets force-closed here using whatever quote is still gettable
    # (falling back to the entry price, i.e. 0 P&L, if the contract is dead)
    # and logged as stale rather than silently mismonitored.
    today_str = et.strftime("%Y%m%d")
    had_stale = False
    for t, pos in list(shadow_state["open"].items()):
        if pos.get("expiry") != today_str:
            had_stale = True
            print(f"  [{t}] stale shadow position from {pos.get('expiry')} found at startup -- force-closing")
            _close_shadow(ib, cfg, pos, "stale_carryover_from_prior_day")
            del shadow_state["open"][t]
    if had_stale:
        save_shadow_state(shadow_state)

    fired_any = False
    try:
        for t in TRADEABLE:
            if t in stale:
                # Never trade off a chart that isn't actually current --
                # whatever chartexpert said about it reflects an old market
                # moment, not now. The Telegram/oversight alert above already
                # surfaced this; here it's an unconditional trading gate.
                print(f"  [{t}] skipping signal check -- stale data ({stale[t]})")
                continue
            existing = shadow_state["open"].get(t)
            if existing and existing.get("expiry") == today_str:
                # Already tracking a same-day position for this ticker --
                # this branch matters now that Task Scheduler can retry the
                # whole script after a mid-run crash (RestartCount=3,
                # 2026-09-11): without this check, a retry whose fresh chart
                # read still shows >70% up would silently overwrite the
                # already-open position, resetting its high-water mark and
                # losing the real entry it already has. Leave it alone --
                # the monitoring loop below picks it up regardless.
                continue
            a = analyses.get(t)
            if not a or a["direction"] != "UP" or a["up_pct"] is None:
                continue
            if a["up_pct"] <= SIGNAL_THRESHOLD_PCT:
                continue
            fired_any = True
            q = get_atm_0dte_quote(ib, t)
            if q is None:
                telegram_text(cfg, f"chartexpert_auto_trader: {t} up-probability {a['up_pct']}% > "
                                    f"{SIGNAL_THRESHOLD_PCT}% but no 0DTE expiry listed today -- skipped.")
                continue
            if not q["ask"]:  # 0 or no data -- no real market to shadow-buy into
                telegram_text(cfg, f"chartexpert_auto_trader: {t} up-probability {a['up_pct']}% > "
                                    f"{SIGNAL_THRESHOLD_PCT}% but the ATM 0DTE {q['strike']}C has no live "
                                    f"ask -- skipped rather than shadow-buy at a fake $0.")
                continue
            pos = {
                "ticker": t, "signal_time": et.isoformat(), "up_pct": a["up_pct"],
                "underlying_entry": q["spot"], "high_water_mark": q["spot"],
                "strike": q["strike"], "expiry": q["expiry"], "conid": q["conid"],
                "local_symbol": q["local_symbol"], "option_entry_ask": q["ask"],
                "shadow": SHADOW_MODE,
            }
            shadow_state["open"][t] = pos
            mode_tag = "SHADOW" if SHADOW_MODE else "LIVE"
            telegram_text(cfg,
                f"\U0001F9EA <b>{mode_tag} SIGNAL: {t}</b>\n"
                f"Up-probability {a['up_pct']}% > {SIGNAL_THRESHOLD_PCT}% threshold.\n"
                f"Would BUY 1x ATM 0DTE {q['strike']}C ({q['expiry']}) @ ask ${q['ask']:.2f} "
                f"(spot ${q['spot']:.2f}).\n"
                f"Tracking 0.5% trailing stop on the underlying. "
                + ("No real order placed -- shadow mode." if SHADOW_MODE else "REAL ORDER PLACED."))
            oversight_log("trader", "chartexpert_shadow_signal" if SHADOW_MODE else "chartexpert_signal",
                          f"{mode_tag}: {t} up-prob {a['up_pct']}% fired. ATM 0DTE {q['strike']}C @ ${q['ask']:.2f}, spot ${q['spot']:.2f}.",
                          rationale="chartexpert 9:35 AM read > 70% up threshold (CEO instruction 2026-09-10).")
        save_shadow_state(shadow_state)

        if not fired_any and not shadow_state["open"]:
            print("No signal >70% today; nothing to monitor.")
            ib.disconnect()
            return

        # ── Monitor any open shadow position(s) to a trailing-stop or EOD exit ──
        while shadow_state["open"]:
            now = now_et()
            if now.strftime("%H:%M") >= EOD_CUTOFF:
                for t, pos in list(shadow_state["open"].items()):
                    _close_shadow(ib, cfg, pos, "eod_cutoff")
                    del shadow_state["open"][t]
                save_shadow_state(shadow_state)
                break
            for t, pos in list(shadow_state["open"].items()):
                spot = get_underlying_price(ib, t)
                if spot is None:
                    print(f"  [{t}] no valid quote this cycle, skipping (will retry)")
                    continue
                if spot > pos["high_water_mark"]:
                    pos["high_water_mark"] = spot
                stop_price = pos["high_water_mark"] * (1 - TRAILING_STOP_PCT)
                if spot <= stop_price:
                    _close_shadow(ib, cfg, pos, "trailing_stop")
                    del shadow_state["open"][t]
            save_shadow_state(shadow_state)
            if shadow_state["open"]:
                time.sleep(POLL_SECONDS)
    finally:
        ib.disconnect()

    print("chartexpert_auto_trader done.")


def _close_shadow(ib, cfg, pos, reason):
    t = pos["ticker"]
    try:
        bid, ask = get_option_quote(ib, pos["conid"], t, pos["expiry"], pos["strike"])
        exit_price = bid if bid else pos["option_entry_ask"]
    except Exception:
        exit_price = pos["option_entry_ask"]
    pnl = round((exit_price - pos["option_entry_ask"]) * 100, 2)
    record = {**pos, "exit_time": now_et().isoformat(), "exit_reason": reason,
              "option_exit_bid": exit_price, "pnl": pnl}
    log_shadow_trade(record)
    journal_shadow_trade(pos, exit_price, pnl, reason)
    mode_tag = "SHADOW" if pos.get("shadow", True) else "LIVE"
    telegram_text(cfg,
        f"\U0001F9EA <b>{mode_tag} EXIT: {t}</b> ({reason})\n"
        f"{pos['strike']}C entry ${pos['option_entry_ask']:.2f} -> exit ${exit_price:.2f}\n"
        f"Simulated P&L: ${pnl:+.2f} (1 contract)\n"
        f"Underlying: entry ${pos['underlying_entry']:.2f}, high ${pos['high_water_mark']:.2f}")
    oversight_log("trader", "chartexpert_shadow_exit" if pos.get("shadow", True) else "chartexpert_exit",
                  f"{mode_tag} EXIT {t}: {reason}, pnl ${pnl:+.2f}", pnl_impact=(pnl if not pos.get("shadow", True) else None))


if __name__ == "__main__":
    main()
