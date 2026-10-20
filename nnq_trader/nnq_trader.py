#!/usr/bin/env python3
"""NNQ chartexpert trader (user ask 2026-10-19). Every 10 minutes, at each 10-minute bar close while
CME's NNQ market is open (nearly 24 h: Sun 18:00 - Fri 17:00 ET, daily 17:00-18:00 break):

  1. pull NNQ 10-minute bars from IBKR and draw the chart (candles, EMA 9/21/50, volume, RSI 14)
  2. have Claude read it in the chartexpert format -> "UP - Probability XX% | Down XX%"
  3. UP > 60%   -> be LONG 1 NNQ      DOWN > 60% -> be SHORT 1 NNQ      otherwise: keep what we hold
     Rules (user, 2026-10-19): hold until the opposite signal, then flip (one 2-lot order). No stop,
     no target, positions carry overnight and over weekends.

PAPER ONLY: refuses unless every managed account starts with DU. Max position 1 contract. Orders are
marketable limits (ask/bid +-2 ticks, widened on retry) so a thin overnight book can't fill at a silly
price. Rolls to the next contract month 7 days before expiry, keeping the same direction.

Needs a valid anthropic_api_key in ibkr_trader/backend/scanner_config.json. Without one it keeps
running and alerts, but never trades on a chart it couldn't read.

    python nnq_trader.py            # run forever (watchdog run_nnq_trader.ps1)
    python nnq_trader.py --once     # one read + decision now, then exit (test)
    python nnq_trader.py --dry-run  # read + decide, log what it WOULD do, place nothing
"""
import asyncio
import base64
import json
import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import requests  # noqa: E402
from ib_insync import IB, Future, LimitOrder  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG_PATH = HERE.parent / "ibkr_trader" / "backend" / "scanner_config.json"
LOG = HERE / "nnq_log.jsonl"
STATE = HERE / "nnq_state.json"
CHARTS = HERE / "nnq_charts"
CHARTS.mkdir(exist_ok=True)

ET = ZoneInfo("America/New_York")
SYMBOL, EXCHANGE, TICK, MULT = "NNQ", "CME", 0.5, 0.2
CLIENT_ID, PORT = 1660, 7497
THRESHOLD = 60            # strictly greater than this
QTY = 1
ROLL_DAYS = 7
BAR_SECONDS = 600
READ_DELAY_S = 20         # after the bar boundary, so IBKR has closed the bar
MODEL = "claude-opus-5-5"

SYSTEM = """You are ChartWise Analyst -- an expert futures and equity chart reader.
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
or labels that aren't actually there. When the chart gives no real edge, say so with probabilities
near 50/50 rather than forcing a direction."""

PROB_RE = re.compile(r"(UP|DOWN)\s*-\s*Probability\s*(\d+)%\s*\|\s*(?:Up|Down)\s*(\d+)%", re.I)


def now():
    return datetime.now(ET)


def cfg():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def log(action, **kw):
    rec = {"time": now().isoformat(), "action": action, **kw}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(f"[{now():%m-%d %H:%M:%S}] {action}: {json.dumps(kw, default=str)[:260]}", flush=True)


def tg(text, photo=None):
    c = cfg()
    base = f"https://api.telegram.org/bot{c['telegram_token']}"
    try:
        if photo:
            with open(photo, "rb") as f:
                requests.post(f"{base}/sendPhoto", data={"chat_id": c["telegram_chat_id"],
                              "caption": ("NNQ: " + text)[:1000]}, files={"photo": f}, timeout=20)
        else:
            requests.post(f"{base}/sendMessage", json={"chat_id": c["telegram_chat_id"],
                          "text": "NNQ: " + text}, timeout=10)
    except Exception as e:
        print(f"telegram failed: {e}")


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"fills": [], "alerts": {}}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1, default=str))


def alert_once(st, key, text, every_s=3600):
    """At most one Telegram per `key` per `every_s` (e.g. a dead API key, every 10 minutes)."""
    last = st["alerts"].get(key)
    if last and (now() - datetime.fromisoformat(last)).total_seconds() < every_s:
        return
    st["alerts"][key] = now().isoformat()
    tg(text)


def keep_awake(on=True):
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000003 if on else 0))


# ----------------------------------------------------------------- market
def parse_hours(hours, tzname):
    """IBKR tradingHours '20261018:1700-20261019:1600;20261019:CLOSED;...' -> [(start, end)] aware."""
    tz = ZoneInfo(tzname)
    out = []
    for part in hours.split(";"):
        if "CLOSED" in part or "-" not in part:
            continue
        a, b = part.split("-")
        out.append((datetime.strptime(a, "%Y%m%d:%H%M").replace(tzinfo=tz),
                    datetime.strptime(b, "%Y%m%d:%H%M").replace(tzinfo=tz)))
    return out


async def front_contract(ib):
    """Nearest NNQ month with more than ROLL_DAYS left, plus its trading sessions."""
    cds = await ib.reqContractDetailsAsync(Future(SYMBOL, exchange=EXCHANGE))
    cds.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
    cutoff = (now() + timedelta(days=ROLL_DAYS)).strftime("%Y%m%d")
    d = next(d for d in cds if d.contract.lastTradeDateOrContractMonth[:8] > cutoff)
    return d.contract, parse_hours(d.tradingHours, d.timeZoneId)


def is_open(sessions, t=None):
    t = t or now()
    return any(a <= t < b for a, b in sessions)


async def quote(ib, c):
    t = ib.reqMktData(c, "", False, False)
    for _ in range(20):
        await asyncio.sleep(0.25)
        if t.bid == t.bid and t.ask == t.ask and t.bid and t.ask and t.bid > 0:
            break
    ib.cancelMktData(c)
    ok = lambda v: v if v == v and v and v > 0 else None
    return ok(t.bid), ok(t.ask)


async def held(ib):
    """{conId: (contract, signed qty)} for every NNQ month on the account."""
    out = {}
    for p in ib.positions():
        if p.contract.symbol == SYMBOL and p.contract.secType == "FUT" and p.position:
            out[p.contract.conId] = (p.contract, int(p.position))
    return out


# ----------------------------------------------------------------- chart
def ema(vals, n):
    k, out, e = 2 / (n + 1), [], None
    for v in vals:
        e = v if e is None else v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(vals, n=14):
    out, g, l = [], 0.0, 0.0
    for i, v in enumerate(vals):
        if i == 0:
            out.append(50.0)
            continue
        d = v - vals[i - 1]
        g = (g * (n - 1) + max(d, 0)) / n if i > n else g + max(d, 0) / n
        l = (l * (n - 1) + max(-d, 0)) / n if i > n else l + max(-d, 0) / n
        out.append(100 - 100 / (1 + g / l) if l else 100.0)
    return out


def draw(bars, path, n=120):
    closes = [b.close for b in bars]
    e9, e21, e50, r = ema(closes, 9), ema(closes, 21), ema(closes, 50), rsi(closes)
    bars, e9, e21, e50, r = bars[-n:], e9[-n:], e21[-n:], e50[-n:], r[-n:]
    x = range(len(bars))
    fig, (ax, av, ar) = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [5, 1.2, 1.5]})
    for i, b in zip(x, bars):
        up = b.close >= b.open
        col = "#26a69a" if up else "#ef5350"
        ax.vlines(i, b.low, b.high, color=col, linewidth=0.8)
        ax.add_patch(plt.Rectangle((i - 0.35, min(b.open, b.close)), 0.7, max(abs(b.close - b.open), TICK / 2),
                                   color=col))
        av.bar(i, b.volume, color=col, width=0.7)
    ax.plot(x, e9, color="#ffb300", lw=1, label="EMA 9")
    ax.plot(x, e21, color="#1e88e5", lw=1, label="EMA 21")
    ax.plot(x, e50, color="#8e24aa", lw=1.2, label="EMA 50")
    last = bars[-1]
    ax.axhline(last.close, color="#555", lw=0.6, ls="--")
    ax.text(len(bars) - 0.5, last.close, f" {last.close:,.2f}", va="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#555"))
    ax.set_title(f"NNQ (Nano Nasdaq-100 futures) - 10-minute - last bar {last.date:%Y-%m-%d %H:%M} ET",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    av.set_ylabel("Vol", fontsize=8)
    ar.plot(x, r, color="#5e35b1", lw=1)
    for lvl in (30, 70):
        ar.axhline(lvl, color="#999", lw=0.6, ls="--")
    ar.set_ylim(0, 100)
    ar.set_ylabel("RSI 14", fontsize=8)
    ticks = list(range(0, len(bars), max(1, len(bars) // 10)))
    ar.set_xticks(ticks)
    ar.set_xticklabels([f"{bars[i].date:%m-%d %H:%M}" for i in ticks], rotation=30, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------- read
def read_chart(client, png):
    data = base64.standard_b64encode(Path(png).read_bytes()).decode()
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000, system=SYSTEM,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
            {"type": "text", "text": "Analyze this NNQ 10-minute chart."}]}],
    )
    if resp.stop_reason == "refusal":
        raise ValueError(f"read refused: {getattr(resp.stop_details, 'category', None)}")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    m = PROB_RE.search(text)
    if not m:
        raise ValueError("no 'UP/DOWN - Probability' line in the read")
    lead, a, b = m.group(1).upper(), int(m.group(2)), int(m.group(3))
    up, down = (a, b) if lead == "UP" else (b, a)
    return {"text": text, "lead": lead, "up": up, "down": down, "model": resp.model}


def target_position(read, current):
    """Hold until the opposite signal (user rule): only a >THRESHOLD read moves the position."""
    if read["up"] > THRESHOLD and read["up"] > read["down"]:
        return QTY
    if read["down"] > THRESHOLD and read["down"] > read["up"]:
        return -QTY
    return current


# ----------------------------------------------------------------- orders
def round_tick(px, up):
    q = px / TICK
    return round((math.ceil(q) if up else math.floor(q)) * TICK, 2)


async def trade(ib, c, qty_delta, why, st):
    """Marketable limit at ask/bid +-2, then +-4, then +-8 ticks; 15 s each. Returns avg fill or None."""
    side = "BUY" if qty_delta > 0 else "SELL"
    n = abs(qty_delta)
    for step, ticks in enumerate((2, 4, 8)):
        bid, ask = await quote(ib, c)
        ref = ask if side == "BUY" else bid
        if ref is None:
            await asyncio.sleep(3)
            continue
        px = round_tick(ref + ticks * TICK if side == "BUY" else ref - ticks * TICK, side == "BUY")
        tr = ib.placeOrder(c, LimitOrder(side, n, px, tif="GTC", outsideRth=True))
        for _ in range(15):
            await asyncio.sleep(1)
            if tr.orderStatus.status == "Filled":
                fill = float(tr.orderStatus.avgFillPrice)
                st["fills"].append({"time": now().isoformat(), "side": side, "qty": n, "px": fill,
                                    "conId": c.conId, "why": why})
                log("filled", side=side, qty=n, px=fill, limit=px, step=step + 1, bid=bid, ask=ask, why=why)
                return fill
        ib.cancelOrder(tr.order)
        for _ in range(10):
            await asyncio.sleep(1)
            if tr.orderStatus.status in ("Cancelled", "ApiCancelled", "Filled", "Inactive"):
                break
        if tr.orderStatus.status == "Filled":
            fill = float(tr.orderStatus.avgFillPrice)
            st["fills"].append({"time": now().isoformat(), "side": side, "qty": n, "px": fill,
                                "conId": c.conId, "why": why})
            log("filled", side=side, qty=n, px=fill, limit=px, step=step + 1, why=why, note="raced cancel")
            return fill
        log("order_step_unfilled", side=side, qty=n, limit=px, bid=bid, ask=ask, status=tr.orderStatus.status)
    tg(f"{side} {n} NNQ for '{why}' did NOT fill after 3 tries -- position unchanged, will retry next bar")
    return None


def realized(fills):
    """FIFO realized P&L ($) and open (qty, avg) from this strategy's own fills."""
    lots, pnl = [], 0.0  # lots: [signed qty, px]
    for f in fills:
        q = f["qty"] if f["side"] == "BUY" else -f["qty"]
        while q and lots and (lots[0][0] > 0) != (q > 0):
            take = min(abs(q), abs(lots[0][0]))
            sign = 1 if lots[0][0] > 0 else -1
            pnl += sign * (f["px"] - lots[0][1]) * take * MULT
            lots[0][0] -= sign * take
            q += sign * take
            if lots[0][0] == 0:
                lots.pop(0)
        if q:
            lots.append([q, f["px"]])
    pos = sum(l[0] for l in lots)
    avg = sum(l[0] * l[1] for l in lots) / pos if pos else None
    return round(pnl, 2), pos, avg


# ----------------------------------------------------------------- loop
async def cycle(ib, client, st, dry):
    c, sessions = await front_contract(ib)
    if not is_open(sessions):
        return "closed"
    # formatDate=2 -> tz-aware UTC bar times. With 1 they come back naive in TWS's own zone (US/Central
    # here), which read as ET put every bar an hour off (caught in the 2026-10-19 dry run).
    bars = await ib.reqHistoricalDataAsync(c, "", "5 D", "10 mins", "TRADES", useRTH=False, formatDate=2)
    for b in bars:
        b.date = b.date.astimezone(ET)
    bars = [b for b in bars if b.date + timedelta(seconds=BAR_SECONDS) <= now()]  # closed bars only
    if len(bars) < 60:
        log("skip", reason=f"only {len(bars)} closed bars")
        return "no bars"
    age = (now() - (bars[-1].date + timedelta(seconds=BAR_SECONDS))).total_seconds()
    if age > 3 * BAR_SECONDS:
        alert_once(st, "stale", f"last 10-min bar is {age / 60:.0f} min old -- skipping reads until data is fresh")
        log("skip", reason=f"stale bars ({age:.0f}s)")
        return "stale"
    png = CHARTS / f"NNQ_{now():%Y%m%d_%H%M}.png"
    draw(bars, png)
    for old in sorted(CHARTS.glob("NNQ_*.png"))[:-300]:
        old.unlink(missing_ok=True)
    try:
        rd = read_chart(client, png)
    except anthropic.AuthenticationError:
        alert_once(st, "auth", "Anthropic API key is invalid (401) -- no chart reads, NO trades. Put a fresh key in "
                               "scanner_config.json (anthropic_api_key); the trader picks it up within 10 minutes.")
        log("read_failed", reason="401 invalid API key")
        return "auth"
    except anthropic.BadRequestError as e:
        if "workspace" in str(e):
            alert_once(st, "auth", "Anthropic key is user-scoped and needs a workspace -- set anthropic_workspace_id "
                                   "in scanner_config.json (or use a workspace-scoped key). NO trades until then.")
        log("read_failed", reason=f"400: {str(e)[:150]}")
        return "auth"
    except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError) as e:
        log("read_failed", reason=f"{type(e).__name__}: {str(e)[:150]}")
        alert_once(st, "readerr", f"chart read failed ({type(e).__name__}) -- holding the current position", 1800)
        return "read error"

    pos = sum(q for cc, q in (await held(ib)).values())
    tgt = target_position(rd, pos)
    log("read", up=rd["up"], down=rd["down"], lead=rd["lead"], price=bars[-1].close, position=pos, target=tgt,
        chart=png.name, model=rd["model"])
    if abs(tgt) > QTY:
        raise RuntimeError(f"target {tgt} exceeds max position {QTY}")
    if tgt == pos:
        return "hold"
    delta = tgt - pos
    why = (f"{'UP' if tgt > 0 else 'DOWN'} {max(rd['up'], rd['down'])}% > {THRESHOLD}%"
           + (" (flip)" if pos and abs(delta) == 2 else ""))
    if dry:
        log("dry_run", would=("BUY" if delta > 0 else "SELL"), qty=abs(delta), why=why)
        return "dry"
    fill = await trade(ib, c, delta, why, st)
    if fill is not None:
        pnl, opos, avg = realized(st["fills"])
        tg(f"{'BUY' if delta > 0 else 'SELL'} {abs(delta)} @ {fill:,.2f} -- now {'LONG' if tgt > 0 else 'SHORT'} 1. "
           f"Read: UP {rd['up']}% / DOWN {rd['down']}%. Realized to date ${pnl:+.2f}.", photo=png)
    save_state(st)
    return "traded"


async def roll_if_needed(ib, st, dry):
    """Holding a month that's within ROLL_DAYS of expiry -> move it to the front contract, same direction."""
    c, _ = await front_contract(ib)
    for con_id, (oc, q) in (await held(ib)).items():
        if con_id == c.conId:
            continue
        await ib.qualifyContractsAsync(oc)
        log("roll", from_=oc.localSymbol, to=c.localSymbol, qty=q)
        if dry:
            continue
        if await trade(ib, oc, -q, f"roll out of {oc.localSymbol}", st) is not None:
            await trade(ib, c, q, f"roll into {c.localSymbol}", st)
        save_state(st)


async def connect(ib, st):
    while True:
        try:
            await ib.connectAsync("127.0.0.1", PORT, clientId=CLIENT_ID, timeout=15)
            accts = ib.managedAccounts()
            if not accts or not all(a.startswith("DU") for a in accts):
                ib.disconnect()
                tg(f"refusing to trade: TWS on {PORT} is logged into {accts}, not paper")
                raise SystemExit(2)
            st["alerts"].pop("tws", None)
            return
        except SystemExit:
            raise
        except Exception as e:
            alert_once(st, "tws", f"TWS paper not answering ({type(e).__name__}) -- retrying every 30 s", 1800)
            await asyncio.sleep(30)


async def main():
    once, dry = "--once" in sys.argv, "--dry-run" in sys.argv
    st = load_state()
    ib = IB()
    await connect(ib, st)
    keep_awake(True)
    log("start", once=once, dry=dry, model=MODEL, threshold=THRESHOLD)
    try:
        while True:
            if not ib.isConnected():
                await connect(ib, st)
            c_ = cfg()  # re-read every cycle so a key/workspace fix applies without a restart
            ws = (c_.get("anthropic_workspace_id") or "").strip()
            # A user-scoped key (sk-ant-usr...) must name its workspace on every request.
            client = anthropic.Anthropic(api_key=c_.get("anthropic_api_key", ""),
                                         default_headers={"anthropic-workspace-id": ws} if ws else None)
            try:
                await roll_if_needed(ib, st, dry)
                result = await cycle(ib, client, st, dry)
            except Exception as e:
                log("cycle_error", error=f"{type(e).__name__}: {str(e)[:200]}")
                alert_once(st, "cycle", f"cycle error {type(e).__name__}: {str(e)[:120]}", 1800)
                result = "error"
            save_state(st)
            if once:
                print("result:", result)
                return
            t = now()
            nxt = t.replace(second=0, microsecond=0) + timedelta(minutes=10 - t.minute % 10)
            await asyncio.sleep(max(5.0, (nxt - t).total_seconds() + READ_DELAY_S))
    finally:
        keep_awake(False)
        ib.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
