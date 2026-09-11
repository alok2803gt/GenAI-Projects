"""
"Safe Income Trader" -- automates Tier 1: safe-income-screener's validated
deep-OTM credit-spread methodology -- the one strategy on this account
with a real backtest
behind it (95.4% puts / 89.8% calls stay safe at 15% cushion, ~15,900
simulated entries over 2yrs real price history) and a real track record
that matches it (MRVL, UNH -- trade-history-playbook's cleanest, most
boring wins). Previously manual-only: a human read the screener's ranked
output and placed trades by hand via Manual Trader. This closes that loop.

EXIT RULE: HOLD TO EXPIRY. Matches the real MRVL/UNH pattern -- no active
stop-loss/profit-target management for a structure this deep OTM; the
backtest itself already prices in the full hold-to-expiry outcome. This
script is entry-only, by design -- no monitor loop, nothing to babysit
intraday. Positions just sit until they expire.

POSITION COUNT IS DYNAMIC, not fixed. It walks safe-income-screener's own
ranked candidate list (same cushion/delta/liquidity/earnings-blackout/
sector-cap gates already in screen.py -- reused via `screen.py --json`
subprocess, not reimplemented) and, for each candidate in order, runs a
real-time CRO/CFO pre-trade review against CURRENT capital headroom
(recomputed after every fill, since headroom shrinks as positions are
added within the same run). Stops when headroom, --max-new, or the
candidate list runs out -- whichever comes first.

Deliberately a NEW, separate module -- not wired into the existing
AutoTrader, which runs a different (Kelly-sized, model-driven, and
historically capital-miscalibrated) CSP methodology. Keeps the one
validated approach traceable to its own backtest instead of diluted into
AutoTrader's legacy logic.

SAFETY RAILS -- same standing pattern as EVC / Day Trader / SPY 0DTE:
  - CRO/CFO pre-trade review before EVERY individual entry (not just an
    aggregate check): real capital-budget check (cro_cfo_capital_budget,
    50% total portfolio risk budget, 5% per-trade cap) + what-if P&L
    scenarios using the screener's own real strikes/credit/max_risk.
  - Sequential single-leg IBKR execution (long leg first, then short) --
    same proven pattern used for every real fill this account has placed
    (migrated from Alpaca 2026-09-07, zero Alpaca dependency now).
  - qty fixed at 1 contract per position -- new code path, small sizing
    until a clean live run is confirmed (this account's standing rule).
  - Real available funds checked before each attempt -- aborts
    cleanly (no partial fill risked) if insufficient.
  - Skips any ticker that already has an open IBKR option position
    (this strategy's or any other's) -- avoids double-entry, same
    cross-strategy-concentration concern the CRO checks for.
  - A pause-flag file (safe_income_auto_PAUSED.flag): any partial fill or
    unexpected state halts further entries this run AND blocks all future
    runs until a human clears it -- same incident-handling pattern as
    spy_0dte_auto.py.

Usage: python safe_income_auto.py [--dry-run] [--max-new 3]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import IB, Option

sys.path.insert(0, str(Path(__file__).parent))
from alpaca_0dte_common import cro_cfo_capital_budget, get_quote, safe_px  # noqa: E402
from ibkr_0dte_common import ibkr_place_leg_with_ladder  # noqa: E402

ET = ZoneInfo("America/New_York")


def market_is_open() -> bool:
    """Weekday + 9:30-16:00 ET, real timezone-aware (handles DST correctly).
    Does NOT account for market holidays -- same simplification as the rest
    of this codebase's scheduling. Scheduled hourly 24/7 at the OS level
    (Task Scheduler doesn't cleanly express 'hourly, market-hours-only' as
    a single trigger -- same reasoning as the weekday guard already used in
    portfolio_oversight_check.py/cro_risk_check.py); this function is what
    actually enforces the real constraint."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t

BACKEND_DIR = Path(__file__).parent
SCREENER_PATH = r"C:\Users\AlokD\.claude\skills\safe-income-screener\screen.py"
PYTHON = r"C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
STATE_FILE = BACKEND_DIR / "safe_income_auto_state.json"
LOG_FILE = BACKEND_DIR / "oversight_log.jsonl"
PAUSE_FLAG = BACKEND_DIR / "safe_income_auto_PAUSED.flag"
TWS_PORT = 7496

# Real measured rate (2026-08-24, live market hours): ~1.6 min/ticker for a
# full brute-force grid-search scan (real per-candidate IBKR qualifyContracts
# round-trips) -- the original 112-ticker universe took ~2h09m confirmed live,
# structurally incompatible with any reasonable per-run timeout. First fix
# was a rotating 12-ticker chunk. CEO then asked to narrow the universe
# itself instead -- originally the top 25 by 20-day avg dollar volume
# (yfinance, same avg_dollar_vol metric breakout_scanner.py's F9 gate uses).
#
# REVISED same day: liquidity-only ranking was a real mistake on its own --
# checked realized_vol_pct (screen.py's own function) for all 25 and found
# 14 of 25 (56%) exceed screen.py's own high_vol_threshold (40%, its default)
# -- MU 105%, MRVL 121%, PLTR 84%, INTC 84%, LRCX 86%, AMAT 86%, TSLA 56%,
# AMD 76%, etc. screen.py only WARNS on this (stderr, informational), it does
# NOT gate the candidate -- confirmed live: TSLA and AMD both cleared every
# real safety gate (cushion 15-20%, prob 82-91%) despite being flagged HIGH.
# That's fine for a human using the interactive screener with judgment; it's
# NOT fine for an unattended automated trader. This list is now the 11
# tickers from the original top-25 that are NOT high-vol (still liquidity-
# ranked within that subset) -- SPY/QQQ/NVDA/AAPL/GOOGL/IWM/GLD/LLY/TLT/WMT/V.
# Also added an explicit vol check below (HIGH_VOL_THRESHOLD) as a second,
# defense-in-depth gate directly in this script's own pre-trade review --
# not just relying on the universe being pre-filtered, in case a ticker's
# vol drifts up later or the universe list changes again without this
# constraint in mind.
HIGH_VOL_THRESHOLD = 40.0  # matches screen.py's own --high-vol-threshold default

# Lowered from screen.py's own 15% default to 12% on 2026-08-24, specifically for
# this curated, options-liquid, low-vol UNIVERSE (not a change to screen.py's own
# default). Real backtest_cushion.py run against this exact 11-ticker universe
# (5y lookback) showed 12% cushion still holds 97.1% put / 90.4% call win@exp --
# in the same range as the account's existing validated 15%-on-full-universe bar
# (95.4%/89.8%) -- while a live re-scan showed 15% cushion produced 0/11 passing
# candidates (roi gate: thin premium is the direct cost of a calm, low-vol
# universe) vs 3/11 at 10% cushion. 12% splits the difference: real throughput
# without giving up as much of the original safety margin as 10% would.
MIN_CUSHION_PCT = 12.0
# Widened 2026-09-03 after a real oversight review found 22 straight
# unattended runs (8/24-9/2) with ZERO entries, all real rejections: NVDA
# on realized vol (structural, won't change day to day) and GOOGL on the
# 5% per-trade cap (see NARROW_WIDTH_PCTS below for that fix). Candidates
# below came from IBKR-SafeIncomeScreenerLog's separate full-113-ticker
# daily log (picks_log.jsonl) -- but that log's own realized_vol_pct field
# shows most of its "passing" names (AMD 76%, MRVL 116%, PLTR 82%, MU 103%,
# etc.) would fail this script's OWN HIGH_VOL_THRESHOLD=40 gate exactly like
# NVDA does, so adding them would just trade one rejected ticker for
# another. Only UNH (25%), NKE (34%), MPC (36%), GS (39%) actually clear
# that gate on their real logged realized_vol_pct -- those are the four
# real additions.
UNIVERSE = [
    "SPY", "QQQ", "NVDA", "AAPL", "GOOGL", "IWM", "GLD", "LLY", "TLT", "WMT", "V",
    "UNH", "NKE", "MPC", "GS",
]
# Progressively narrower long-leg widths (vs screen.py's global 3% default)
# tried, in order, ONLY when a candidate is rejected purely on trade SIZE
# (the CFO per-trade-cap or portfolio-headroom check) -- never for a CRO
# high-vol veto, which a narrower spread can't fix. Real strike-increment
# rounding means a given width_pct may not land on a real narrower strike
# for a given ticker/price -- each attempt is checked against the actual
# returned max_risk, nothing here is assumed to work ahead of time.
NARROW_WIDTH_PCTS = [0.015, 0.008, 0.004]
CLIENT_ID = 1620
QTY = 1


def load_config():
    with open(BACKEND_DIR / "scanner_config.json") as f:
        return json.load(f)


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"positions": {}, "runs": []}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def telegram(cfg: dict, msg: str, high_priority: bool = False):
    import requests
    try:
        prefix = "🚨 " if high_priority else ""
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": prefix + msg, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as e:
        print(f"Telegram send failed: {e}")


def log_oversight(summary: str, outcome: str, rationale: str, pnl_impact=None):
    entry = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(),
        "actor": "trader",
        "category": "safe_income_auto",
        "summary": summary,
        "rationale": rationale,
        "outcome": outcome,
        "pnl_impact": pnl_impact,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def occ_symbol(ticker: str, expiry_yyyymmdd: str, right: str, strike: float) -> str:
    """Standard OCC option symbol: TICKER + YYMMDD + C/P + strike*1000, 8-digit zero-padded."""
    yymmdd = expiry_yyyymmdd[2:]
    return f"{ticker}{yymmdd}{right}{int(round(strike * 1000)):08d}"


def get_screener_candidates(tickers: str | None = None, timeout_s: int = 1800,
                             width_pct: float | None = None) -> list[dict]:
    """Runs safe-income-screener's real scan (same gates: cushion, delta,
    liquidity, earnings-blackout, sector-cap) via subprocess + --json, same
    integration pattern main.py's Telegram /screen command already uses.
    Does NOT reimplement any of screen.py's logic.

    tickers: comma-separated subset passthrough (screen.py's own --tickers)
    -- for testing/bounding runtime. A full 112-ticker scan does real
    per-candidate IBKR contract-qualification round-trips and can
    genuinely take longer than 30 min outside active market hours
    (confirmed: hit this exact timeout at 2am 2026-08-24) -- this hasn't
    yet been timed during real market hours, where quote/liquidity checks
    should resolve much faster.

    width_pct: passthrough to screen.py's own --width-pct (added 2026-09-03)
    -- narrows the long leg's target distance for THIS call only, used by
    the size-rejection retry below. Never changes the module-level default
    used for the main universe scan."""
    # Distinct clientId from both this script's own IBKR connection (CLIENT_ID)
    # and screen.py's own default (97) -- avoids the exact collision hit
    # 2026-08-24 testing (Error 326, client id already in use) if a manual
    # /screen run happens to be active at the same moment.
    cmd = [PYTHON, SCREENER_PATH, "--json", "--client-id", str(CLIENT_ID + 1),
           "--min-cushion", str(MIN_CUSHION_PCT)]
    if tickers:
        cmd += ["--tickers", tickers]
    if width_pct is not None:
        cmd += ["--width-pct", str(width_pct)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"screen.py exited {proc.returncode}: {proc.stderr[-2000:]}")
    data = json.loads(proc.stdout)
    return data.get("results", [])


def journal_insert_closed(pos: dict, exit_price, pnl, exit_reason: str):
    """Write a real trade_journal row for a resolved Safe Income position --
    this strategy had NO journal writes at all before 2026-09-11 (found live:
    a real position, NKE 33/32P entered 2026-09-03, had zero record anywhere
    of ever closing -- see oversight_log for the incident). is_paper=0 --
    this is real money, same convention as every other live strategy."""
    import sqlite3
    try:
        con = sqlite3.connect(BACKEND_DIR / "trade_journal.db")
        con.execute("""INSERT INTO trade_journal
            (opened_at, closed_at, ticker, expiry, strike, right, action, qty,
             entry_price, exit_price, exit_reason, pnl, win, strategy_type, is_paper, notes)
            VALUES (?, ?, ?, ?, ?, ?, 'CREDIT_SPREAD', 1, ?, ?, ?, ?, ?, 'SAFE_INCOME', 0, ?)""",
            (pos.get("entered_at"), datetime.now(timezone.utc).isoformat(), pos["ticker"],
             pos["expiry"], pos["short_k"], pos["right"], pos.get("entry_credit"), exit_price,
             exit_reason, pnl, 1 if (pnl or 0) > 0 else 0,
             f"short {pos['short_k']}/long {pos['long_k']} {pos['right']}, hold-to-expiry"))
        con.commit()
        con.close()
    except Exception as e:
        print(f"  journal_insert_closed failed: {e}")


def check_and_close_resolved_alpaca_positions(cfg: dict, state: dict) -> bool:
    """Real gap fixed 2026-09-11: this script is entry-only by design (see
    module docstring) and had ZERO mechanism to ever detect or record a
    position closing, on either broker. Found live: NKE 33/32P (entered
    2026-09-03, pre-dates the 2026-09-07 IBKR migration, still lives at
    Alpaca) had the Independent Traders tab correctly showing "1 open" --
    it's genuinely still open -- but there was no code path that would EVER
    have noticed or recorded it closing when it eventually does.

    This covers the legacy pre-migration Alpaca legs specifically (this
    strategy places everything through IBKR since 2026-09-07 -- a NEW
    IBKR-placed position is instead covered by the general reconciliation
    engine in main.py, the same mechanism already covering Manual Trader,
    chartexpert, and harami_daily). Returns True if state changed."""
    changed = False
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        client = TradingClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
                                paper=False, url_override=cfg.get("alpaca_base_url"))
        held = {p.symbol for p in client.get_all_positions()}
    except Exception as e:
        print(f"  check_and_close_resolved_alpaca_positions: Alpaca fetch failed, skipping: {e}")
        return False

    for key, pos in list(state["positions"].items()):
        short_sym = occ_symbol(pos["ticker"], pos["expiry"], pos["right"], pos["short_k"])
        long_sym = occ_symbol(pos["ticker"], pos["expiry"], pos["right"], pos["long_k"])
        if short_sym in held or long_sym in held:
            continue  # still genuinely open at Alpaca

        # Both legs gone -- try to find real closing fills (opposite side from
        # entry, filled after entry) to compute a real P&L; fall back to a
        # null-pnl record with a clear note rather than fabricating a number
        # (a true worthless-expiration doesn't generate a Alpaca "order" fill
        # at all, so "no closing order found" is the expected, common case,
        # not a sign something went wrong).
        exit_price, pnl, note = None, None, "no closing order found (consistent with worthless expiration/assignment -- real P&L not independently verified)"
        try:
            entry_dt = datetime.fromisoformat(pos["entered_at"])
            short_req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[short_sym], limit=10)
            long_req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[long_sym], limit=10)
            short_closes = [o for o in client.get_orders(short_req)
                            if o.filled_at and o.side.value.lower() == "buy" and o.filled_at > entry_dt]
            long_closes = [o for o in client.get_orders(long_req)
                           if o.filled_at and o.side.value.lower() == "sell" and o.filled_at > entry_dt]
            if short_closes and long_closes:
                short_buyback = float(short_closes[-1].filled_avg_price)
                long_sale = float(long_closes[-1].filled_avg_price)
                closing_debit = round(short_buyback - long_sale, 4)
                pnl = round((pos["entry_credit"] - closing_debit) * 100 * QTY, 2)
                exit_price = closing_debit
                note = f"real closing fills found: bought back short @ {short_buyback}, sold long @ {long_sale}"
        except Exception as e:
            note = f"closing-fill lookup failed ({e}); real P&L not independently verified"

        journal_insert_closed(pos, exit_price, pnl, "resolved_legacy_alpaca_position")
        log_oversight(
            f"safe_income_auto {pos['ticker']}: legacy Alpaca position resolved (legs no longer held). {note}",
            f"trade_journal row written (pnl={'$'+format(pnl,'.2f') if pnl is not None else 'null'}), removed from open state.",
            "check_and_close_resolved_alpaca_positions -- fixing the zero-exit-tracking gap found 2026-09-11.",
            pnl_impact=pnl,
        )
        telegram(cfg, f"Safe Income: {pos['ticker']} legacy Alpaca position resolved. {note}"
                      + (f" P&L: ${pnl:+.2f}" if pnl is not None else " P&L not independently verified -- please confirm in Alpaca."))
        del state["positions"][key]
        changed = True
    return changed


def existing_position_tickers(ib) -> set[str]:
    """Tickers with ANY currently-open IBKR option position -- this
    strategy's own or any other's. Avoids double-entry / unmonitored
    cross-strategy concentration, same concern the CRO sweep checks for.
    Migrated from Alpaca 2026-09-07 -- simpler than the old OCC-symbol
    parsing since ib_insync's own Option contract already exposes the
    underlying ticker directly as .symbol."""
    tickers = set()
    try:
        for p in ib.positions():
            if p.position != 0 and p.contract.secType == "OPT":
                tickers.add(p.contract.symbol)
    except Exception as e:
        print(f"Could not fetch existing positions ({e}) -- treating as unknown, will not skip on this basis")
    return tickers


def pretrade_review(cand: dict, budget: dict) -> dict:
    """Real-time CRO/CFO pre-trade review for one candidate -- BLOCKING
    gate before any order is placed. Mirrors _evc_pretrade_review's
    structure (main.py) adapted for a simple defined-risk vertical spread
    (no earnings-hit-rate check needed here -- this isn't earnings-driven;
    the screener's own cushion/delta gates already did that validation
    work via the real cushion backtest)."""
    findings = []
    approved = True

    max_risk = cand["max_risk"]
    credit = cand["cons_credit"]
    spot = cand["spot"]
    short_k = cand["short_k"]

    # CRO: high-volatility structural-fit check -- defense in depth. screen.py
    # only WARNS on this (stderr, informational, does not gate) because a
    # human using it interactively can apply judgment; an unattended
    # automated trader can't, so this is a real, hard gate here. Confirmed
    # live 2026-08-24: TSLA (56% vol) and AMD (76% vol) both cleared every
    # other real safety gate despite trade-history-playbook explicitly
    # flagging high-vol names as a structural mismatch for this methodology
    # (calibrated on calm large-caps, not TSLA/MSTR/COIN/PLTR-style names).
    vol_pct = cand.get("realized_vol_pct")
    if vol_pct is not None and vol_pct >= HIGH_VOL_THRESHOLD:
        findings.append(f"CRO: {vol_pct:.0f}% realized vol >= {HIGH_VOL_THRESHOLD:.0f}% threshold -- "
                         f"structurally a poor fit for this methodology regardless of cushion/prob, REJECTED")
        approved = False

    # CFO: this trade's size vs the 5% per-trade cap
    pct_of_acct = (max_risk / budget["combined_net_liq"]) if budget["combined_net_liq"] else 1.0
    findings.append(
        f"CFO: max risk ${max_risk:,.0f} = {pct_of_acct:.1%} of combined net liq ${budget['combined_net_liq']:,.0f}"
    )
    if max_risk > budget["per_strategy_cap"]:
        findings.append(f"EXCEEDS per-trade cap (${budget['per_strategy_cap']:,.0f} = 5% of net liq)")
        approved = False

    # CFO: real remaining headroom against the 50% total-portfolio budget
    findings.append(f"CFO: portfolio headroom ${budget['headroom']:,.0f} of ${budget['total_budget']:,.0f} total budget")
    if max_risk > budget["headroom"]:
        findings.append("EXCEEDS remaining portfolio headroom")
        approved = False

    # CRO: what-if scenarios at real cushion-relative moves
    move_pct = abs(spot - short_k) / spot  # the actual cushion this candidate has
    scenarios = []
    for mult in (0.0, 0.5, 1.0, 1.5, 2.0):
        px = round(spot * (1 - move_pct * mult), 2) if cand["right"] == "P" else round(spot * (1 + move_pct * mult), 2)
        breached = (px < short_k) if cand["right"] == "P" else (px > short_k)
        pnl = round(credit * 100, 2) if not breached else round((credit - abs(px - short_k)) * 100, 2)
        pnl = max(pnl, -max_risk)
        scenarios.append({"move": f"{mult:.1f}x cushion", "price": px, "pnl": pnl})
    findings.append(f"Prob(max profit) from live delta: {cand.get('prob_max_profit')}% | cushion: {cand['cushion_pct']:.1f}%")

    return {"approved": approved, "findings": findings, "scenarios": scenarios}


def is_size_only_rejection(review: dict) -> bool:
    """True if EVERY failing reason is a trade-SIZE issue (per-trade cap or
    portfolio headroom) -- both fixable by a narrower spread. False if the
    CRO high-vol veto fired, which narrowing width can't fix."""
    if review["approved"]:
        return False
    return not any(f.startswith("CRO:") for f in review["findings"])


def try_narrower_width(ticker: str, right: str, budget: dict) -> dict | None:
    """Re-scans ONE ticker at progressively narrower spread widths (see
    NARROW_WIDTH_PCTS), looking for one whose real max_risk clears the CFO
    per-trade cap. Real strike-increment rounding on a given ticker/price
    can mean a narrower width_pct doesn't actually land on a narrower real
    strike -- each attempt is checked against the ACTUAL returned max_risk,
    never assumed to work. Returns the first candidate that clears the cap,
    or None if none of the attempts do."""
    for w in NARROW_WIDTH_PCTS:
        try:
            cands = get_screener_candidates(tickers=ticker, width_pct=w)
        except Exception as e:
            print(f"    narrow-width retry (width_pct={w}) for {ticker} failed: {e}")
            continue
        match = next((c for c in cands if c["right"] == right), None)
        if match is None:
            print(f"    width_pct={w}: no passing {right} candidate for {ticker} at this width")
            continue
        print(f"    width_pct={w}: max_risk=${match['max_risk']:.0f} "
              f"(cap ${budget['per_strategy_cap']:,.0f}) cushion={match['cushion_pct']:.1f}% "
              f"credit=${match['cons_credit']:.2f}")
        if match["max_risk"] <= budget["per_strategy_cap"]:
            return match
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Run the full pipeline but never submit a real order")
    ap.add_argument("--max-new", type=int, default=1,
                     help="Cap on NEW positions this run can open, in addition to the dynamic capital-based limit "
                          "(default 1 -- small live-test sizing until a clean run is confirmed)")
    ap.add_argument("--tickers", help="comma-separated subset passthrough to screen.py -- overrides the default "
                                       "curated 25-ticker UNIVERSE entirely (for manual testing)")
    ap.add_argument("--timeout", type=int, default=3000,
                     help="screener subprocess timeout in seconds (default 3000 = 50min -- real margin over the "
                          "~40min expected time for 25 tickers at the measured ~1.6min/ticker rate)")
    ap.add_argument("--ignore-market-hours", action="store_true",
                     help="Bypass the market-hours check (for manual/dry-run testing outside 9:30-16:00 ET)")
    args = ap.parse_args()

    if not args.ignore_market_hours and not market_is_open():
        print("Market is closed (weekday 9:30-16:00 ET only) -- skipping this run.")
        return

    cfg = load_config()
    state = load_state()

    # Runs before the market-hours/pause/candidate gates below so a resolved
    # position gets recorded even on a run that otherwise does nothing --
    # see check_and_close_resolved_alpaca_positions's docstring for why this
    # exists (a real position had zero exit-tracking anywhere).
    if check_and_close_resolved_alpaca_positions(cfg, state):
        save_state(state)

    if PAUSE_FLAG.exists():
        print(f"PAUSED -- {PAUSE_FLAG} exists. Refusing to run until a human clears it.")
        return

    tickers_arg = args.tickers or ",".join(UNIVERSE)
    print(f"Scanning curated universe ({len(UNIVERSE)} tickers, liquidity-ranked): {tickers_arg}")

    print("Fetching real screener candidates (safe-income-screener --json)...")
    try:
        candidates = get_screener_candidates(tickers=tickers_arg, timeout_s=args.timeout)
    except Exception as e:
        telegram(cfg, f"⚠️ Safe Income Trader: failed to run screener: {e}", high_priority=True)
        log_oversight(f"Safe Income Trader: screener run failed: {e}", "Aborted this run, no orders considered.",
                      "Unattended entry scan for the validated deep-OTM credit-spread strategy.")
        return
    print(f"  {len(candidates)} candidates passed all screener safety gates.")

    if not candidates:
        log_oversight("Safe Income Trader: 0 candidates passed screener gates this run.",
                      "No orders considered.", "Unattended entry scan.")
        return

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)

    entered = []
    try:
        skip_tickers = existing_position_tickers(ib)
        for cand in candidates:  # already ranked by screen.py (prob_max_profit, then ROI)
            if len(entered) >= args.max_new:
                print(f"Hit --max-new={args.max_new} for this run, stopping.")
                break

            ticker = cand["ticker"]
            if ticker in skip_tickers:
                print(f"  SKIP {ticker}: already has an open position (this strategy or another)")
                continue

            # cap_pct raised 5%->20% (CEO decision 2026-09-08, same
            # mechanism/precedent as GOOG Condor's existing 15% override)
            budget = cro_cfo_capital_budget(ib, cap_pct=0.20)
            if budget["headroom"] <= 0:
                print(f"  STOP: no portfolio headroom left (${budget['headroom']:,.0f})")
                break

            review = pretrade_review(cand, budget)
            print(f"\n--- {ticker} {cand['right']} {cand['short_k']:.0f}/{cand['long_k']:.0f} ---")
            for f in review["findings"]:
                print(f"  {f}")

            if not review["approved"] and is_size_only_rejection(review):
                print(f"  Rejected on size only -- retrying {ticker} with a narrower spread...")
                narrowed = try_narrower_width(ticker, cand["right"], budget)
                if narrowed is not None:
                    cand = narrowed
                    review = pretrade_review(cand, budget)
                    print(f"--- {ticker} {cand['right']} {cand['short_k']:.0f}/{cand['long_k']:.0f} (narrowed) ---")
                    for f in review["findings"]:
                        print(f"  {f}")

            if not review["approved"]:
                log_oversight(
                    f"safe_income_auto {ticker}: REJECTED — {'; '.join(review['findings'])}",
                    "No order placed.", "Real-time CRO/CFO pre-trade review.",
                )
                continue

            expiry = cand["expiry"]
            right = cand["right"]
            short_sym = occ_symbol(ticker, expiry, right, cand["short_k"])
            long_sym = occ_symbol(ticker, expiry, right, cand["long_k"])

            if args.dry_run:
                print(f"  [DRY RUN] would BUY {long_sym} then SELL {short_sym}, qty={QTY}")
                entered.append({"ticker": ticker, "dry_run": True})
                continue

            vals = {v.tag: v.value for v in ib.accountValues()}
            avail = safe_px(vals.get("AvailableFunds")) or 0.0
            if avail < cand["max_risk"]:
                print(f"  ABORT {ticker}: available funds ${avail:,.0f} < needed ${cand['max_risk']:,.0f}")
                log_oversight(f"safe_income_auto {ticker}: insufficient available funds (${avail:,.0f} < ${cand['max_risk']:,.0f})",
                              "No order placed, aborted before any leg.", "Real-time collateral check.")
                continue

            # Real individual-leg quotes via IBKR -- cons_credit/max_risk from the
            # screener are the NET spread economics, not a per-leg price. Never
            # derive an order's limit price from the net figure.
            long_contract = Option(ticker, expiry, cand["long_k"], right, "SMART", "100", "USD")
            short_contract = Option(ticker, expiry, cand["short_k"], right, "SMART", "100", "USD")
            try:
                long_q = get_quote(ib, long_contract)
                short_q = get_quote(ib, short_contract)
            except Exception as e:
                print(f"  ABORT {ticker}: could not get real leg quotes ({e})")
                log_oversight(f"safe_income_auto {ticker}: could not qualify/quote real legs ({e})",
                              "No order placed.", "Pre-execution quote fetch.")
                continue
            if not (long_q["bid"] and long_q["ask"] and short_q["bid"] and short_q["ask"]):
                print(f"  ABORT {ticker}: no live two-sided market on one or both legs")
                log_oversight(f"safe_income_auto {ticker}: no live bid/ask on one or both legs",
                              "No order placed.", "Pre-execution quote fetch.")
                continue

            # Long leg first -- proven sequential pattern, zero naked exposure at any
            # point. Same repricing-ladder execution validated 2026-08-20 (DE close),
            # now on IBKR (migrated from Alpaca 2026-09-07) -- reuses the already-
            # qualified long_contract/short_contract Option objects from the quote
            # fetch above, so no OCC-symbol-to-contract lookup is needed for routing;
            # occ_symbol() (long_sym/short_sym) is now purely a registry/log label.
            long_ok, long_px = ibkr_place_leg_with_ladder(ib, long_contract, "BUY", f"{ticker} long",
                                                            QTY, long_q["bid"], long_q["ask"], long_q["mid"])
            if not long_ok:
                log_oversight(f"safe_income_auto {ticker}: long leg failed to fill, nothing committed.",
                              "Aborted before short leg -- clean, no exposure.", "Sequential entry.")
                continue

            short_ok, short_px = ibkr_place_leg_with_ladder(ib, short_contract, "SELL", f"{ticker} short",
                                                              QTY, short_q["bid"], short_q["ask"], short_q["mid"])
            if not short_ok:
                PAUSE_FLAG.write_text(
                    f"Long leg filled ({long_sym} @ {long_px}) but short leg did not. "
                    f"Naked long position — needs human review before this script runs again.\n"
                )
                telegram(cfg, f"🚨 safe_income_auto {ticker}: long leg filled but short leg FAILED — "
                              f"naked position, PAUSED until cleared", high_priority=True)
                log_oversight(f"safe_income_auto {ticker}: PARTIAL FILL — long filled, short failed. PAUSED.",
                              "Wrote pause flag, HIGH-priority Telegram sent. Needs human review.",
                              "Sequential entry, short leg failed after long leg committed.")
                break

            real_credit = round(short_px - long_px, 2)
            entered_at = datetime.now(timezone.utc).isoformat()
            entered.append({"ticker": ticker, "right": right, "short_k": cand["short_k"], "long_k": cand["long_k"],
                             "expiry": expiry, "credit": real_credit, "max_risk": cand["max_risk"]})
            state["positions"][f"{ticker}_{expiry}_{right}"] = {
                "ticker": ticker, "right": right, "short_k": cand["short_k"], "long_k": cand["long_k"],
                "expiry": expiry, "entry_credit": real_credit, "entered_at": entered_at,
                "exit_rule": "hold_to_expiry", "venue": "ibkr",
            }
            save_state(state)
            # Real gap fixed 2026-09-11: entries never wrote a trade_journal
            # row at all -- see check_and_close_resolved_alpaca_positions's
            # docstring. This IBKR-placed leg is now also picked up by the
            # general reconciliation engine (main.py _recon_iter_all_open /
            # _recon_apply_close), which UPDATEs this same open row via
            # _recon_journal_close when the legs disappear from IBKR.
            try:
                import sqlite3
                con = sqlite3.connect(BACKEND_DIR / "trade_journal.db")
                con.execute("""INSERT INTO trade_journal
                    (opened_at, ticker, expiry, strike, right, action, qty,
                     entry_price, strategy_type, is_paper, notes)
                    VALUES (?, ?, ?, ?, ?, 'CREDIT_SPREAD', 1, ?, 'SAFE_INCOME', 0, ?)""",
                    (entered_at, ticker, expiry, cand["short_k"], right, real_credit,
                     f"short {cand['short_k']}/long {cand['long_k']} {right}, hold-to-expiry, max risk ${cand['max_risk']:.0f}"))
                con.commit()
                con.close()
            except Exception as e:
                print(f"  journal open-row insert failed: {e}")
            log_oversight(
                f"safe_income_auto {ticker}: ENTERED {right} {cand['short_k']:.0f}/{cand['long_k']:.0f}, "
                f"real credit ${real_credit:.2f}, max risk ${cand['max_risk']:.0f}",
                f"Filled: long ${long_px}, short ${short_px}. Exit rule: hold to expiry ({expiry}).",
                "Automated Tier 1 (validated deep-OTM credit spread) entry.",
                pnl_impact=None,
            )
    finally:
        ib.disconnect()

    state["runs"].append({"time": datetime.now(timezone.utc).astimezone().isoformat(),
                           "candidates_seen": len(candidates), "entered": len(entered)})
    save_state(state)

    if entered:
        lines = "\n".join(f"  {e['ticker']} {e.get('right','')} {e.get('short_k','')}/{e.get('long_k','')} "
                           f"credit ${e.get('credit','')}" for e in entered)
        telegram(cfg, f"✅ Safe Income Trader: {len(entered)} new position(s) this run:\n{lines}")
    print(f"\nDone. {len(entered)} new position(s) this run (of {len(candidates)} candidates seen).")


if __name__ == "__main__":
    main()
