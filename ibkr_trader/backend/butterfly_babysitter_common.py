"""
Shared logic for the per-ticker 0DTE butterfly babysitters (qqq_butterfly_
babysitter.py, iwm_butterfly_babysitter.py, ...). Factored out 2026-09-04
when the IWM babysitter was built as a near-copy of QQQ's and it became
clear the ticker-specific assumptions in the QQQ-only version were real
correctness bugs waiting to happen for any other ticker, not just
boilerplate to duplicate:

  1. GEX band scan: the QQQ-only version generated a synthetic strike list
     from a hardcoded $1 increment. IWM's real listed strikes near the
     money are NOT uniformly spaced (confirmed live 2026-09-04: 291, 292,
     292.5, 293, 294, ... -- $0.50 strikes mixed into a mostly-$1 grid).
     This version reads IBKR's own real chain.strikes and takes the N
     strikes nearest spot on each side, so it never guesses an increment
     and never misses a real half-strike.
  2. Option chain tradingClass: BOTH QQQ and IWM carry a second, sparse
     "2<TICKER>" tradingClass on SMART with a single far-dated expiry
     (confirmed live for both 2026-09-04) -- a bare exchange=="SMART"
     filter can silently resolve "today's expiry" to weeks out. Must
     filter on tradingClass == ticker, matching how every real order leg
     in this codebase already qualifies these contracts.
  3. Butterfly max-profit math: assumed symmetric wings (wing_hi-body ==
     body-wing_lo). Real, live: today's IWM butterfly is 291/294/297.5 --
     a 3.0/3.5 split, not symmetric, because the wing_step config lands on
     whatever strike IBKR actually lists. The true max-profit-at-expiry
     for a long call butterfly (verified via the payoff function
     max(S-K1,0) - 2*max(S-K2,0) + max(S-K3,0), evaluated at S=K2) is
     (body - wing_lo) - entry_debit -- depends only on the LOWER wing
     width, not an assumed shared "wing_width". Using the wrong width
     silently overstates max profit whenever wing_hi is the wider side.

Each per-ticker script is a thin wrapper: set TICKER/CLIENT_ID/
INSIGHT_CLIENT_ID and call run_babysitter().
"""
import json
from datetime import date, datetime, timedelta, timezone

import requests

from unusual_whales_client import UnusualWhalesClient
from ib_insync import IB, Option, Stock
from alpaca_0dte_common import load_config
from ibkr_0dte_common import (
    ibkr_place_leg_with_ladder, ibkr_has_open_position, parse_occ_symbol, occ_contract,
)

ET_OFFSET = timedelta(hours=-4)
ET = timezone(ET_OFFSET)
REGISTRY_FILE = "alpaca_0dte_positions.json"
VERIFY_START = (15, 47)  # matches the trader's own fallback-close time -- verify right after

INSIGHT_GEX_BAND = 20     # real listed strikes each side of spot, same band used
                           # for the manual live QQQ GEX validation 2026-09-04
INSIGHT_INTERVAL_SEC = 30 * 60  # min gap between full insight scans -- a real
                                 # GEX band scan is ~2min of IBKR calls, not cheap
                                 # enough to run on every 10min babysitter cycle
DARKPOOL_MIN_PREMIUM = 1_000_000.0  # same server-side floor darkpool_activity_monitor.py uses
DARKPOOL_WINDOW_MIN = 30  # "recent" window for burst detection


def now_et() -> datetime:
    return datetime.now(ET)


def safe_px(v) -> float:
    try:
        f = float(v)
        return f if (f > 0 and f == f and f not in (float("inf"), float("-inf"))) else 0.0
    except (TypeError, ValueError):
        return 0.0


def telegram(cfg, text, high_priority=False):
    try:
        prefix = "\U0001F6A8 " if high_priority else "\U0001F4E1 "
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": prefix + text},
            timeout=10,
        )
    except Exception as exc:
        print(f"Telegram send failed: {exc}")


def oversight_log(ticker, summary, rationale="", outcome=""):
    entry = {
        "time": datetime.now(timezone.utc).isoformat(), "actor": "trader",
        "category": f"{ticker.lower()}_butterfly_babysitter", "summary": summary,
        "rationale": rationale, "outcome": outcome, "pnl_impact": None,
    }
    with open("oversight_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_state(state_file):
    try:
        with open(state_file) as f:
            s = json.load(f)
        if s.get("date") == date.today().isoformat():
            return s
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {"date": date.today().isoformat(), "verified": False}


def save_state(state_file, s):
    with open(state_file, "w") as f:
        json.dump(s, f)


def find_todays_butterfly(ticker):
    with open(REGISTRY_FILE) as f:
        reg = json.load(f)
    today = date.today().isoformat()
    for pos_id, p in reg.get("positions", {}).items():
        if p.get("ticker") == ticker and p.get("strategy") == "long_butterfly_0dte" and \
           p.get("entry_time", "").startswith(today):
            return p, "positions", reg
    for p in reg.get("closed", []):
        if p.get("ticker") == ticker and p.get("strategy") == "long_butterfly_0dte" and \
           p.get("entry_time", "").startswith(today):
            return p, "closed", reg
    return None, None, reg


def _todays_expiry_chain(ib, ticker):
    """Real chain for TICKER's own tradingClass on SMART (not the sparse
    2<TICKER> class -- see module docstring), and today's actual 0DTE
    expiry from its real expirations list. Returns (chain, expiry) or
    (None, None).
    """
    stk = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(stk)
    chains = ib.reqSecDefOptParams(stk.symbol, "", stk.secType, stk.conId)
    chain = next((c for c in chains if c.exchange == "SMART" and c.tradingClass == ticker), None)
    if not chain:
        return None, None
    today = date.today()
    todays_expiry = min(
        (e for e in chain.expirations if datetime.strptime(e, "%Y%m%d").date() >= today),
        default=None,
    )
    return chain, todays_expiry


def compute_live_gex(ib, ticker, spot, band=INSIGHT_GEX_BAND):
    """Real, live GEX for TICKER's TODAY's actual 0DTE expiry.

    NOT a call into the gex-vex-calculator skill -- that skill's
    calc_gex_vex.py is built for a 25-45 DTE monthly-style window and
    explicitly SKIPs QQQ/IWM-style daily-expiry names. Same real
    methodology validated manually: modelGreeks.gamma + real open interest
    via generic tick "101", dollar-gamma convention
    gamma * OI * 100 * spot^2 * 0.01. Strikes come from IBKR's own real
    chain.strikes (nearest `band` on each side of spot) rather than an
    assumed increment, since real strike spacing is not always uniform
    near the money (confirmed for IWM). Returns None if no usable
    expiry/strikes.
    """
    chain, todays_expiry = _todays_expiry_chain(ib, ticker)
    if not chain or not todays_expiry:
        return None

    strikes_all = sorted(chain.strikes)
    if not strikes_all:
        return None
    atm_idx = min(range(len(strikes_all)), key=lambda i: abs(strikes_all[i] - spot))
    lo = max(0, atm_idx - band)
    hi = min(len(strikes_all), atm_idx + band + 1)
    strikes = strikes_all[lo:hi]

    # Real latency fix 2026-09-07: batch-qualify + batch-request market data
    # for every (strike, right) contract up front, sleep ONCE, then read
    # everything -- previously did qualify+reqMktData+sleep(1.2)+cancel per
    # contract SEQUENTIALLY (measured live at band=6/26 contracts: 36.4s
    # real). IBKR streams data for every contract it's been asked to watch
    # concurrently regardless of how many separate reqMktData calls were
    # made -- the only serial part was our own per-contract sleep.
    # Settle time scales with contract count (greeks/OI settle slower than
    # plain bid/ask, and this can be a large batch -- INSIGHT_GEX_BAND=20
    # is 82 contracts) but is capped, not linear with the old per-contract
    # 1.2s: min 3s (matches get_quotes_batch's proven-sufficient window for
    # a small batch), scaling up for a big one, capped at 10s. NOT yet
    # live-validated for real OI/greeks data quality at this new timing
    # (market closed at build time -- structural/mechanical test only
    # showed 0 contracts scanned either way, old code included, since no
    # live data was flowing) -- watch the first real live run's
    # contracts_scanned count against what band implies (up to 2*band+1
    # strikes x 2 rights) to confirm this settle window is actually enough.
    candidates = [(k, right, Option(ticker, todays_expiry, k, right, "SMART", tradingClass=ticker))
                  for k in strikes for right in ("C", "P")]
    ib.qualifyContracts(*(c for _, _, c in candidates))
    tickers = [(k, right, c, ib.reqMktData(c, "101", False, False))
               for k, right, c in candidates if c.conId]
    settle_s = min(max(3.0, 0.15 * len(tickers)), 10.0)
    ib.sleep(settle_s)

    net_gex = 0.0
    wall_strike, wall_gex = None, 0.0
    n_scanned = 0
    for k, right, c, td in tickers:
        oi = safe_px(td.callOpenInterest if right == "C" else td.putOpenInterest)
        mg = td.modelGreeks
        if oi <= 0 or mg is None or mg.gamma is None:
            continue
        dollar_gamma = mg.gamma * oi * 100 * (spot ** 2) * 0.01
        signed = dollar_gamma if right == "C" else -dollar_gamma
        net_gex += signed
        n_scanned += 1
        if abs(signed) > abs(wall_gex):
            wall_gex, wall_strike = signed, k
    for _, _, c, _ in tickers:
        ib.cancelMktData(c)

    return {
        "expiry": todays_expiry, "net_gex": round(net_gex, 0),
        "regime": "positive_gamma" if net_gex > 0 else "negative_gamma",
        "wall_strike": wall_strike, "wall_gex": round(wall_gex, 0),
        "contracts_scanned": n_scanned,
    }


def check_dark_pool_activity(ticker):
    """Real, current-day dark pool prints via Unusual Whales (same proven
    client darkpool_activity_monitor.py already uses). Flags a 'burst' if
    the last DARKPOOL_WINDOW_MIN minutes' real premium exceeds 2x today's
    average per-window rate so far -- a simple, self-referential threshold
    (vs. a precomputed 20-day ADV) since this is a single-ticker intraday
    check, not the universe-wide screener.
    """
    try:
        uw = UnusualWhalesClient()
        trades = uw.ticker_trades(ticker, limit=500, min_premium=DARKPOOL_MIN_PREMIUM)
    except Exception as exc:
        return {"error": str(exc)}

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(minutes=DARKPOOL_WINDOW_MIN)
    total_prem, total_n = 0.0, 0
    recent_prem, recent_n = 0.0, 0
    top_prints = []
    for t in trades:
        prem = float(t.get("premium", 0) or 0)
        total_prem += prem
        total_n += 1
        ts = t.get("executed_at")
        t_dt = None
        if ts:
            try:
                t_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass
        if t_dt and t_dt >= cutoff:
            recent_prem += prem
            recent_n += 1
        if prem >= 3_000_000:
            top_prints.append({"premium": prem, "size": t.get("size"), "price": t.get("price"), "executed_at": ts})

    et_now = now_utc.astimezone(ET)
    session_start = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_elapsed = max((et_now - session_start).total_seconds() / 60.0, DARKPOOL_WINDOW_MIN)
    expected_per_window = total_prem / minutes_elapsed * DARKPOOL_WINDOW_MIN
    elevated = recent_prem > 2 * expected_per_window and recent_prem > DARKPOOL_MIN_PREMIUM

    return {
        "recent_window_premium": round(recent_prem, 0), "recent_window_count": recent_n,
        "today_total_premium": round(total_prem, 0), "today_total_count": total_n,
        "elevated": elevated, "top_prints": sorted(top_prints, key=lambda x: -x["premium"])[:3],
    }


def intraday_insight_check(cfg, ticker, pos, legs, ib, spot):
    """Synthesize real GEX + dark-pool + momentum signals into ONE
    Telegram insight. Never places or closes an order -- this is
    deliberately alert-only: "cut a loss early based on deteriorating
    GEX/momentum/dark-pool signals" is a discretionary trading judgment
    call that changes the P&L outcome either way, distinct from the
    separate emergency-close path (for a pure MECHANICAL failure -- a leg
    that didn't close when the script's own logic says it should have),
    which is safe to auto-remediate. This function surfaces a
    recommendation for the CEO to act on; it never acts itself.
    """
    body = legs["body"]["strike"]
    wing_lo = legs["wing_lo"]["strike"]
    wing_hi = legs["wing_hi"]["strike"]
    wing_width_lo = body - wing_lo
    wing_width_hi = wing_hi - body

    mids = {}
    for name, l in legs.items():
        _, expiry, strike, right = parse_occ_symbol(l["symbol"])
        c = occ_contract(ticker, expiry, strike, right)
        ib.qualifyContracts(c)
        tk = ib.reqTickers(c)[0]
        ib.sleep(1)
        bid, ask = tk.bid, tk.ask
        mids[name] = (bid + ask) / 2 if bid and ask and bid > 0 and ask > 0 else safe_px(tk.last)

    entry_debit = -pos["net_entry_credit"]
    qty = pos.get("qty", 1)
    max_loss = pos.get("max_risk", entry_debit * 100 * qty)
    # True max profit at expiry-at-body for a long call butterfly, derived from
    # the payoff function max(S-K1,0) - 2*max(S-K2,0) + max(S-K3,0) evaluated at
    # S=K2=body: payoff = (body - wing_lo), independent of wing_hi. Must use
    # wing_width_lo specifically -- NOT an assumed shared "wing width" -- since
    # real wings here are not always symmetric (see module docstring).
    max_profit_theoretical = qty * (wing_width_lo - entry_debit) * 100

    live_pnl = None
    if all(v > 0 for v in mids.values()):
        close_value = mids["wing_lo"] + mids["wing_hi"] - 2 * mids["body"]
        live_pnl = qty * (close_value - entry_debit) * 100

    pct_to_wing_hi = (spot - body) / wing_width_hi if wing_width_hi else 0
    pct_to_wing_lo = (body - spot) / wing_width_lo if wing_width_lo else 0

    gex = compute_live_gex(ib, ticker, spot)
    darkpool = check_dark_pool_activity(ticker)

    lines = [
        f"{ticker} butterfly insight check ({now_et().strftime('%H:%M:%S')} ET):",
        f"  Spot ${spot:.2f} | body ${body:.2f} | wings ${wing_lo:.2f}/${wing_hi:.2f} | "
        f"{pct_to_wing_hi*100:+.0f}% toward hi wing / {pct_to_wing_lo*100:+.0f}% toward lo wing",
    ]
    if live_pnl is not None:
        lines.append(f"  Live P&L: ${live_pnl:+.0f} (max loss ~${max_loss:.0f}, "
                      f"theoretical max profit ~${max_profit_theoretical:.0f})")
    else:
        lines.append("  Live P&L: unavailable this check (bad/missing leg quote)")

    if gex:
        lines.append(f"  GEX: net={gex['net_gex']:+,.0f} ({gex['regime']}), dominant wall ${gex['wall_strike']} "
                      f"(gex {gex['wall_gex']:+,.0f}), expiry {gex['expiry']}")
    else:
        lines.append("  GEX: unavailable this check")

    if darkpool.get("error"):
        lines.append(f"  Dark pool: unavailable ({darkpool['error']})")
    else:
        flag = "ELEVATED" if darkpool["elevated"] else "normal"
        lines.append(f"  Dark pool ({flag}): last {DARKPOOL_WINDOW_MIN}min ${darkpool['recent_window_premium']:,.0f} "
                      f"across {darkpool['recent_window_count']} prints (today total ${darkpool['today_total_premium']:,.0f})")

    concerns = []
    if abs(pct_to_wing_hi) >= 0.7 or abs(pct_to_wing_lo) >= 0.7:
        concerns.append("price is closing in on a wing")
    if gex and gex["regime"] == "negative_gamma":
        concerns.append("negative gamma regime (moves can amplify)")
    if darkpool.get("elevated"):
        concerns.append("elevated real dark-pool premium in the last window")
    if live_pnl is not None and max_loss and live_pnl < -0.5 * max_loss:
        concerns.append("live loss already over 50% of max loss")

    if concerns:
        rec = ("Recommendation: worth considering closing early for a smaller loss/locking in gain -- " +
               "; ".join(concerns) + ". This is an insight, not an automated action -- no order has been placed.")
        high_pri = True
    else:
        rec = ("Recommendation: structure looks intact, no signals suggesting an early exit right now -- "
               "riding per standard plan (hold to 15:47 verification).")
        high_pri = False
    lines.append(f"  {rec}")

    msg = "\n".join(lines)
    print(msg)
    telegram(cfg, msg, high_priority=high_pri)
    oversight_log(ticker, msg, "Intraday GEX/dark-pool/momentum insight check for the daily butterfly.",
                  outcome="insight_concern" if concerns else "insight_clear")


def run_babysitter(ticker, tws_port, client_id, insight_client_id):
    """Full per-run logic for one ticker's daily 0DTE butterfly. See
    module docstring + each wrapper script for the per-run behavior
    description (find position -> pre-verify insight checks -> verify ->
    emergency close if needed).
    """
    state_file = f"{ticker.lower()}_butterfly_babysitter_state.json"
    cfg = load_config()
    now = now_et()

    if now.weekday() >= 5:
        return
    if now.hour < 9 or (now.hour, now.minute) < (9, 30):
        return

    pos, where, reg = find_todays_butterfly(ticker)
    if pos is None:
        print(f"[{now.strftime('%H:%M:%S')}] No {ticker} butterfly found for today ({date.today().isoformat()}) -- nothing to babysit.")
        return

    legs = {l["leg"]: l for l in pos["legs"]}
    syms = {name: l["symbol"] for name, l in legs.items()}

    state = load_state(state_file)
    if state.get("verified"):
        print(f"[{now.strftime('%H:%M:%S')}] Already verified today -- nothing more to do.")
        return

    verify_time_reached = (now.hour, now.minute) >= VERIFY_START

    def _leg_contract(sym):
        t, expiry_ibkr, strike, right = parse_occ_symbol(sym)
        c = occ_contract(t, expiry_ibkr, strike, right)
        return c

    if not verify_time_reached:
        ib = IB()
        try:
            ib.connect("127.0.0.1", tws_port, clientId=insight_client_id, timeout=20)
            leg_contracts = {name: _leg_contract(sym) for name, sym in syms.items()}
            ib.qualifyContracts(*leg_contracts.values())
            real_open = [name for name, c in leg_contracts.items() if ibkr_has_open_position(ib, c)]
            print(f"[{now.strftime('%H:%M:%S')}] Pre-verify check: registry phase={where}, "
                  f"real open legs on IBKR: {real_open or 'none'}")

            last_check = state.get("last_intraday_check")
            due = last_check is None or \
                (now - datetime.fromisoformat(last_check)).total_seconds() >= INSIGHT_INTERVAL_SEC
            if due and real_open:
                stk = Stock(ticker, "SMART", "USD")
                ib.qualifyContracts(stk)
                [tk] = ib.reqTickers(stk)
                ib.sleep(1)
                spot = safe_px(tk.last) or safe_px(tk.close)
                if spot > 0:
                    intraday_insight_check(cfg, ticker, pos, legs, ib, spot)
                    state["last_intraday_check"] = now.isoformat()
                    save_state(state_file, state)
                else:
                    print(f"[{now.strftime('%H:%M:%S')}] Intraday insight check skipped: no real spot price")
        except Exception as exc:
            print(f"Pre-verify/intraday insight check failed: {exc}")
            oversight_log(ticker, f"{ticker} butterfly pre-verify/intraday check failed: {exc}", outcome="error")
        finally:
            ib.disconnect()
        return

    # ── Verification window (>= 15:47 ET) ────────────────────────────────
    # Real safety-net change, IBKR migration 2026-09-07: the butterfly
    # trader itself now attempts a real close at its own HARD_CLOSE_TIME
    # (15:47), since IBKR (unlike Alpaca) never auto-closes 0DTE option
    # positions -- this babysitter's emergency close below is now a genuine
    # second-line backstop for that primary close, not a fallback for a
    # broker action that no longer exists.
    ib = IB()
    ib.connect("127.0.0.1", tws_port, clientId=client_id, timeout=20)
    try:
        leg_contracts = {name: _leg_contract(sym) for name, sym in syms.items()}
        ib.qualifyContracts(*leg_contracts.values())
        real_open = {name: c for name, c in leg_contracts.items() if ibkr_has_open_position(ib, c)}

        if not real_open:
            real_pnl = pos.get("close_pnl") if where == "closed" else None
            msg = (f"{ticker} butterfly ({pos['pos_id']}): verified all 3 legs genuinely flat on IBKR "
                   f"at {now.strftime('%H:%M:%S')} ET. " +
                   (f"Registry already shows close_pnl=${real_pnl}." if real_pnl is not None else
                    "Registry has no confirmed close_pnl yet -- needs manual reconciliation "
                    "against real IBKR order history, since automatic leg-by-leg reconstruction "
                    "isn't reliable enough to trust unattended."))
            print(msg)
            telegram(cfg, msg, high_priority=False)
            oversight_log(ticker, msg, "Post-close verification for the daily butterfly.",
                          outcome="verified_flat")
            state["verified"] = True
            save_state(state_file, state)
            return

        # ── Real, unclosed leg(s) found past verify time ──────────────────
        alert_lines = [f"{ticker} butterfly ({pos['pos_id']}) still has REAL OPEN leg(s) at "
                       f"{now.strftime('%H:%M:%S')} ET, past the expected 15:47 close -- "
                       f"attempting emergency close now."]
        for name, c in real_open.items():
            sym = syms[name]
            tk = ib.reqTickers(c)[0]
            ib.sleep(2)
            bid, ask = tk.bid, tk.ask
            mid = (bid + ask) / 2 if bid and ask and bid > 0 and ask > 0 else tk.last
            alert_lines.append(f"  {name} {sym}: bid={bid} ask={ask} mid={mid}")
            try:
                ok, fill_px = ibkr_place_leg_with_ladder(
                    ib, c, "SELL", f"{ticker.lower()}_babysitter {name}",
                    legs[name].get("qty", 1), bid, ask, mid)
                alert_lines.append(f"    emergency close: {'FILLED @ ' + str(fill_px) if ok else 'DID NOT FILL'}")
            except Exception as exc:
                alert_lines.append(f"    emergency close FAILED: {exc}")
    finally:
        ib.disconnect()

    msg = "\n".join(alert_lines)
    print(msg)
    telegram(cfg, msg, high_priority=True)
    oversight_log(ticker, msg, "Real leg(s) still open past expected close time -- emergency close attempted.",
                  outcome="emergency_close_attempted")
    # Deliberately do NOT mark verified=True here -- if the emergency close didn't
    # fully work, the next run (10min later) should try again and re-alert, not go quiet.
