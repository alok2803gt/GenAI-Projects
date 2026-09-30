"""
Standing reconciliation daemon: catches state-file/IBKR desyncs that a
strategy's own restart-recovery or forced-close logic can miss.

Real incident that prompted this (2026-09-18): Ashley's SPY 760C hard-close
at 15:55 ET failed 3x with margin-deficiency errors, because IBKR's own
risk desk had already liquidated the position ~30s earlier (execution
orderId=0 -- not one of Ashley's own orders). Ashley's code only updates
alpaca_0dte_positions.json on ITS OWN order fills, so the registry stayed
stuck at phase="open" indefinitely even though the real account had
already closed the position for a real, known $51 profit. This is a
structurally different failure mode than the harami_daily_trader.py bug
fixed the same day (stale ib.trades() cache across a fresh connection) --
here the state file's own writer never even attempted a matching write,
because the close happened entirely outside its control.

Scope: only auto-corrects the unambiguous direction -- a position the
registry says is OPEN but which IBKR's live portfolio shows as COMPLETELY
FLAT across every leg. Never invents an OPEN position (no entry-side
backfill here); never touches a position where IBKR shows ANY nonzero
size in ANY leg, even one that doesn't match expectations -- that's
correctly left alone for a human/the strategy's own code to resolve,
rather than guessed at.

Deliberately does NOT touch harami_trader_state.json -- that strategy's
own script (harami_daily_trader.py) already does portfolio-based
reconciliation on every scheduled run (twice daily), and having two
independent writers touch the same file on an overlapping schedule is a
collision risk with no offsetting benefit. Also does not touch
chartexpert_shadow_state.json (shadow/simulated only, no real IBKR
positions to reconcile against) or Manual Trader (in-process state inside
main.py, which already has a live `ib` handle of its own).

Usage: python ibkr_position_reconciler.py [--dry-run]
Scheduled via launchd every 15 min, 9:35-16:30 ET weekdays (see
com.ibkrtrader.reconciler.plist) -- runs once and exits, same pattern as
every other standalone trader in this repo.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import IB, ExecutionFilter

sys.path.insert(0, str(Path(__file__).parent))
from alpaca_0dte_common import load_registry, save_registry  # noqa: E402
from ibkr_0dte_common import parse_occ_symbol  # noqa: E402

HERE = Path(__file__).parent
CFG_PATH = HERE / "scanner_config.json"
SAFE_INCOME_STATE_PATH = HERE / "safe_income_auto_state.json"

ET = ZoneInfo("America/New_York")
TWS_PORT = 7496
CLIENT_ID = 2440


def now_et():
    return datetime.now(ET)


def load_cfg():
    try:
        return json.loads(CFG_PATH.read_text())
    except Exception:
        return {}


def telegram_text(cfg, text):
    import requests
    try:
        from telegram_alert_gate import alert_enabled
        if not alert_enabled("reconciliation"):
            return
    except Exception:
        pass
    if not cfg.get("telegram_token"):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      data={"chat_id": cfg["telegram_chat_id"], "text": text, "parse_mode": "HTML"},
                      timeout=10)
    except Exception as e:
        print(f"Telegram send failed: {e}")


def live_option_positions(ib):
    """(ticker, expiry 'YYYYMMDD', right, strike) -> signed position size,
    summed across any duplicate contract rows IBKR occasionally returns."""
    out = {}
    for p in ib.portfolio():
        c = p.contract
        if c.secType != "OPT":
            continue
        key = (c.symbol, c.lastTradeDateOrContractMonth, c.right, float(c.strike))
        out[key] = out.get(key, 0.0) + p.position
    return out


def find_closing_execution(ib, occ_symbol_str):
    """Best-effort lookup of the real fill that closed this contract today.
    Returns (price, time_et) or (None, None) if nothing matching is found
    (executions only reliably cover the current session on this account --
    same limitation already documented for harami_daily_trader.py)."""
    try:
        ticker, expiry, strike, right = parse_occ_symbol(occ_symbol_str)
    except ValueError:
        return None, None
    try:
        fills = ib.reqExecutions(ExecutionFilter(symbol=ticker, secType="OPT"))
    except Exception:
        return None, None
    candidates = [
        f for f in fills
        if f.contract.lastTradeDateOrContractMonth == expiry
        and f.contract.right == right
        and abs(float(f.contract.strike) - strike) < 1e-6
        and f.execution.side == "SLD"
    ]
    if not candidates:
        return None, None
    latest = max(candidates, key=lambda f: f.execution.time)
    return latest.execution.avgPrice, latest.execution.time.astimezone(ET)


def reconcile_alpaca_registry(ib, live, cfg, dry_run):
    """alpaca_0dte_positions.json -- shared by Ashley, the three SPY/QQQ/IWM
    0DTE butterflies, and the GOOG weekly condor."""
    reg = load_registry()
    corrections = []
    for pos_id, pos in list(reg["positions"].items()):
        if pos.get("phase") != "open":
            continue
        legs = pos.get("legs", [])
        try:
            keys = [parse_occ_symbol(leg["symbol"]) for leg in legs]
        except (KeyError, ValueError):
            continue  # can't safely parse -- leave for manual review
        # parse_occ_symbol returns (ticker, expiry, strike, right); live dict
        # is keyed (ticker, expiry, right, strike) -- reorder to match.
        live_keys = [(t, e, r, s) for (t, e, s, r) in keys]
        sizes = [live.get(k, 0.0) for k in live_keys]
        if any(abs(s) > 1e-6 for s in sizes):
            continue  # at least one leg still genuinely held -- leave alone

        # Every leg is flat in the real account but the registry still says
        # open -- try to recover the real closing price/time from today's
        # executions (single-leg positions only; multi-leg net P&L from
        # executions alone is unreliable without also re-deriving qty
        # weighting, so those get flagged with pnl=None instead of guessed).
        pnl = None
        closed_at = now_et().isoformat()
        if len(legs) == 1:
            price, ts = find_closing_execution(ib, legs[0]["symbol"])
            if price is not None:
                entry_fill = legs[0].get("fill") or 0.0
                qty = legs[0].get("qty", 1)
                pnl = round((price - entry_fill) * qty * 100, 2)
                closed_at = ts.isoformat()
        reason = "auto_reconciled_ibkr_flat" if pnl is not None else "auto_reconciled_ibkr_flat_pnl_unknown"
        corrections.append((pos_id, pos["ticker"], reason, pnl, closed_at))
        if not dry_run:
            # Build the closed record directly (rather than close_position() +
            # patch) so the real fill time -- not "now" -- lands in closed_at
            # in a single write.
            reg = load_registry()
            live_pos = reg["positions"].pop(pos_id, None)
            if live_pos is None:
                continue  # already reconciled by a concurrent run
            live_pos["phase"] = "closed"
            live_pos["close_reason"] = reason
            live_pos["close_pnl"] = pnl
            live_pos["closed_at"] = closed_at
            reg.setdefault("closed", []).append(live_pos)
            save_registry(reg)

    for pos_id, ticker, reason, pnl, _ in corrections:
        msg = (f"\U0001F527 <b>ibkr_position_reconciler:</b> {pos_id} ({ticker}) was OPEN in the registry "
               f"but IBKR shows the position fully closed. Reconciled -> CLOSED"
               + (f" (pnl ${pnl:+.2f})" if pnl is not None else " (real pnl unknown -- verify in IBKR)")
               + f". reason={reason}")
        print(msg)
        telegram_text(cfg, msg)
    return corrections


def reconcile_safe_income(ib, live, cfg, dry_run):
    """safe_income_auto_state.json -- credit spreads (short_k/long_k), own
    schema, not on the shared alpaca_0dte registry."""
    try:
        state = json.loads(SAFE_INCOME_STATE_PATH.read_text())
    except FileNotFoundError:
        return []
    positions = state.get("positions", {})
    corrections = []
    for pid, pos in list(positions.items()):
        try:
            ticker, expiry, right = pos["ticker"], pos["expiry"], pos["right"]
            short_k, long_k = float(pos["short_k"]), float(pos["long_k"])
        except (KeyError, ValueError):
            continue
        expiry_yyyymmdd = expiry.replace("-", "") if "-" in expiry else expiry
        short_key = (ticker, expiry_yyyymmdd, right, short_k)
        long_key = (ticker, expiry_yyyymmdd, right, long_k)
        if abs(live.get(short_key, 0.0)) > 1e-6 or abs(live.get(long_key, 0.0)) > 1e-6:
            continue  # still genuinely held (either leg) -- leave alone

        corrections.append((pid, ticker, "auto_reconciled_ibkr_flat_pnl_unknown", None))
        if not dry_run:
            closed_at = now_et().isoformat()
            pos["phase"] = "closed"
            pos["close_reason"] = "auto_reconciled_ibkr_flat"
            pos["closed_at"] = closed_at
            pos["exit_pnl"] = None
            state.setdefault("closed", []).append(pos)
            del state["positions"][pid]
            SAFE_INCOME_STATE_PATH.write_text(json.dumps(state, indent=2))

    for pid, ticker, reason, pnl in corrections:
        msg = (f"\U0001F527 <b>ibkr_position_reconciler:</b> Safe Income {pid} ({ticker}) was OPEN "
               f"but IBKR shows both legs flat. Reconciled -> CLOSED (real pnl unknown -- verify in IBKR).")
        print(msg)
        telegram_text(cfg, msg)
    return corrections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report mismatches without writing any state file")
    args = ap.parse_args()

    cfg = load_cfg()
    ib = IB()
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    try:
        live = live_option_positions(ib)
        alpaca_corrections = reconcile_alpaca_registry(ib, live, cfg, args.dry_run)
        safe_income_corrections = reconcile_safe_income(ib, live, cfg, args.dry_run)
        total = len(alpaca_corrections) + len(safe_income_corrections)
        if total == 0:
            print(f"[{now_et().isoformat()}] reconcile check: no desyncs found "
                  f"({len(live)} live option positions checked).")
        else:
            print(f"[{now_et().isoformat()}] reconcile check: {total} desync(s) "
                  f"{'found (dry-run, not written)' if args.dry_run else 'corrected'}.")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
