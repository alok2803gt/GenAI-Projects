"""
Second correction for TODAY's (2026-09-02) SPY 0DTE position. Sequence:
  1. Butterfly entered 10:29 ET (761C x1 long / 765C x2 short / 769C x1 long).
  2. User manually bought back 1x body at 10:59:55 ET -> became a 1:1:1
     ratio spread. close_spy_ratio_position_today.py was launched for that
     shape (killed once this script supersedes it).
  3. User decided (thinks SPY stays in this range) to sell 1x body back at
     11:2x ET, RESTORING the real 1:2:1 butterfly. This script reflects
     THAT current, real state -- confirmed via Alpaca:
       761C long 1x  @ 4.60
       765C short 2x @ 1.275 (blended: 1.44 first fill + 1.11 second fill)
       769C long 1x  @ 0.16
     Real net entry debit = 4.60 - 2*1.275 + 0.16 = $2.21/contract ($221 total).

Uses the real place_butterfly_sequential/close_butterfly_sequential
primitives (2x body, matching a real butterfly) instead of the 1:1:1
one-off math from the previous correction. Same MUST-CLOSE rule: SPY
options are physically settled, force-close at 15:55 ET regardless of
P&L, no exceptions.
"""
import time
from datetime import datetime

from ib_insync import IB

from alpaca_0dte_common import (
    load_config, alpaca_client, get_quote, etf_option, target_px,
    close_butterfly_sequential, now_et,
)

TWS_PORT = 7496
CLIENT_ID = 1664
HARD_CLOSE_TIME = "15:55"
MONITOR_INTERVAL_S = 60
TODAY_IBKR = "20260902"

STRIKES = {"wing_lo": 761.0, "body": 765.0, "wing_hi": 769.0}
SYMS = {"wing_lo": "SPY260902C00761000", "body": "SPY260902C00765000", "wing_hi": "SPY260902C00769000"}
NET_ENTRY_DEBIT = 4.60 - 2 * 1.275 + 0.16  # = 2.21


def mark(ib):
    legs = {name: etf_option("SPY", TODAY_IBKR, strike, "C") for name, strike in STRIKES.items()}
    quotes = {name: get_quote(ib, c) for name, c in legs.items()}
    if any(not q["mid"] for q in quotes.values()):
        return None, quotes
    value = quotes["wing_lo"]["mid"] + quotes["wing_hi"]["mid"] - 2 * quotes["body"]["mid"]
    return value, quotes


def main():
    cfg = load_config()
    client = alpaca_client(cfg)
    print(f"=== SPY butterfly monitor (RESTORED to 1:2:1) -- entry_net_debit=${NET_ENTRY_DEBIT:.2f} ===")
    print(f"Positions: {SYMS}")

    entry_time = now_et()
    hard_close_dt = datetime.strptime(f"{entry_time.strftime('%Y-%m-%d')} {HARD_CLOSE_TIME}", "%Y-%m-%d %H:%M")
    hard_close_dt = hard_close_dt.replace(tzinfo=entry_time.tzinfo)

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    try:
        while True:
            now = now_et()
            value, quotes = mark(ib)
            if value is None:
                print(f"[{now.strftime('%H:%M:%S')}] quote gap, retrying next cycle")
            else:
                live_pnl = (value - NET_ENTRY_DEBIT) * 100
                print(f"[{now.strftime('%H:%M:%S')}] butterfly_value=${value:.2f}  live_pnl=${live_pnl:+.2f}")
            if now >= hard_close_dt:
                print(f"\n{HARD_CLOSE_TIME} HARD CLOSE -- closing regardless of P&L (physically-settled options).")
                _, quotes = mark(ib)
                for _ in range(3):
                    if quotes and all(q["bid"] and q["ask"] for q in quotes.values()):
                        break
                    time.sleep(5)
                    _, quotes = mark(ib)
                if quotes and all(q["mid"] for q in quotes.values()):
                    close_limits = {
                        "wing_lo": target_px(quotes["wing_lo"], is_short=True),
                        "wing_hi": target_px(quotes["wing_hi"], is_short=True),
                        "body":    target_px(quotes["body"], is_short=False),
                    }
                else:
                    print("WARNING: no live quotes for close -- using aggressive fallback.")
                    close_limits = {"wing_lo": 0.01, "wing_hi": 0.01, "body": 999.0}
                ok, close_fills = close_butterfly_sequential(client, SYMS, close_limits, 1)
                final_value, _ = mark(ib)
                final_pnl = (final_value - NET_ENTRY_DEBIT) * 100 if final_value is not None else None
                print(f"Closed. ok={ok} fills={close_fills} final_pnl={final_pnl}")
                if not ok:
                    print("WARNING: close did not fully confirm -- verify Alpaca positions manually NOW.")
                return
            time.sleep(MONITOR_INTERVAL_S)
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
