"""
One-off corrected monitor/close for TODAY's (2026-09-02) SPY 0DTE position,
after the user manually bought back 1x of the butterfly's short body leg
(10:59:55 ET, BUY 1x SPY765C @ $1.30), converting it from the original
1:2:1 long butterfly (761C x1 long / 765C x2 short / 769C x1 long) into a
1:1:1 ratio spread (761C x1 long / 765C x1 short / 769C x1 long).

Replaces the original alpaca_0dte_butterfly_trader.py monitoring process
(PID 4484, killed 2026-09-02 ~11:01 ET) -- that process still held qty=1
with a hardcoded "body = 2x" close formula and would have overshot by one
contract at the 15:55 close, buying back 2x on a position only short 1x
and flipping into an unintended net-long 765C. This script uses the real,
current 1:1:1 quantities throughout.

Real current position (confirmed via Alpaca 2026-09-02 11:01 ET):
  761C long 1x  @ 4.60
  765C short 1x @ 1.44
  769C long 1x  @ 0.16

Same MUST-CLOSE rule as the original script: SPY options are physically
settled, so this is force-closed at 15:55 ET regardless of P&L, no
exceptions -- not left to expire.
"""
import sys
import time
from datetime import datetime

from ib_insync import IB

from alpaca_0dte_common import load_config, alpaca_client, get_quote, etf_option, target_px, place_leg_with_ladder, now_et
from alpaca.trading.enums import OrderSide

TWS_PORT = 7496
CLIENT_ID = 1663
HARD_CLOSE_TIME = "15:55"
MONITOR_INTERVAL_S = 60
TODAY_IBKR = "20260902"

STRIKES = {"wing_lo": 761.0, "body": 765.0, "wing_hi": 769.0}
SYMS = {"wing_lo": "SPY260902C00761000", "body": "SPY260902C00765000", "wing_hi": "SPY260902C00769000"}
ENTRY_PRICES = {"wing_lo": 4.60, "body": 1.44, "wing_hi": 0.16}  # real fills, for P&L reporting only


def mark(ib):
    """1:1:1 structure value = wing_lo + wing_hi - body (NOT -2*body)."""
    legs = {name: etf_option("SPY", TODAY_IBKR, strike, "C") for name, strike in STRIKES.items()}
    quotes = {name: get_quote(ib, c) for name, c in legs.items()}
    if any(not q["mid"] for q in quotes.values()):
        return None, quotes
    value = quotes["wing_lo"]["mid"] - quotes["body"]["mid"] + quotes["wing_hi"]["mid"]
    return value, quotes


def main():
    cfg = load_config()
    client = alpaca_client(cfg)
    entry_net = ENTRY_PRICES["wing_lo"] - ENTRY_PRICES["body"] + ENTRY_PRICES["wing_hi"]  # 3.32 debit
    print(f"=== SPY 1:1:1 ratio spread monitor (post-manual-adjustment) -- entry_net_debit=${entry_net:.2f} ===")
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
                live_pnl = (value - entry_net) * 100
                print(f"[{now.strftime('%H:%M:%S')}] position_value=${value:.2f}  live_pnl=${live_pnl:+.2f}")
            if now >= hard_close_dt:
                print(f"\n{HARD_CLOSE_TIME} HARD CLOSE -- closing regardless of P&L (physically-settled options).")
                _, quotes = mark(ib)
                for _ in range(3):
                    if quotes and all(q["bid"] and q["ask"] for q in quotes.values()):
                        break
                    time.sleep(5)
                    _, quotes = mark(ib)

                # Two-phase close, this account's standing rule: buy back the short first.
                print("--- Close 1/3: BUY TO CLOSE body (765C, 1x) ---")
                q = quotes["body"]
                px = target_px(q, is_short=False) if q["mid"] else 999.0
                ok1, fill1 = place_leg_with_ladder(client, SYMS["body"], OrderSide.BUY, "close body", 1,
                                                    q["bid"], q["ask"], q["mid"]) if q["mid"] else (False, None)

                print("--- Close 2/3: SELL TO CLOSE wing_lo (761C, 1x) ---")
                q = quotes["wing_lo"]
                ok2, fill2 = place_leg_with_ladder(client, SYMS["wing_lo"], OrderSide.SELL, "close wing_lo", 1,
                                                    q["bid"], q["ask"], q["mid"]) if q["mid"] else (False, None)

                print("--- Close 3/3: SELL TO CLOSE wing_hi (769C, 1x) ---")
                q = quotes["wing_hi"]
                ok3, fill3 = place_leg_with_ladder(client, SYMS["wing_hi"], OrderSide.SELL, "close wing_hi", 1,
                                                    q["bid"], q["ask"], q["mid"]) if q["mid"] else (False, None)

                print(f"\nClose results: body(buy)={fill1} wing_lo(sell)={fill2} wing_hi(sell)={fill3} "
                      f"all_ok={ok1 and ok2 and ok3}")
                if not (ok1 and ok2 and ok3):
                    print("WARNING: at least one leg did not confirm -- check Alpaca positions manually NOW.")
                return
            time.sleep(MONITOR_INTERVAL_S)
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
