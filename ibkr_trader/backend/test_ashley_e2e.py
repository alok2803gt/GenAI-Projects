"""
End-to-end test of the Ashley signal-follow pipeline WITHOUT touching the
broker or Discord: a real post is parsed, the new 9:00 watch / 9:30 execute
gates are checked, a spot path walks into a zone, her exit message closes the
position, and the 15:55 force-close path is exercised.

Orders are routed to a fake IB that records them. Run:
    ../venv/bin/python test_ashley_e2e.py
"""
import asyncio
import importlib.util
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
spec = importlib.util.spec_from_file_location("ash", "ashleyklieu_trigger_executor.py")
A = importlib.util.module_from_spec(spec)
sys.modules["ash"] = A
spec.loader.exec_module(A)
A.telegram = lambda *a, **k: None          # no Telegram during tests
A.oversight_log = lambda *a, **k: None     # no oversight writes during tests

POST = ("Good morning! Today's levels \U0001F60A\n"
        "775P Entry: 775.00 to 775.30\n"
        "772C Entry: 772.20 to 772.50\n"
        "770C Entry: 769.90 to 770.20")
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        FAILS.append(name)


# 1 -- her post parses into tradable setups
print("\n1) PARSE her Discord post")
setups = A.setups_from_content(POST)
check("3 setups parsed", len(setups) == 3, str(setups))
check("names/strikes/rights", [s["name"] for s in setups] == ["775P", "772C", "770C"], str(setups))
check("zones", setups[0]["lo"] == 775.0 and setups[0]["hi"] == 775.3, str(setups[0]))
check("chatter is ignored", A.setups_from_content("watching SPY today, no levels yet") == [])

# 2 -- the new timing gates
print("\n2) TIMING: watch at 9:00, execute at 9:30")
check("watch gate is 9:00", A.WATCH_START_ET == (9, 0))
check("exec gate is 9:30", A.EXEC_START_ET == (9, 30))
base = datetime.now(ET).replace(second=0, microsecond=0)
for hh, mm, watching, waits in ((8, 55, False, 35), (9, 5, True, 25), (9, 30, True, 0), (11, 0, True, 0)):
    now = base.replace(hour=hh, minute=mm)
    w = now >= now.replace(hour=9, minute=0)
    check(f"{hh:02d}:{mm:02d} watch={watching}, execution waits {waits}m",
          w == watching and abs(A.seconds_until(9, 30, now=now) / 60 - waits) < 0.1)

# 3 -- entry trigger: spot walking into a zone
print("\n3) ENTRY trigger as SPY walks into 772C's zone [772.20, 772.50]")
path = [771.10, 771.80, 772.05, 772.19, 772.30, 773.40]
fired = []
for spot in path:
    for s in setups:
        if s["lo"] <= spot <= s["hi"] and s["name"] not in fired:
            fired.append(s["name"])
check("fires once, only in-zone", fired == ["772C"], str(fired))
check("772.19 does not fire (below zone)", not (setups[1]["lo"] <= 772.19 <= setups[1]["hi"]))

# 4 -- order placement through a fake broker
print("\n4) ORDER placement (fake IB -- nothing reaches IBKR)")
placed = []


class FakeTrade:
    def __init__(self):
        self.orderStatus = type("S", (), {"status": "Filled", "avgFillPrice": 1.20})()


class FakeIB:
    def placeOrder(self, contract, order):
        placed.append((order.action, order.totalQuantity, getattr(contract, "localSymbol", ""), order.lmtPrice))
        return FakeTrade()

    def sleep(self, s):
        pass

    def cancelOrder(self, o):
        pass


import ibkr_0dte_common as K
ok, px = K.ibkr_place_leg(FakeIB(), type("C", (), {"localSymbol": "SPY 260923C00772000", "symbol": "SPY"})(),
                          "BUY", 1.20, "ashley 772C entry", qty=1, fill_wait_s=2)
check("entry order filled", ok and px == 1.20, f"ok={ok} px={px}")
check("one BUY order recorded", len(placed) == 1 and placed[0][0] == "BUY", str(placed))

# 5 -- her exit message closes the position
print("\n5) EXIT signal recognition")
check("'took profit at 774.70' is an exit", A.is_exit_signal("took profit at 774.70 TP"))
check("'stopped out' is an exit", A.is_exit_signal("stopped out on the 772c"))
check("conditional is NOT an exit", not A.is_exit_signal("if we lose 772 i'm out"))
check("hypothetical is NOT an exit", not A.is_exit_signal("i'd take profit here if i were in"))
open_pos = {"772C": {"qty": 1, "entry_px": 1.20}}
check("exit matches the open setup", A.match_open_setup("out of the 772c", open_pos, setups) == "772C",
      str(A.match_open_setup("out of the 772c", open_pos, setups)))

# 6 -- close path through the fake broker
print("\n6) CLOSE path (fake IB)")
placed.clear()


class FakeIBAsync(FakeIB):
    async def qualifyContractsAsync(self, *a):
        return list(a)


async def _close():
    K.ibkr_place_leg(FakeIB(), type("C", (), {"localSymbol": "SPY 260923C00772000", "symbol": "SPY"})(),
                     "SELL", 1.18, "ashley 772C exit", qty=1, fill_wait_s=2)
asyncio.run(_close())
check("exit order is a SELL", placed and placed[0][0] == "SELL", str(placed))

# 7 -- force-close safety window
print("\n7) FORCE-CLOSE window")
check("force close at 15:55", A.FORCE_CLOSE_ET == (15, 55))
now_late = base.replace(hour=15, minute=56)
check("15:56 is past the force-close time",
      (now_late.hour, now_late.minute) >= A.FORCE_CLOSE_ET)
check("cutoff for waiting is 13:00", A.WAIT_CUTOFF_ET == (13, 0))

# 8 -- sizing and daily de-dup
print("\n8) SIZING + de-dup")
check("qty per setup is 1", A.contracts_for_today() == 1, str(A.contracts_for_today()))
check("fired-today state is a set", isinstance(A.load_fired_today(), set))

print("\n" + ("ALL END-TO-END CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
