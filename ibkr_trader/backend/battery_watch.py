"""Power watchdog for this laptop. Task Scheduler (IBKR-BatteryWatch) runs it every minute, 24/7.

On battery, every trader, monitor and TWS keeps running (tasks are allowed to), which drains the
battery fast -- and when the laptop sleeps, all of them freeze with positions unwatched
(2026-10-02: unplugged ~11:00, slept 12:47-17:06, 0DTE condors expired unmonitored).

So: alert the minute it goes on battery, then repeat -- every 5 min above 50%, every 2 min
from 20-50%, every minute below 20% -- and confirm when AC power is back. Stdlib only.
"""
import ctypes
import json
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "battery_watch_state.json"
LOG = HERE / "battery_watch.log"


class SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def power():
    s = SYSTEM_POWER_STATUS()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    if s.BatteryFlag == 128:                      # no system battery
        return None
    pct = s.BatteryLifePercent if s.BatteryLifePercent != 255 else -1
    return s.ACLineStatus == 0, pct


def send(text):
    with open(LOG, "a") as f:
        f.write(f"{datetime.now().isoformat()} {text}\n")
    cfg = json.loads((HERE / "scanner_config.json").read_text())
    body = json.dumps({"chat_id": cfg["telegram_chat_id"], "text": text}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        with open(LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()} telegram failed: {e}\n")


def main():
    p = power()
    if p is None:
        return
    on_battery, pct = p
    now = datetime.now()
    prev = json.loads(STATE.read_text()) if STATE.exists() else {}
    if on_battery:
        since = datetime.fromisoformat(prev["since"]) if prev.get("on_battery") else now
        last = datetime.fromisoformat(prev["last_alert"]) if prev.get("on_battery") and prev.get("last_alert") else None
        every = 1 if pct < 20 else 2 if pct < 50 else 5
        if last is None or (now - last).total_seconds() >= every * 60 - 10:
            mins = int((now - since).total_seconds() // 60)
            send(f"[HIGH] POWER: laptop on BATTERY {pct}% (unplugged {mins} min). PLUG IN -- when it sleeps, "
                 f"every trader, monitor and TWS stops and open positions go unwatched. "
                 f"Repeating every {every} min until AC is back.")
            last = now
        STATE.write_text(json.dumps({"on_battery": True, "since": since.isoformat(),
                                     "last_alert": last.isoformat(), "pct": pct}))
    else:
        if prev.get("on_battery"):
            mins = int((now - datetime.fromisoformat(prev["since"])).total_seconds() // 60)
            send(f"POWER: back on AC ({pct}%) after {mins} min on battery.")
        STATE.write_text(json.dumps({"on_battery": False, "pct": pct}))


if __name__ == "__main__":
    main()
