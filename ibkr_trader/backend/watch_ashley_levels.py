"""Watch Ashley's channel for level changes the live executor cannot yet see.

WHY (2026-09-29): the running executor (pid 50660) predates the strikethrough
retirement fix, and restarting it to pick that fix up would replay her 10:51
"765 TP" against the now-open 765C and close it at market -- which the CEO has
explicitly chosen not to do. So the gap is covered by watching from outside
instead of restarting.

Exits as soon as anything actionable appears, so the session is notified:
  * a struck-through (retired) level that is still ARMED in the executor
    -- the exact condition that caused the 765C to be bought 9 minutes after
    she closed it;
  * a NEW entry level not in today's parsed setups;
  * an exit/TP signal while a position is open.

Read-only. Places no orders and does not touch the executor's state.
"""
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

HERE = Path(__file__).parent
ET = ZoneInfo("America/New_York")
CHANNEL_ID = "1540614267152371752"
WHO = "ashleyklieu"
STATE = HERE / "ashleyklieu_trigger_executor_state.json"
FIRED = HERE / "ashleyklieu_fired_today.json"

ENTRY_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*([CP])\b[^0-9]{0,15}Entry:?\s*(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE)
STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
TP_WORD = re.compile(r"\btp\b", re.IGNORECASE)


def names_in(text: str) -> set:
    return {f"{float(m.group(1)):g}{m.group(2).upper()}" for m in ENTRY_PATTERN.finditer(text or "")}


def main() -> None:
    cfg = json.load(open(HERE / "scanner_config.json"))
    headers = {"Authorization": f"Bot {cfg['discord_bot_token']}"}
    st = json.load(open(STATE))
    armed = {s["name"] for s in st.get("alert_setups", [])}
    try:
        fired = set(json.load(open(FIRED)).get("fired", []))
    except Exception:
        fired = set()
    live = armed - fired
    print(f"watching {WHO}; armed-and-unfired: {sorted(live) or '[]'}", flush=True)

    # Prime `seen` with everything already posted, WITHOUT acting on it. The
    # first version of this script started with an empty set, so its first poll
    # treated the whole day's backlog as new and immediately fired on the
    # historical 10:51 "765 TP" -- replaying a stale signal, the same class of
    # bug this watcher exists to catch. Only messages posted from now on count.
    seen = set()
    priming = True
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        now = datetime.now(ET)
        if (now.hour, now.minute) >= (16, 0):
            print("market closed -- stopping watch", flush=True)
            return
        try:
            r = requests.get(f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
                             headers=headers, params={"limit": 30}, timeout=20)
            if r.status_code == 429:
                time.sleep(float(r.json().get("retry_after", 5)) + 1)
                continue
            r.raise_for_status()
            msgs = sorted(r.json(), key=lambda m: int(m["id"]))
        except Exception as exc:
            print(f"{now:%H:%M:%S} poll failed: {type(exc).__name__}", flush=True)
            time.sleep(60)
            continue

        for m in msgs:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            if priming:
                continue                      # backlog: record, never act
            if m.get("author", {}).get("username") != WHO:
                continue
            ts = datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00")).astimezone(ET)
            if ts.date() != now.date():
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue

            struck = set()
            for span in STRIKE_RE.findall(content):
                struck |= names_in(span)
            retired_live = struck & live
            fresh = names_in(STRIKE_RE.sub("", content)) - armed

            if retired_live:
                print(f"\n*** RETIRED WHILE ARMED: {sorted(retired_live)} ***", flush=True)
                print(f"    {ts:%H:%M} ET: {content[:200]}", flush=True)
                print("    The executor will still buy these if price enters the zone.", flush=True)
                return
            if fresh:
                print(f"\n*** NEW LEVEL(S) NOT IN TODAY'S SETUPS: {sorted(fresh)} ***", flush=True)
                print(f"    {ts:%H:%M} ET: {content[:200]}", flush=True)
                return
            if TP_WORD.search(content):
                print(f"\n*** TP / EXIT SIGNAL ***  {ts:%H:%M} ET: {content[:200]}", flush=True)
                return
            print(f"  {ts:%H:%M} {content[:110]}", flush=True)
        if priming:
            print(f"primed with {len(seen)} existing messages; now watching for NEW ones only",
                  flush=True)
            priming = False
        time.sleep(45)
    print("watch window elapsed", flush=True)


if __name__ == "__main__":
    sys.exit(main())
