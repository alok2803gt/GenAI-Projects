"""
Real-time monitor for ashleyklieu's SPY alerts in the Discord "alerts(rose)"
channel (server 1540611610027102319, channel 1540614267152371752). Built
2026-08-26 -- CEO has personally tracked this trader's calls outside this
system and is confident in them, but this system has zero independent
track record and the execution pipeline is brand new, so this is
MONITOR-AND-ALERT ONLY. No IBKR/Alpaca orders are placed here -- every
alert goes to Telegram for the CEO to act on manually. Auto-execution is a
real, separate, later decision once there's a track record and a bounded
test budget, not something to skip straight to.

Polls Discord's REST API every ~15s (well within rate limits for a single
channel) using the `after` message-ID cursor so only genuinely new
messages are ever alerted on -- no duplicates, no re-alerting on restart.

Message parsing is deliberately conservative: a clear "<strike><right>
Entry: <lo> to <hi>" pattern gets structured extraction (can appear
multiple times in one message); everything else (entry confirmations that
reference an earlier message's strike, profit-taking notes, general
commentary) gets classified by simple keyword heuristics and forwarded
with the RAW original text, never a guessed structure -- the risk of
silently mis-parsing an ambiguous message ("I'm in now, 764.2 will be
stop, entry 764.5" -- which setup?) into a wrong structured summary is
worse than just showing the real text and letting a human read it.

Usage: python ashleyklieu_alert_monitor.py
Runs continuously; only actively alerts within regular market hours
(9:30-16:00 ET, weekdays) -- outside that window it stays quiet (still
polls, in case pre/post-market commentary matters, but the account's own
real trading day is what these alerts are for).
"""
import json
import re
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

# Same fix as darkpool_activity_monitor.py (2026-08-24): Windows console
# cp1252 can't print some characters this script's own output uses.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONFIG_PATH = "scanner_config.json"
STATE_FILE = "ashleyklieu_monitor_state.json"
LOG_FILE = "oversight_log.jsonl"
CHANNEL_ID = "1540614267152371752"
TARGET_USERNAME = "ashleyklieu"
POLL_SECONDS = 15
ET = ZoneInfo("America/New_York")

ENTRY_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*([CP])\b[^0-9]{0,15}Entry:?\s*(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def telegram(token, chat_id, msg, high_priority=False):
    try:
        prefix = "🚨 " if high_priority else "📡 "
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": prefix + msg, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as e:
        print(f"Telegram send failed: {e}")


def log_line(summary: str, rationale: str):
    entry = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(),
        "actor": "trader",
        "category": "discord_alert_monitor",
        "summary": summary,
        "rationale": rationale,
        "outcome": "Forwarded to Telegram -- monitor-only, no orders placed.",
        "pnl_impact": None,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"last_message_id": None}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)


def is_market_hours() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t


def classify_and_format(content: str) -> str:
    """Best-effort classification -- ALWAYS includes the raw text. Never
    invents a structured trade the message doesn't clearly state."""
    setups = ENTRY_PATTERN.findall(content)
    lower = content.lower()

    if setups:
        lines = [f"<b>New setup(s) posted:</b>"]
        for strike, right, lo, hi in setups:
            side = "CALL" if right.upper() == "C" else "PUT"
            lines.append(f"  SPY {strike}{right.upper()} ({side}) — entry {lo} to {hi}")
        lines.append(f"\nRaw: {content}")
        return "\n".join(lines)

    if any(k in lower for k in ("i'm in", "im in", "entry", "stop")) and "entry" in lower:
        return f"<b>Possible entry confirmation</b> (strike may refer to an earlier message -- read raw text):\n{content}"

    if any(k in lower for k in ("took profit", "tp hit", "hit", "stopped", "closed")):
        return f"<b>Possible exit/update:</b>\n{content}"

    return f"<b>New message:</b>\n{content}"


def main():
    cfg = load_config()
    token, chat_id = cfg["telegram_token"], cfg["telegram_chat_id"]
    bot_token = cfg["discord_bot_token"]
    headers = {"Authorization": f"Bot {bot_token}"}

    state = load_state()
    print(f"Starting ashleyklieu alert monitor. last_message_id={state.get('last_message_id')}")

    # On first-ever run, seed from the current latest message so we don't
    # replay the whole channel history as "new" alerts.
    if state.get("last_message_id") is None:
        r = requests.get(f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
                          headers=headers, params={"limit": 1}, timeout=15)
        r.raise_for_status()
        msgs = r.json()
        if msgs:
            state["last_message_id"] = msgs[0]["id"]
            save_state(state)
            print(f"Seeded last_message_id={state['last_message_id']} (not alerting on channel history)")

    while True:
        try:
            params = {"limit": 20}
            if state.get("last_message_id"):
                params["after"] = state["last_message_id"]
            r = requests.get(f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
                              headers=headers, params=params, timeout=15)
            if r.status_code == 429:
                retry_after = r.json().get("retry_after", 5)
                time.sleep(float(retry_after) + 1)
                continue
            r.raise_for_status()
            msgs = r.json()  # Discord returns newest-first even with `after`
            msgs.sort(key=lambda m: int(m["id"]))  # process oldest-first

            for m in msgs:
                author = m.get("author", {}).get("username", "")
                content = m.get("content", "").strip()
                state["last_message_id"] = m["id"]

                if author != TARGET_USERNAME:
                    continue
                if not content and not m.get("attachments"):
                    continue

                atts = m.get("attachments", [])
                att_note = f"\n({len(atts)} image/attachment — check Discord to view)" if atts else ""

                if is_market_hours():
                    formatted = classify_and_format(content or "(image/attachment only, no text)") + att_note
                    telegram(token, chat_id, f"ashleyklieu alert:\n{formatted}")
                    log_line(f"New ashleyklieu message forwarded: {content[:200]}",
                             "Monitor-only per CEO instruction 2026-08-26 -- no auto-execution yet.")
                    print(f"[ALERTED] {content[:100]}")
                else:
                    print(f"[OUTSIDE MARKET HOURS, logged only] {content[:100]}")
                    log_line(f"New ashleyklieu message (outside market hours, not alerted): {content[:200]}",
                             "Outside 9:30-16:00 ET -- logged but not pushed to Telegram.")

            save_state(state)

        except Exception as exc:
            print(f"Monitor loop error: {exc}")
            time.sleep(5)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
