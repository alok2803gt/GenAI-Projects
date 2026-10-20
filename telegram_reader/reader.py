#!/usr/bin/env python3
"""Read-only access to the user's own Telegram account (user ask 2026-10-16: "set up telegram
connector to be able to read my telegram chats").

READ-ONLY BY DESIGN: this file only lists chats, reads and searches messages. It contains no code that
sends, edits, deletes, forwards, reacts or marks anything as read -- reading here does not clear your
unread badges. Keep it that way; a sender belongs in a separate, deliberately approved tool.

Secrets never reach git:
  api_id/api_hash  "telegram_api_id"/"telegram_api_hash" in ibkr_trader/backend/scanner_config.json
                   (gitignored), from https://my.telegram.org -> API development tools. Fallback:
                   %USERPROFILE%\\.telegram_reader\\config.json {"api_id": ..., "api_hash": "..."}
  account.session  in %USERPROFILE%\\.telegram_reader\\, created by `login`. It is a full login to your account: treat it like a password.
                   Delete it (or end the session in Telegram > Settings > Devices) to revoke access.

    python reader.py login                        # once, in YOUR terminal: phone, code, 2FA password
    python reader.py chats [--limit 50]           # chats, newest first, with unread counts
    python reader.py read <chat> [--limit 30] [--since 2026-10-15]
    python reader.py search <text> [--chat <chat>] [--limit 30]
<chat> is a numeric id from `chats`, an @username, or part of the chat's name.
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from telethon import TelegramClient

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOME = Path.home() / ".telegram_reader"
CONFIG = HOME / "config.json"
SCANNER_CONFIG = Path(__file__).resolve().parent.parent / "ibkr_trader" / "backend" / "scanner_config.json"
SESSION = HOME / "account"  # Telethon adds .session
MAX_TEXT = 2000


def credentials():
    """api_id/api_hash: telegram_api_id/telegram_api_hash in the gitignored scanner_config.json (user
    choice 2026-10-16), else ~/.telegram_reader/config.json."""
    if SCANNER_CONFIG.exists():
        sc = json.loads(SCANNER_CONFIG.read_text(encoding="utf-8"))
        if str(sc.get("telegram_api_id", "")).strip() and str(sc.get("telegram_api_hash", "")).strip():
            return int(sc["telegram_api_id"]), sc["telegram_api_hash"].strip()
    if CONFIG.exists():
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        return int(cfg["api_id"]), cfg["api_hash"]
    sys.exit(f"No Telegram API credentials. Fill telegram_api_id and telegram_api_hash in {SCANNER_CONFIG} "
             f"with the values from https://my.telegram.org (API development tools).")


def client():
    HOME.mkdir(exist_ok=True)  # the login session stays here, outside the repo
    api_id, api_hash = credentials()
    return TelegramClient(str(SESSION), api_id, api_hash)


async def resolve(c, chat):
    """A numeric id, an @username, or a case-insensitive piece of a chat's name."""
    if chat.lstrip("-").isdigit():
        return await c.get_entity(int(chat))
    if chat.startswith("@"):
        return await c.get_entity(chat)
    hits = [d for d in await c.get_dialogs(limit=500) if chat.lower() in (d.name or "").lower()]
    if not hits:
        sys.exit(f"no chat matching {chat!r} -- run `chats` for ids")
    if len(hits) > 1:
        exact = [d for d in hits if (d.name or "").lower() == chat.lower()]
        if len(exact) != 1:
            sys.exit("several chats match: " + "; ".join(f"{d.name} ({d.id})" for d in hits[:15])
                     + " -- use the id")
        hits = exact
    return hits[0].entity


def sender_name(m):
    s = m.sender
    if s is None:
        return "?"
    return getattr(s, "title", None) or " ".join(x for x in (getattr(s, "first_name", None),
                                                               getattr(s, "last_name", None)) if x) \
        or getattr(s, "username", None) or str(m.sender_id)


def show(m, chat_label=None):
    text = (m.message or "").strip()
    if m.media and not text:
        text = f"[{type(m.media).__name__.replace('MessageMedia', '').lower() or 'media'}]"
    elif m.media:
        text += f"  [+{type(m.media).__name__.replace('MessageMedia', '').lower()}]"
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT] + " ...[truncated]"
    where = f"[{chat_label}] " if chat_label else ""
    print(f"{m.date.astimezone():%Y-%m-%d %H:%M} {where}{sender_name(m)}: {text}")


async def cmd_chats(c, a):
    for d in await c.get_dialogs(limit=a.limit):
        last = d.date.astimezone().strftime("%Y-%m-%d %H:%M") if d.date else ""
        kind = "channel" if d.is_channel and not d.is_group else "group" if d.is_group else "user"
        print(f"{d.id:>15}  {kind:<7} unread {d.unread_count:>4}  {last}  {d.name}")


async def cmd_read(c, a):
    ent = await resolve(c, a.chat)
    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc) if a.since else None
    msgs = []
    async for m in c.iter_messages(ent, limit=a.limit):
        if since and m.date < since:
            break
        msgs.append(m)
    for m in reversed(msgs):  # oldest first, like a chat
        show(m)


async def cmd_search(c, a):
    ent = await resolve(c, a.chat) if a.chat else None
    async for m in c.iter_messages(ent, search=a.text, limit=a.limit):
        label = None
        if ent is None:
            chat = await m.get_chat()
            label = getattr(chat, "title", None) or getattr(chat, "first_name", None) or str(m.chat_id)
        show(m, label)


def main():
    p = argparse.ArgumentParser(description="Read-only Telegram reader")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login")
    s = sub.add_parser("send-code"); s.add_argument("phone")
    s = sub.add_parser("verify"); s.add_argument("code")
    s = sub.add_parser("chats"); s.add_argument("--limit", type=int, default=50)
    s = sub.add_parser("read"); s.add_argument("chat"); s.add_argument("--limit", type=int, default=30)
    s.add_argument("--since")
    s = sub.add_parser("search"); s.add_argument("text"); s.add_argument("--chat")
    s.add_argument("--limit", type=int, default=30)
    a = p.parse_args()

    asyncio.run(run(a))


PENDING = HOME / "pending_login.json"


async def run(a):
    c = client()
    try:
        # Two-step login for a non-interactive shell: send-code, then verify with the code Telegram
        # sends to the user's app. The phone_code_hash links the two; it is kept next to the session.
        if a.cmd == "send-code":
            await c.connect()
            sent = await c.send_code_request(a.phone)
            PENDING.write_text(json.dumps({"phone": a.phone, "hash": sent.phone_code_hash}))
            print(f"code sent to the Telegram app for {a.phone[:-4]}****")
            return
        if a.cmd == "verify":
            from telethon.errors import SessionPasswordNeededError
            p = json.loads(PENDING.read_text())
            await c.connect()
            try:
                await c.sign_in(phone=p["phone"], code=a.code, phone_code_hash=p["hash"])
            except SessionPasswordNeededError:
                sys.exit("this account has a 2FA password -- finish with `python reader.py login` in your "
                         "own terminal (it asks for the password; don't send it in chat)")
            PENDING.unlink(missing_ok=True)
            me = await c.get_me()
            print(f"logged in as {me.first_name} (@{me.username}) -- session saved to {SESSION}.session")
            return
        if a.cmd == "login":
            await c.start()  # prompts for phone, the code Telegram sends you, and your 2FA password if set
            me = await c.get_me()
            print(f"logged in as {me.first_name} (@{me.username}) -- session saved to {SESSION}.session")
            return
        await c.connect()
        if not await c.is_user_authorized():
            sys.exit("not logged in -- run `python reader.py login` in your own terminal first")
        await {"chats": cmd_chats, "read": cmd_read, "search": cmd_search}[a.cmd](c, a)
    finally:
        await c.disconnect()


if __name__ == "__main__":
    main()
