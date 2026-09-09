"""
One-time August 2026 news test digest -- SINGLE summary Telegram message.
Reuses the already-scored, already-cleaned qualifying set from
news_august_backfill.py's last run (562 real CRITICAL/HIGH stories, post
law-firm-spam/ticker-false-positive fixes). Picks a representative,
date-spread subset of the highest-priority tier (CRITICAL + real ticker
match) rather than every story, since the point is a one-time pipeline
test digest, not a full archive (the full 562-story set is already saved
at news_august_backfill_qualifying.json for anyone who wants the rest).
"""
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import main as m
import news_august_backfill as nab

TELEGRAM_LIMIT = 3800
MAX_HIGHLIGHTS = 15

with open("news_august_backfill_qualifying.json", encoding="utf-8") as f:
    qualifying = json.load(f)

total = len(qualifying)
crit = [q for q in qualifying if q["severity"] == "CRITICAL"]
high = [q for q in qualifying if q["severity"] == "HIGH"]
crit_ticker = [q for q in crit if q["tickers"]]
crit_market = [q for q in crit if not q["tickers"]]

crit_ticker.sort(key=lambda q: q["published_utc"])
stride = max(len(crit_ticker) // MAX_HIGHLIGHTS, 1)
selected = crit_ticker[::stride][:MAX_HIGHLIGHTS]

lines = []
for q in selected:
    verdict = m._nm_build_verdict(q["title"], q["desc"], q["tickers"], q["severity"], q["n_sources"])
    date_str = q["published_utc"][:10]
    lines.append(f"{date_str} {verdict}\n{q['url']}")

header = (
    f"AUGUST 2026 NEWS PIPELINE TEST -- one-time backfill, live incremental feed resumes from today.\n\n"
    f"562 CRITICAL/HIGH stories found this month (of 3,662 raw articles scanned; "
    f"231 law-firm-spam/bad-ticker false positives already removed by today's fixes).\n"
    f"Breakdown: {len(crit)} CRITICAL ({len(crit_ticker)} ticker-specific, {len(crit_market)} market-wide), "
    f"{len(high)} HIGH.\n\n"
    f"Below: {len(selected)} representative CRITICAL ticker-specific highlights, spread across the month "
    f"(full 562-story list saved locally, not sent here):\n"
)

body = header
for line in lines:
    candidate = body + "\n" + line
    if len(candidate) > TELEGRAM_LIMIT:
        break
    body = candidate

print(f"Final message length: {len(body)} chars, {body.count(chr(10)+chr(10))} items included")
print("=" * 60)
print(body)
print("=" * 60)

token, chat_id = m._load_telegram_creds()
ok = nab.send_telegram_plain(token, chat_id, body)
print("SENT:", ok)
