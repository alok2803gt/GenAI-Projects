"""
One-time historical backfill: pulls real August 2026 news from Polygon.io's
news archive (this account's existing Polygon key -- confirmed via a real
call to already have historical date-range access, unlike Polygon's options
quotes endpoint which this plan lacks), filters it through the exact same
severity/ticker logic as the live RSS-based news monitor in main.py, runs
survivors through the new FinBERT+deterministic-override verdict pipeline,
and sends ONE consolidated Telegram digest (chunked to Telegram's message
limit) -- a real end-to-end test of the new pipeline against a real month
of data, not a fabricated sample set.

After this runs, the live news_monitor_coro() in main.py (already running,
60s poll interval) picks up incrementally from today onward -- this script
does not touch its "seen" dedup state, so there is no overlap/double-alert
risk with the live loop.

Known approximation, disclosed rather than hidden: _nm_build_verdict()'s
HELD/WATCHED split checks TODAY's real open positions against each
August-dated headline. A ticker held in August but closed since (or opened
since but not in August) will be tiered against today's book, not August's
-- there is no historical position snapshot to check against instead.
"""
import sys
import io
import json
import hashlib
from collections import defaultdict
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import requests
import main as m

AUG_START = "2026-08-01T00:00:00Z"
AUG_END   = "2026-09-01T00:00:00Z"
TELEGRAM_CHUNK_LIMIT = 3500


def fetch_polygon_news(api_key: str) -> list:
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "published_utc.gte": AUG_START,
        "published_utc.lt": AUG_END,
        "limit": 1000,
        "order": "asc",
        "sort": "published_utc",
        "apiKey": api_key,
    }
    articles = []
    page = 0
    while url:
        r = requests.get(url, params=params if page == 0 else None, timeout=30)
        r.raise_for_status()
        data = r.json()
        articles.extend(data.get("results", []))
        url = data.get("next_url")
        if url and "apiKey=" not in url:
            url = url + ("&" if "?" in url else "?") + f"apiKey={api_key}"
        params = None
        page += 1
        print(f"  page {page}: total so far {len(articles)}")
        if page > 50:  # safety cap, ~50k articles
            break
    return articles


def is_noise(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in m._NEWS_NOISE_BLOCKLIST)


def main_run():
    with open("scanner_config.json") as f:
        cfg = json.load(f)
    polygon_key = cfg["polygon_api_key"]

    print("Fetching real Polygon news for August 2026...")
    articles = fetch_polygon_news(polygon_key)
    print(f"Total raw articles fetched: {len(articles)}")

    # Group by fingerprint to (a) dedup syndicated copies (b) count real
    # independent-source coverage, mirroring the live pipeline's n_sources.
    groups: dict = defaultdict(list)
    for a in articles:
        title = (a.get("title") or "").strip()
        if not title or is_noise(title):
            continue
        fp = m._nm_fingerprint(title)
        groups[fp].append(a)

    print(f"After noise-blocklist + fingerprint grouping: {len(groups)} distinct stories")

    qualifying = []
    for fp, group in groups.items():
        group.sort(key=lambda a: a.get("published_utc", ""))
        rep = group[0]
        title = rep.get("title", "").strip()
        desc = (rep.get("description") or "")[:350]
        severity = m._nm_severity(title, desc)
        if severity == "NORMAL":
            continue
        tickers = m._nm_extract_tickers(title + " " + desc)
        publishers = {a.get("publisher", {}).get("name", "") for a in group}
        n_sources = len(publishers) if len(publishers) > 1 else len(group)
        qualifying.append({
            "published_utc": rep.get("published_utc", ""),
            "title": title,
            "desc": desc,
            "url": rep.get("article_url", ""),
            "severity": severity,
            "tickers": tickers,
            "n_sources": max(n_sources, 1),
        })

    qualifying.sort(key=lambda x: x["published_utc"])
    print(f"CRITICAL/HIGH qualifying stories: {len(qualifying)}")
    crit = sum(1 for q in qualifying if q["severity"] == "CRITICAL")
    high = len(qualifying) - crit
    print(f"  CRITICAL: {crit} | HIGH: {high}")

    # Save the full qualifying set to disk regardless of send outcome/cap,
    # so the real underlying data is inspectable even if the Telegram
    # digest below caps/truncates for message-length reasons.
    with open("news_august_backfill_qualifying.json", "w", encoding="utf-8") as f:
        json.dump(qualifying, f, indent=2, ensure_ascii=False)
    print("Full qualifying set saved to news_august_backfill_qualifying.json")

    print("\nRunning FinBERT + override verdict on each qualifying story...")
    lines = []
    for i, q in enumerate(qualifying):
        verdict = m._nm_build_verdict(q["title"], q["desc"], q["tickers"], q["severity"], q["n_sources"])
        date_str = q["published_utc"][:10]
        line = f"{date_str} {verdict}\n{q['url']}"
        lines.append(line)
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(qualifying)}")

    return qualifying, lines


def send_telegram_plain(token: str, chat_id: str, text: str) -> bool:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=15,
        )
        if not r.ok:
            print(f"Telegram send FAILED [HTTP {r.status_code}]: {r.text[:300]}")
            return False
        return True
    except Exception as exc:
        print(f"Telegram send FAILED: {exc}")
        return False


def chunk_and_send(lines: list, total_found: int):
    token, chat_id = m._load_telegram_creds()
    if not token or not chat_id:
        print("No Telegram credentials configured -- aborting send.")
        return

    header = (f"AUGUST 2026 NEWS BACKFILL (one-time test of new FinBERT+override "
              f"verdict pipeline)\n{total_found} CRITICAL/HIGH stories found. "
              f"Live incremental monitoring resumes from today onward.\n")

    chunks = []
    current = header
    for line in lines:
        candidate = current + "\n\n" + line
        if len(candidate) > TELEGRAM_CHUNK_LIMIT:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)

    print(f"\nSending {len(chunks)} Telegram message(s)...")
    for i, chunk in enumerate(chunks):
        prefix = f"[Part {i+1}/{len(chunks)}]\n" if len(chunks) > 1 else ""
        ok = send_telegram_plain(token, chat_id, prefix + chunk)
        print(f"  part {i+1}/{len(chunks)}: {'sent' if ok else 'FAILED'}")


if __name__ == "__main__":
    qualifying, lines = main_run()
    chunk_and_send(lines, len(qualifying))
    print("\nDone.")
