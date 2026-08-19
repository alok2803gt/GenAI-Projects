"""
One-shot check: pull COHR's live 4-leg iron condor P&L from Alpaca and
send it to Telegram. Built 2026-08-13 to monitor the position through to
its 2026-08-14 expiry per the CEO's decision to hold rather than close
early (position was +$87/$242 credit at decision time, wide cushion both
sides, held specifically to let thin OTM legs expire worthless for free
rather than pay slippage closing them -- see oversight_log.jsonl).

Usage: python cohr_pnl_telegram.py
"""
import json

import requests

LEGS = {
    "COHR260814C00400000": "short_call",
    "COHR260814C00410000": "long_call",
    "COHR260814P00310000": "long_put",
    "COHR260814P00320000": "short_put",
}
ENTRY_CREDIT = 242.0


def main():
    with open("scanner_config.json") as f:
        cfg = json.load(f)

    resp = requests.get("http://localhost:8000/alpaca/positions", timeout=20)
    resp.raise_for_status()
    positions = resp.json()["positions"]

    legs_found = {p["symbol"]: p for p in positions if p["symbol"] in LEGS}
    if len(legs_found) != 4:
        missing = set(LEGS) - set(legs_found)
        text = f"⚠️ COHR P&L check: only found {len(legs_found)}/4 legs (missing {missing}) -- position may have partially closed or expired."
    else:
        total_pnl = sum(p["unrealized_pl"] for p in legs_found.values())
        pct_of_credit = total_pnl / ENTRY_CREDIT * 100
        lines = [f"COHR iron condor -- live P&L check"]
        for sym, label in LEGS.items():
            p = legs_found[sym]
            lines.append(f"  {label}: {p['current_price']:.2f} (entry {p['avg_entry_price']:.2f})  P&L ${p['unrealized_pl']:+.2f}")
        lines.append(f"Net: ${total_pnl:+.2f} ({pct_of_credit:.0f}% of ${ENTRY_CREDIT:.0f} credit)  |  expires 2026-08-14")
        text = "\n".join(lines)

    print(text)
    tg_resp = requests.post(
        f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
        json={"chat_id": cfg["telegram_chat_id"], "text": text},
        timeout=20,
    )
    print("Telegram status:", tg_resp.status_code)


if __name__ == "__main__":
    main()
