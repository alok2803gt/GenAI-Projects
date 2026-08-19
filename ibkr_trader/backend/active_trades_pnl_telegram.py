"""
Live P&L check across ALL genuinely active trades on this account (not
just one position) -- sends a single consolidated Telegram message.
Supersedes cohr_pnl_telegram.py (COHR-only) per CEO request 2026-08-13.

Covers, as of 2026-08-13:
  - COHR iron condor (Alpaca, 4 legs, expires 2026-08-14)
  - CSCO put credit spread (Alpaca, 2 legs, expires 2026-08-14)
  - AAPL Bull Put Spread (IBKR via Manual Trader, expires 2026-09-18)

Deliberately excludes: plain stock positions (BMBL/CORZ/F/NNDM -- not
option trades with a credit/thesis to track), AutoTrader's "untracked"
AAPL entries (same real position Manual Trader already reports, would
double-count), Day Trader/SPX 0DTE/FX Trader (no open positions as of
this build).

Usage: python active_trades_pnl_telegram.py
"""
import json

import requests

ALPACA_TRADES = {
    "COHR": {
        "entry_credit": 242.0,
        "legs": {
            "COHR260814C00400000": "short_call",
            "COHR260814C00410000": "long_call",
            "COHR260814P00310000": "long_put",
            "COHR260814P00320000": "short_put",
        },
        "expiry": "2026-08-14",
    },
    "CSCO": {
        "entry_credit": None,   # thin/mixed fill, no clean single credit figure -- report raw P&L only
        "legs": {
            "CSCO260814P00109000": "long_put",
            "CSCO260814P00114000": "short_put",
        },
        "expiry": "2026-08-14",
    },
}


def alpaca_section():
    resp = requests.get("http://localhost:8000/alpaca/positions", timeout=20)
    resp.raise_for_status()
    positions = {p["symbol"]: p for p in resp.json()["positions"]}

    lines = []
    for ticker, cfg in ALPACA_TRADES.items():
        legs_found = {sym: positions[sym] for sym in cfg["legs"] if sym in positions}
        if not legs_found:
            lines.append(f"{ticker}: no open legs found (closed or expired?)")
            continue
        total_pnl = sum(p["unrealized_pl"] for p in legs_found.values())
        header = f"{ticker} ({len(legs_found)}/{len(cfg['legs'])} legs, expires {cfg['expiry']})"
        if cfg["entry_credit"]:
            pct = total_pnl / cfg["entry_credit"] * 100
            header += f": ${total_pnl:+.2f} ({pct:.0f}% of ${cfg['entry_credit']:.0f} credit)"
        else:
            header += f": ${total_pnl:+.2f}"
        lines.append(header)
    return lines


def manual_trader_section():
    resp = requests.get("http://localhost:8000/manual-trader/status", timeout=20)
    resp.raise_for_status()
    positions = resp.json().get("positions", {})
    lines = []
    for pos_id, p in positions.items():
        lines.append(f"{p['ticker']} {p['strategy']} ({p['name']}, expires {p['expiry']}): "
                      f"${p['live_pnl']:+.2f}  (pt=${p.get('profit_target_usd')}, sl=${p.get('stop_loss_usd')})")
    if not lines:
        lines.append("Manual Trader: no open positions")
    return lines


def main():
    with open("scanner_config.json") as f:
        cfg = json.load(f)

    lines = ["Active trades -- live P&L check"]
    lines.append("")
    lines.append("Alpaca:")
    lines.extend(f"  {l}" for l in alpaca_section())
    lines.append("")
    lines.append("IBKR (Manual Trader):")
    lines.extend(f"  {l}" for l in manual_trader_section())

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
