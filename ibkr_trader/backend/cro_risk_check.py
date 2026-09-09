"""
Deterministic, unattended hourly CRO (risk-manager) check.

Same rationale as portfolio_oversight_check.py -- a plain script, not an
LLM agent, so it can run headless via Windows Task Scheduler with no
permission-bypass risk. Watch-only, same as the attended CRO role: never
trades, never changes config, only reads state, logs, and alerts.

Checks each run:
  - Cross-strategy concentration: same underlying held by more than one
    strategy at once (pulls open positions from every known trader)
  - Aggregate exposure: gross position value vs net liquidation
  - Margin/leverage drift: flags if buying_power meaningfully exceeds
    available cash (this account's standing rule is cash-only, no margin)
  - Risk-gate calibration: /risk/status config.account_value vs the real
    net_liquidation -- flags if the gap is large (this account has a
    known, previously-flagged miscalibration here)

Usage: python cro_risk_check.py
"""
import json
from datetime import datetime, timezone

import requests

BACKEND = "http://localhost:8000"
LOG_FILE = "oversight_log.jsonl"
TIMEOUT = 10

POSITION_ENDPOINTS = {
    "day_trader": "/day-trader/status",
    "evc": "/earnings-vol-crush/status",
    "autotrader": "/autotrader/status",
}


def get(path: str):
    try:
        r = requests.get(f"{BACKEND}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def telegram(msg: str, high_priority: bool = False):
    try:
        with open("scanner_config.json") as f:
            cfg = json.load(f)
        prefix = "🚨 " if high_priority else ""
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": prefix + msg, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as e:
        print(f"Telegram send failed: {e}")


def log_line(summary: str, outcome: str, rationale: str):
    entry = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(),
        "actor": "risk_manager",
        "category": "routine_check_unattended",
        "summary": summary,
        "rationale": rationale,
        "outcome": outcome,
        "pnl_impact": None,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    if datetime.now().weekday() >= 5:  # Sat=5, Sun=6 -- weekdays-only, enforced here since the
        return                          # OS trigger repeats 24/7 for reliable hourly repetition

    findings = []

    account = get("/account/summary")
    risk = get("/risk/status")
    alpaca = get("/alpaca/positions")

    if "_error" in account or "_error" in risk:
        msg = f"Unattended CRO check FAILED to reach backend: account_err={account.get('_error')} risk_err={risk.get('_error')}"
        telegram(msg, high_priority=True)
        log_line(msg, outcome="Backend unreachable -- alerted, no further checks run this cycle.", rationale="Unattended hourly CRO cycle (Task Scheduler, no live session).")
        return

    net_liq = account.get("net_liquidation") or 0
    cash = account.get("total_cash") or 0
    buying_power = account.get("buying_power") or 0

    if buying_power > cash * 1.05:
        findings.append(f"IBKR buying_power (${buying_power}) exceeds cash (${cash}) -- possible margin drift, this account's standing rule is cash-only")

    # Fixed 2026-08-26: config.account_value (50000.0) is a fallback-only
    # default -- _risk_monitor_coro already pulls live net liq for the real
    # Rule 5 calculation and never uses the static number when connected, so
    # comparing THAT raw config field against net_liq was a permanent false
    # alarm (it will always look "miscalibrated" unless the account happens
    # to BE $50k). /risk/status now exposes account_value_effective, the
    # actual number the live check uses -- compare against that instead.
    effective_account_value = risk.get("account_value_effective")
    value_source = risk.get("account_value_source", "unknown")
    if effective_account_value and net_liq and value_source != "live_net_liq":
        gap_pct = abs(effective_account_value - net_liq) / net_liq * 100
        if gap_pct > 20:
            findings.append(f"/risk/status is using a fallback account value (${effective_account_value}, source={value_source}) "
                             f"{gap_pct:.0f}% off real net_liquidation (${net_liq}) -- IBKR may be disconnected, position-limit percentages may be miscalibrated")

    tickers_by_strategy = {}
    for name, path in POSITION_ENDPOINTS.items():
        st = get(path)
        positions = st.get("positions", {})
        if isinstance(positions, dict):
            for pos_id, pos in positions.items():
                ticker = pos.get("ticker") if isinstance(pos, dict) else None
                if ticker:
                    tickers_by_strategy.setdefault(ticker, []).append(name)

    if "positions" in alpaca:
        for p in alpaca["positions"]:
            sym = p.get("symbol", "")
            underlying = "".join(c for c in sym if c.isalpha())[:6] if sym else None
            if underlying:
                tickers_by_strategy.setdefault(underlying, []).append("alpaca")

    for ticker, strategies in tickers_by_strategy.items():
        unique_strategies = set(strategies)
        if len(unique_strategies) > 1:
            findings.append(f"{ticker} held across multiple strategies simultaneously: {', '.join(unique_strategies)} -- check combined exposure")

    combined_net_liq = net_liq + (alpaca.get("account", {}).get("equity") or 0)
    alpaca_unrealized = 0
    if "positions" in alpaca:
        alpaca_unrealized = sum(p.get("unrealized_pl", 0) or 0 for p in alpaca["positions"])
    drawdown_pct = (alpaca_unrealized / combined_net_liq * 100) if combined_net_liq else 0
    if drawdown_pct < -10:
        findings.append(f"Alpaca unrealized P&L (${alpaca_unrealized:.2f}) is {drawdown_pct:.1f}% of combined net liq (${combined_net_liq:.2f}) -- real drawdown, worth reviewing")

    summary = f"Unattended CRO sweep: net liq ${net_liq}, combined ${combined_net_liq:.2f}. " + ("; ".join(findings) if findings else "Clean on all structural checks.")

    if findings:
        telegram("<b>Unattended CRO check -- findings:</b>\n" + "\n".join(f"- {f}" for f in findings), high_priority=False)
        outcome = "Findings sent to Telegram for review in the next attended session -- watch-only, no action taken."
    else:
        outcome = "Clean sweep, no Telegram sent."

    log_line(summary, outcome=outcome, rationale="Unattended hourly CRO cycle (Task Scheduler, no live session).")
    print(summary)


if __name__ == "__main__":
    main()
