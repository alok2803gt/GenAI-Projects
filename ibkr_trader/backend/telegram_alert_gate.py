"""Shared Telegram alert category gate.

Every strategy/scanner/babysitter in this codebase (main.py's in-process
strategies AND ~25 standalone OS processes) already reads Telegram
credentials fresh from scanner_config.json on every send -- see
main.py's _load_telegram_creds() and its near-identical copies across
the standalone scripts. This module adds a second, independent gate on
top of that: a per-CATEGORY on/off switch, read fresh from
telegram_alerts_config.json on every call, so a toggle flipped in the
admin panel takes effect on the very next alert in every process, no
restart needed anywhere.

Fails OPEN (alert allowed) if the config file is missing, unreadable, or
the category isn't recognized -- an admin-panel bug or a not-yet-listed
category should never silently kill a live-trading alert. The existing
credential-based guard remains the real "Telegram is off entirely"
switch; this one is purely about which categories a user wants routed
through it.
"""
import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telegram_alerts_config.json")

# category key -> (display label, group). Order here is the order the
# admin panel renders them in, grouped by `group`. Keep this the single
# source of truth for the full alert taxonomy -- both main.py's endpoints
# and the frontend pull the label/group from here (via /telegram-alerts/config).
ALERT_CATEGORIES: dict[str, tuple[str, str]] = {
    # Core auto-traders
    "day_trader":          ("Day Trader (agent) entries/exits", "Core Auto-Traders"),
    "spy_weekly_condor":   ("SPY Weekly Condor agent", "Core Auto-Traders"),
    "spx_0dte":            ("SPX 0DTE alerts", "Core Auto-Traders"),
    "evc":                 ("Earnings Vol Crush (EVC) alerts", "Core Auto-Traders"),
    "manual_trader":       ("Manual Trader alerts", "Core Auto-Traders"),
    "fx_trader":           ("FX Trader alerts", "Core Auto-Traders"),
    "safe_income":         ("Safe Income Trader (CSP/LEAP)", "Core Auto-Traders"),
    # Butterfly / condor babysitters + standalone position traders
    "butterfly_babysitters": ("SPY/QQQ/IWM 0DTE butterfly babysitters", "Babysitters & Standalone Traders"),
    "condor_babysitters":    ("PDD/INTU/GOOG condor + residual-put babysitters", "Babysitters & Standalone Traders"),
    "earnings_butterfly":    ("Earnings butterfly babysitter", "Babysitters & Standalone Traders"),
    "ko_put_spread":         ("KO put spread trader", "Babysitters & Standalone Traders"),
    "harami_trader":         ("Harami daily trader", "Babysitters & Standalone Traders"),
    # Signal feeds & scanners
    "breakout_scanner":    ("Breakout scanner signal fires", "Signal Feeds & Scanners"),
    "daytrader_scanner":   ("Day Trader premarket scanner", "Signal Feeds & Scanners"),
    "ashley_signals":      ("Ashley signal-follow (monitor + executor)", "Signal Feeds & Scanners"),
    "chartexpert":         ("ChartExpert auto-trader", "Signal Feeds & Scanners"),
    # Research desk
    "research_desk":       ("Darkpool / sector-catalyst / NFLX / shadow-filter research monitors", "Research Desk"),
    # Oversight & risk
    "risk_gate":           ("Risk Monitor violations", "Oversight & Risk"),
    "news_monitor":        ("News Monitor verdicts", "Oversight & Risk"),
    "reconciliation":      ("Position reconciliation auto-corrects", "Oversight & Risk"),
    "system_health":       ("Morning/hourly oversight + CFO weekly checks", "Oversight & Risk"),
    # Misc chart/pattern alerts
    "chart_price_alerts":       ("User-set chart price alerts", "Misc Chart Alerts"),
    "harami_1m_alerts":         ("1-min bullish harami detection", "Misc Chart Alerts"),
    "unusual_options_activity": ("Unusual options activity flags", "Misc Chart Alerts"),
    "red_day_alert":            ("Red-day premium-on-sale alert (SPY <= -1.5%)", "Misc Chart Alerts"),
}


def _read_raw() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def alert_enabled(category: str) -> bool:
    """True unless this category is explicitly set to false in
    telegram_alerts_config.json. Fails open on any read error or unknown
    category -- call this as the very first line of a send function."""
    return bool(_read_raw().get(category, True))


def get_alert_config() -> dict:
    """Every known category present, defaulting True, for the admin panel."""
    raw = _read_raw()
    return {cat: bool(raw.get(cat, True)) for cat in ALERT_CATEGORIES}


def set_alert_config(updates: dict) -> dict:
    """Merge `updates` (category -> bool) into the persisted config and
    return the resulting full config. Unknown keys are ignored so a
    stale frontend can't inject arbitrary categories."""
    cfg = get_alert_config()
    for k, v in updates.items():
        if k in ALERT_CATEGORIES:
            cfg[k] = bool(v)
    tmp_path = _CONFIG_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp_path, _CONFIG_PATH)
    return cfg
