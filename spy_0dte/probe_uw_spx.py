"""Probe Unusual Whales + Polygon SPX coverage. Prints status/shape only, never keys."""
import json, sys
from pathlib import Path
import requests
cfg = json.loads(Path(sys.argv[1]).read_text())
uw = {"Authorization": f"Bearer {cfg['unusual_whales_api_key']}", "Accept": "application/json"}
B = "https://api.unusualwhales.com"
tests = [
    ("greek-exposure daily (history)", f"{B}/api/stock/SPY/greek-exposure", {"timeframe": "5y"}),
    ("greek-exposure by date", f"{B}/api/stock/SPY/greek-exposure", {"date": "2024-06-03"}),
    ("spot exposures intraday (past date)", f"{B}/api/stock/SPY/spot-exposures", {"date": "2024-06-03"}),
    ("spot exposures by strike (past)", f"{B}/api/stock/SPY/spot-exposures/strike", {"date": "2024-06-03"}),
    ("net premium ticks (past)", f"{B}/api/stock/SPY/net-prem-ticks", {"date": "2024-06-03"}),
    ("option volume/OI daily", f"{B}/api/stock/SPY/options-volume", {"limit": 500}),
    ("iv term structure (past)", f"{B}/api/stock/SPY/volatility/term-structure", {"date": "2024-06-03"}),
    ("realized vol", f"{B}/api/stock/SPY/volatility/realized", {"timeframe": "5y"}),
    ("flow alerts", f"{B}/api/option-trades/flow-alerts", {"ticker_symbol": "SPY", "limit": 5}),
    ("market tide (past)", f"{B}/api/market/market-tide", {"date": "2024-06-03"}),
    ("SPX greek exposure", f"{B}/api/stock/SPX/greek-exposure", {"timeframe": "5y"}),
]
for name, url, params in tests:
    try:
        r = requests.get(url, headers=uw, params=params, timeout=30)
        body = r.json() if "json" in r.headers.get("content-type", "") else {}
        data = body.get("data", body) if isinstance(body, dict) else body
        n = len(data) if isinstance(data, list) else "-"
        first = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else "")
        keys = list(first.keys())[:12] if isinstance(first, dict) else str(first)[:120]
        span = ""
        if isinstance(data, list) and data and isinstance(data[0], dict):
            dk = next((k for k in ("date", "timestamp", "tape_time", "time", "start_time", "created_at") if k in data[0]), None)
            if dk:
                span = f" range {min(str(d[dk]) for d in data)[:19]} .. {max(str(d[dk]) for d in data)[:19]}"
        print(f"UW {name:<36} HTTP {r.status_code} rows {n}{span}\n     fields {keys}")
    except Exception as e:
        print(f"UW {name:<36} ERROR {type(e).__name__} {str(e)[:120]}")
pk = cfg["polygon_api_key"]
for tkr in ("O:SPXW250303C05900000", "I:SPX"):
    r = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{tkr}/range/1/minute/2025-03-03/2025-03-03",
                     params={"limit": 3, "apiKey": pk}, timeout=30)
    print(f"Polygon {tkr}: HTTP {r.status_code} {r.text[:160]}")
