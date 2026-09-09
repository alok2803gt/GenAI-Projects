"""
Thin client for Unusual Whales' public dark pool API.

Real endpoints (verified against the live OpenAPI spec at
https://api.unusualwhales.com/api/openapi, 2026-08-23):
  GET /api/darkpool/recent               -- latest prints, market-wide
  GET /api/darkpool/{ticker}              -- prints for one ticker/day
  GET /api/darkpool/{ticker}/price-levels -- dark-pool vs regular volume
                                              concentration by price level

Auth: "Authorization: Bearer <key>" header. Base URL:
https://api.unusualwhales.com.

This module is read-only market data -- it never places an order, never
touches live config. Rate limits are not yet characterized empirically
(tier-dependent, not stated in the OpenAPI spec itself); callers should
treat 429s as a real possibility and back off, not assume unlimited calls.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

BASE_URL = "https://api.unusualwhales.com"
CONFIG_PATH = Path(__file__).parent / "scanner_config.json"


def _load_key() -> str:
    # Absolute path, not CWD-relative -- this module gets imported from other
    # directories (e.g. darkpool-levels-calculator lives under ~/.claude/skills),
    # where a bare "scanner_config.json" open() would silently fail.
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    key = cfg.get("unusual_whales_api_key", "")
    if not key:
        raise RuntimeError("unusual_whales_api_key is empty in scanner_config.json")
    return key


class UnusualWhalesClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
        self.api_key = api_key or _load_key()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })

    def _get(self, path: str, params: Optional[dict] = None, retries: int = 3) -> dict:
        url = f"{BASE_URL}{path}"
        last_err = None
        for attempt in range(retries):
            r = self.session.get(url, params=params or {}, timeout=self.timeout)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 2 * (attempt + 1)))
                time.sleep(wait)
                last_err = f"429 rate-limited (waited {wait}s)"
                continue
            if r.status_code == 401:
                raise RuntimeError("401 Unauthorized -- check unusual_whales_api_key in scanner_config.json")
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Gave up after {retries} attempts: {last_err}")

    def recent_trades(self, limit: int = 100, min_premium: Optional[float] = None,
                       min_size: Optional[int] = None, order_by: str = "premium",
                       order: str = "desc") -> list[dict]:
        """Latest dark pool prints market-wide. limit: default 100, max 200."""
        params = {"limit": min(limit, 200), "order_by": order_by, "order": order}
        if min_premium is not None:
            params["min_premium"] = min_premium
        if min_size is not None:
            params["min_size"] = min_size
        data = self._get("/api/darkpool/recent", params)
        return data.get("data", data) if isinstance(data, dict) else data

    def ticker_trades(self, ticker: str, date: Optional[str] = None, limit: int = 500,
                       min_premium: Optional[float] = None, min_size: Optional[int] = None,
                       order_by: str = "executed_at", order: str = "desc") -> list[dict]:
        """Dark pool prints for one ticker on one day (default: current/last trading day)."""
        params = {"limit": min(limit, 500), "order_by": order_by, "order": order}
        if date:
            params["date"] = date
        if min_premium is not None:
            params["min_premium"] = min_premium
        if min_size is not None:
            params["min_size"] = min_size
        data = self._get(f"/api/darkpool/{ticker}", params)
        return data.get("data", data) if isinstance(data, dict) else data

    def price_levels(self, ticker: str, date: Optional[str] = None) -> dict:
        """Dark-pool vs regular volume concentration by price level (volume-profile-style)."""
        params = {"date": date} if date else {}
        return self._get(f"/api/darkpool/{ticker}/price-levels", params)

    def flow_alerts(self, ticker: str, limit: int = 100, **filters) -> list[dict]:
        """Real rule-based options-flow alerts for one ticker (e.g.
        RepeatedHits) -- each carries REAL ask-side/bid-side premium split
        (total_ask_side_prem/total_bid_side_prem, genuine trade-side
        classification, not an approximation), call/put type, has_sweep,
        all_opening_trades, volume_oi_ratio. Added 2026-08-30 for a real
        bullish/bearish options-flow read. Endpoint: /api/option-trades/flow-alerts.
        """
        params = {"ticker_symbol": ticker, "limit": min(limit, 200)}
        params.update({k: v for k, v in filters.items() if v is not None})
        data = self._get("/api/option-trades/flow-alerts", params)
        return data.get("data", data) if isinstance(data, dict) else data

    def lit_flow(self, ticker: str, limit: int = 200) -> list[dict]:
        """Real lit-market (visible exchange, NOT dark pool) trade tape for
        one ticker -- includes real NBBO bid/ask per trade, so an aggressor
        side (buy vs sell) can be inferred the standard way (price vs mid).
        Added 2026-08-30 as a stock-tape cross-check for options-flow reads --
        more reliably available than this account's own tape_prints table,
        which only has data when a live WS tape session happened to be open
        for that specific ticker. Endpoint: /api/lit-flow/{ticker}.
        """
        params = {"limit": min(limit, 500)}
        data = self._get(f"/api/lit-flow/{ticker}", params)
        return data.get("data", data) if isinstance(data, dict) else data


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quick manual smoke test of the UW dark pool client")
    ap.add_argument("--ticker", default=None, help="If given, fetch this ticker's trades + price levels instead of market-wide recent")
    args = ap.parse_args()

    c = UnusualWhalesClient()
    if args.ticker:
        trades = c.ticker_trades(args.ticker, limit=10, order_by="premium")
        print(f"Top 10 {args.ticker} darkpool trades by premium:")
        print(json.dumps(trades, indent=2)[:3000])
        levels = c.price_levels(args.ticker)
        print(f"\n{args.ticker} price levels:")
        print(json.dumps(levels, indent=2)[:2000])
    else:
        trades = c.recent_trades(limit=10, order_by="premium")
        print("Top 10 market-wide darkpool trades by premium:")
        print(json.dumps(trades, indent=2)[:3000])
