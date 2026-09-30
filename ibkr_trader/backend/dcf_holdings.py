"""
Reverse DCF on the Inside Day Reversal holdings.

WHY REVERSE, NOT FORWARD: a forward DCF needs free cash flow (operating cash
flow MINUS capex) and net debt (debt MINUS cash). Polygon's standardized
statements carry neither capex nor a cash line -- only `long_term_debt` and
`net_cash_flow_from_investing_activities`, which mixes capex with acquisitions
and securities. A forward DCF built on that would be a guess wearing a model's
clothes. So this asks the question the data CAN answer:

    what operating-cash-flow growth does today's price already assume?

and compares it with the growth the company has actually delivered. If the
market demands 15%/yr from a business that has compounded 4%, that is
expensive regardless of anyone's target price.

STATED ASSUMPTIONS (change them and the answer changes -- that is the point):
    discount rate     9%    (a generic large-cap equity cost; not per-name beta)
    terminal growth   2.5%  (roughly long-run nominal GDP)
    horizon           10y explicit, then terminal
    cash-flow base    operating cash flow, TTM, WITH NO CAPEX DEDUCTED

That last line matters: excluding capex flatters every name, so the "implied
growth" printed here is LOWER than the truth. A capital-hungry business (RCL's
ships, RIVN's factories) is flattered most.

NOT SUITABLE FOR THIS METHOD, and skipped with a reason:
    banks/lenders (TFC, SOFI) -- for a bank, borrowing IS the raw material, so
        operating cash flow is not owner earnings; they need a dividend-discount
        or residual-income model
    pre-profit names (RIVN) -- with negative cash flow, 100% of the value sits
        in terminal assumptions, i.e. the model outputs whatever you assumed

    ../venv/bin/python dcf_holdings.py
"""
import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "scanner_config.json").read_text())
KEY = CFG["polygon_api_key"]
DISCOUNT, TERMINAL_G, YEARS = 0.09, 0.025, 10
SKIP = {"TFC": "bank -- operating cash flow is not owner earnings",
        "SOFI": "lender/fintech -- same problem as a bank",
        "RIVN": "negative cash flow -- a DCF would just echo the assumptions"}


def get(path, **params):
    """Polygon rate-limits hard on this plan (HTTP 429 with an empty body, which
    looked like 'no data for UAL/NKE' until it was checked) -- back off and retry."""
    for attempt in range(6):
        r = requests.get(f"https://api.polygon.io{path}", params={**params, "apiKey": KEY}, timeout=30)
        if r.status_code == 200:
            time.sleep(0.4)
            return r.json()
        if r.status_code == 429:
            time.sleep(2 + 3 * attempt)
            continue
        return {}
    print(f"   (rate-limited out on {path} {params.get('ticker','')})")
    return {}


def financials(ticker, timeframe="annual", limit=6):
    return get("/vX/reference/financials", ticker=ticker, timeframe=timeframe, limit=limit).get("results", [])


def val(rep, section, field):
    v = rep.get("financials", {}).get(section, {}).get(field, {}).get("value")
    return float(v) if isinstance(v, (int, float)) else None


def implied_growth(market_cap, cf0):
    """Bisect for the growth rate that makes the DCF equal today's market cap."""
    if not cf0 or cf0 <= 0 or not market_cap:
        return None
    def pv(g):
        total, cf = 0.0, cf0
        for t in range(1, YEARS + 1):
            cf *= (1 + g)
            total += cf / (1 + DISCOUNT) ** t
        terminal = cf * (1 + TERMINAL_G) / (DISCOUNT - TERMINAL_G)
        return total + terminal / (1 + DISCOUNT) ** YEARS
    lo, hi = -0.50, 0.60
    if pv(hi) < market_cap:
        return float("inf")
    if pv(lo) > market_cap:
        return float("-inf")
    for _ in range(200):
        mid = (lo + hi) / 2
        if pv(mid) < market_cap:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cagr(newest, oldest, years):
    if not newest or not oldest or oldest <= 0 or newest <= 0 or years <= 0:
        return None
    return (newest / oldest) ** (1 / years) - 1


def main():
    holdings = ["GE", "RCL", "UAL", "NKE", "TFC", "SOFI", "RIVN"]
    print(f"assumptions: discount {DISCOUNT:.0%}, terminal growth {TERMINAL_G:.1%}, "
          f"{YEARS}y explicit, base = TTM operating cash flow (NO capex deducted)\n")
    print(f"{'sym':<6}{'price':>8}{'mkt cap $B':>12}{'OCF TTM $B':>12}{'P/OCF':>7}"
          f"{'rev CAGR':>10}{'OCF CAGR':>10}{'implied g':>11}   verdict")
    for t in holdings:
        if t in SKIP:
            print(f"{t:<6}{'':>8}{'':>12}{'':>12}{'':>7}{'':>10}{'':>10}{'SKIPPED':>11}   {SKIP[t]}")
            continue
        det = get(f"/v3/reference/tickers/{t}").get("results", {})
        mcap = det.get("market_cap")
        ttm = financials(t, "ttm", 1)
        ann = financials(t, "annual", 6)
        if not ttm or not ann or not mcap:
            print(f"{t:<6}  no data (mcap={bool(mcap)}, ttm={bool(ttm)}, annual={len(ann)})")
            continue
        ocf = val(ttm[0], "cash_flow_statement", "net_cash_flow_from_operating_activities")
        rev_new = val(ann[0], "income_statement", "revenues")
        rev_old = val(ann[-1], "income_statement", "revenues")
        ocf_new = val(ann[0], "cash_flow_statement", "net_cash_flow_from_operating_activities")
        ocf_old = val(ann[-1], "cash_flow_statement", "net_cash_flow_from_operating_activities")
        yrs = len(ann) - 1
        g_rev, g_ocf = cagr(rev_new, rev_old, yrs), cagr(ocf_new, ocf_old, yrs)
        ig = implied_growth(mcap, ocf)
        prev = get(f"/v2/aggs/ticker/{t}/prev").get("results", [{}])
        price = prev[0].get("c") if prev else None
        pocf = mcap / ocf if ocf and ocf > 0 else None
        if ig is None or ig == float("inf"):
            verdict = "cannot solve (cash flow too low for this price)"
        elif g_ocf is None:
            verdict = f"needs {ig:.1%}/yr; no clean history to compare"
        elif ig <= (g_ocf or 0):
            verdict = f"needs {ig:.1%}/yr vs {g_ocf:.1%} delivered -> undemanding"
        elif ig <= (g_ocf or 0) + 0.05:
            verdict = f"needs {ig:.1%}/yr vs {g_ocf:.1%} delivered -> roughly fair"
        else:
            verdict = f"needs {ig:.1%}/yr vs {g_ocf:.1%} delivered -> demanding"
        f = lambda x, s="{:>10.1%}": s.format(x) if x is not None else f"{'n/a':>10}"
        print(f"{t:<6}{(price or 0):>8.2f}{mcap/1e9:>12.1f}{(ocf or 0)/1e9:>12.2f}"
              f"{(pocf or 0):>7.1f}{f(g_rev)}{f(g_ocf)}"
              f"{(f'{ig:.1%}' if ig not in (None, float('inf')) else 'n/a'):>11}   {verdict}")
    print(f"\ncaveat: capex is NOT deducted (Polygon has no capex field), so every implied-growth")
    print("figure above is too LOW -- most for capital-heavy names. Treat as a floor, not a fair value.")


if __name__ == "__main__":
    main()
