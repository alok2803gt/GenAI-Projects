"""
Reverse DCF on the Inside Day Reversal holdings, using SEC EDGAR filings
for the fields Polygon's standardized statements do not carry.

Polygon gave operating cash flow but NO capex and NO cash line, so the first
pass could not compute free cash flow or net debt -- it flattered capital-heavy
names (RCL's ships, UAL's aircraft) by pretending capex was free. EDGAR's XBRL
"companyconcept" API is free, official, and has all of it:

    OCF        NetCashProvidedByUsedInOperatingActivities
               (+ ...ContinuingOperations fallback)
    capex      PaymentsToAcquirePropertyPlantAndEquipment
               (+ PaymentsToAcquireProductiveAssets fallback)
    cash       CashAndCashEquivalentsAtCarryingValue
               (+ ShortTermInvestments when reported)
    debt       LongTermDebtNoncurrent + LongTermDebtCurrent
               (falls back to LongTermDebt / DebtCurrent)
    shares     dei:EntityCommonStockSharesOutstanding

FCF = OCF - capex. Enterprise value = market cap + net debt, so the DCF
discounts cash flows available to ALL capital and is then compared with EV --
the consistent pairing (comparing FCF against equity value only would
understate how much leverage matters for UAL and RCL).

Output: the FCF growth today's price implies, against what each company has
actually delivered. Still not a fair-value estimate -- the discount rate is a
single generic 9%, not a per-name WACC -- but every input is now a filed number.

    ../venv/bin/python dcf_sec.py
"""
import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
POLY = json.loads((HERE / "scanner_config.json").read_text())["polygon_api_key"]
UA = {"User-Agent": "Research Tool admin@example.com", "Accept-Encoding": "gzip, deflate"}
DISCOUNT, TERMINAL_G, YEARS = 0.09, 0.025, 10
HOLDINGS = ["GE", "RCL", "UAL", "NKE", "TFC", "SOFI", "RIVN"]
BANKS = {"TFC", "SOFI"}          # FCF is not owner earnings for a lender

CONCEPTS = {
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "sti": ["ShortTermInvestments", "OtherShortTermInvestments"],
    "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "debt_cur": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
}


def sec_get(url):
    for attempt in range(5):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 200:
            time.sleep(0.2)                     # SEC asks for <= 10 requests/second
            return r.json()
        if r.status_code in (429, 503):
            time.sleep(1 + attempt)
            continue
        return None
    return None


def cik_map():
    d = sec_get("https://www.sec.gov/files/company_tickers.json") or {}
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in d.values()}


def concept_series(cik, names, unit="USD"):
    """Annual (FY, 10-K) values for the first concept that exists, newest last."""
    for name in names:
        d = sec_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{name}.json")
        if not d:
            continue
        pts = d.get("units", {}).get(unit, [])
        fy = {}
        for p in pts:
            if p.get("form") in ("10-K", "10-K/A") and p.get("fp") == "FY" and p.get("end"):
                fy[p["end"]] = p["val"]         # later filings overwrite restated years
        if fy:
            return name, sorted(fy.items())
    return None, []


def latest_quarterly_ttm(cik, names):
    """Sum of the last 4 quarterly values, for a TTM figure."""
    for name in names:
        d = sec_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{name}.json")
        if not d:
            continue
        pts = [p for p in d.get("units", {}).get("USD", [])
               if p.get("form", "").startswith("10-Q") and p.get("start") and p.get("end")]
        pts = [p for p in pts if 80 <= (
            __import__("datetime").date.fromisoformat(p["end"]) -
            __import__("datetime").date.fromisoformat(p["start"])).days <= 100]
        if len(pts) >= 4:
            pts.sort(key=lambda p: p["end"])
            return sum(p["val"] for p in pts[-4:])
    return None


def implied_growth(target_value, cf0):
    if not cf0 or cf0 <= 0 or not target_value or target_value <= 0:
        return None
    def pv(g):
        total, cf = 0.0, cf0
        for t in range(1, YEARS + 1):
            cf *= (1 + g)
            total += cf / (1 + DISCOUNT) ** t
        return total + (cf * (1 + TERMINAL_G) / (DISCOUNT - TERMINAL_G)) / (1 + DISCOUNT) ** YEARS
    lo, hi = -0.60, 0.80
    if pv(hi) < target_value:
        return float("inf")
    if pv(lo) > target_value:
        return float("-inf")
    for _ in range(200):
        mid = (lo + hi) / 2
        if pv(mid) < target_value:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cagr(new, old, yrs):
    if not new or not old or old <= 0 or new <= 0 or yrs <= 0:
        return None
    return (new / old) ** (1 / yrs) - 1


def main():
    ciks = cik_map()
    print(f"SEC EDGAR filings | discount {DISCOUNT:.0%}, terminal {TERMINAL_G:.1%}, {YEARS}y explicit")
    print("FCF = operating cash flow - capex (both as filed); EV = market cap + net debt\n")
    hdr = (f"{'sym':<6}{'FY':>6}{'OCF $B':>9}{'capex $B':>10}{'FCF $B':>9}{'net debt $B':>13}"
           f"{'EV $B':>8}{'EV/FCF':>8}{'FCF CAGR':>10}{'implied g':>11}")
    print(hdr)
    rows = []
    for t in HOLDINGS:
        cik = ciks.get(t)
        if not cik:
            print(f"{t:<6}  no CIK found")
            continue
        _, ocf = concept_series(cik, CONCEPTS["ocf"])
        _, capex = concept_series(cik, CONCEPTS["capex"])
        _, cash = concept_series(cik, CONCEPTS["cash"])
        _, sti = concept_series(cik, CONCEPTS["sti"])
        _, dlt = concept_series(cik, CONCEPTS["debt_lt"])
        _, dcur = concept_series(cik, CONCEPTS["debt_cur"])
        if not ocf:
            print(f"{t:<6}  no operating cash flow in EDGAR")
            continue
        fy_end, ocf_v = ocf[-1]
        cap_v = dict(capex).get(fy_end)
        if cap_v is None and capex:
            cap_v = capex[-1][1]
        fcf = ocf_v - (cap_v or 0)
        cash_v = (dict(cash).get(fy_end) or (cash[-1][1] if cash else 0)) + \
                 (dict(sti).get(fy_end) or 0)
        debt_v = (dict(dlt).get(fy_end) or (dlt[-1][1] if dlt else 0)) + \
                 (dict(dcur).get(fy_end) or 0)
        net_debt = debt_v - cash_v
        det = requests.get(f"https://api.polygon.io/v3/reference/tickers/{t}",
                           params={"apiKey": POLY}, timeout=30)
        mcap = (det.json().get("results", {}) or {}).get("market_cap") if det.status_code == 200 else None
        ev = (mcap + net_debt) if mcap else None
        # FCF history for the growth comparison
        hist = []
        cap_by_year = dict(capex)
        for end, o in ocf[-6:]:
            c = cap_by_year.get(end)
            if c is not None:
                hist.append((end, o - c))
        g_fcf = cagr(hist[-1][1], hist[0][1], len(hist) - 1) if len(hist) >= 3 else None
        ig = implied_growth(ev, fcf) if (ev and t not in BANKS) else None
        f = lambda v, s="{:>9.2f}": s.format(v / 1e9) if v is not None else f"{'n/a':>9}"
        print(f"{t:<6}{fy_end[:4]:>6}{f(ocf_v)}{f(cap_v, '{:>10.2f}')}{f(fcf)}{f(net_debt, '{:>13.2f}')}"
              f"{f(ev, '{:>8.1f}')}"
              f"{(f'{ev/fcf:>8.1f}' if ev and fcf and fcf > 0 else f'{chr(110)+chr(47)+chr(97):>8}')}"
              f"{(f'{g_fcf:>10.1%}' if g_fcf is not None else f'{chr(110)+chr(47)+chr(97):>10}')}"
              f"{(f'{ig:>11.1%}' if ig not in (None, float('inf'), float('-inf')) else f'{chr(110)+chr(47)+chr(97):>11}')}")
        rows.append((t, fcf, net_debt, ev, g_fcf, ig))

    print("\nverdicts:")
    for t, fcf, nd, ev, g, ig in rows:
        if t in BANKS:
            print(f"  {t:<5} skipped -- lender, free cash flow is not owner earnings")
        elif fcf is not None and fcf <= 0:
            print(f"  {t:<5} negative free cash flow (${fcf/1e9:.2f}B) -- no DCF is meaningful")
        elif ig is None:
            print(f"  {t:<5} could not solve")
        elif g is None:
            print(f"  {t:<5} price implies {ig:+.1%}/yr FCF growth; too little history to compare")
        elif ig < g:
            print(f"  {t:<5} price implies {ig:+.1%}/yr vs {g:+.1%} delivered -> UNDEMANDING")
        elif ig < g + 0.05:
            print(f"  {t:<5} price implies {ig:+.1%}/yr vs {g:+.1%} delivered -> roughly fair")
        else:
            print(f"  {t:<5} price implies {ig:+.1%}/yr vs {g:+.1%} delivered -> DEMANDING")


if __name__ == "__main__":
    main()
