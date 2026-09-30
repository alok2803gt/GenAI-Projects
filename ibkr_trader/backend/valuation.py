"""
Valuation for the Inside Day Reversal holdings -- filed data, per-name cost of
capital, and the right model for each business type.

WHAT THIS FIXES (each step came from an actual wrong answer earlier today):
  1. capex was missing -> RCL looked cheap at 9x "cash flow"; with its $5.2B of
     ship capex it is 55x FREE cash flow. Capex now comes from EDGAR.
  2. ONE year of capex is noisy for fleet builders, so free cash flow is also
     computed on NORMALIZED capex: the median capex/revenue over up to 10 filed
     years, applied to the latest revenue.
  3. a flat 9% discount rate flattered levered cyclicals. Now: cost of equity
     from CAPM with beta measured on 2 years of daily returns vs SPY, blended
     with an after-tax cost of debt at the company's own debt weighting.
  4. a DCF on a bank is meaningless (borrowing is raw material). TFC and SOFI
     get a RESIDUAL INCOME model instead: equity value = book value + the present
     value of profits above the cost of equity.
  5. a DCF on a cash burner just echoes the terminal assumption. RIVN gets a
     CASH RUNWAY calculation instead.

Everything is sourced: EDGAR XBRL companyfacts for financials, Alpaca daily
bars for beta, Polygon for market cap. One companyfacts call per company.

    ../venv/bin/python valuation.py            # the current IDR holdings
    ../venv/bin/python valuation.py GE NKE     # any tickers
"""
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "scanner_config.json").read_text())
UA = {"User-Agent": "Research Tool admin@example.com", "Accept-Encoding": "gzip, deflate"}
RISK_FREE, EQUITY_PREMIUM, COST_OF_DEBT, TAX = 0.042, 0.050, 0.055, 0.21
TERMINAL_G, YEARS = 0.025, 10
WACC_FLOOR, WACC_CAP = 0.06, 0.14
BANKS = {"TFC", "SOFI", "JPM", "BAC", "WFC", "C"}

FIELDS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements", "PaymentsToAcquirePropertyPlantAndEquipmentExcludingInterestCapitalized"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "sti": ["ShortTermInvestments", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "debt_cur": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    # net income to COMMON is the right numerator for a bank ROE (TFC has
    # preferred stock, and files under ProfitLoss rather than NetIncomeLoss);
    # these are definitional, so they are never merged across aliases
    "net_income": ["NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLoss", "ProfitLoss"],
}


def sec_facts(cik):
    for attempt in range(5):
        r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=UA, timeout=60)
        if r.status_code == 200:
            time.sleep(0.2)
            return r.json()
        time.sleep(1 + attempt)
    return None


def annual(facts, names, merge=True):
    """{fiscal-year-end: value} from 10-K FY facts, MERGED across concept aliases.

    Taking only the first matching concept left gaps: RTX files operating cash
    flow under one tag for 2009-2014 and another after the 2020 Raytheon merger,
    so the primary name alone gave 2009-2014 + 2022-2025 -- a hole exactly where
    the growth test needed data, and the name was wrongly rejected as having "no
    clean history". Earlier names win where they overlap; later names only fill
    years the earlier ones do not cover."""
    us = facts.get("facts", {}).get("us-gaap", {})
    out = {}
    for n in names:
        unit_data = us.get(n, {}).get("units", {})
        pts = unit_data.get("USD") or next((v for v in unit_data.values()), [])
        got = {}
        for p in pts:
            if p.get("form") in ("10-K", "10-K/A") and p.get("fp") == "FY" and p.get("end"):
                got[p["end"]] = p["val"]
        if not got:
            continue
        if not merge:
            return dict(sorted(got.items()))      # definitional fields: first match only
        for y, v in got.items():
            out.setdefault(y, v)                  # cash-flow series: fill gaps only
    return dict(sorted(out.items()))


def best_revenue(facts):
    """Pick the LARGEST plausible total-revenue series, not the first name that
    matches. Real bug: for RCL the concept "Revenues" carries a $2.3B segment
    figure while total revenue is $17.9B under
    RevenueFromContractWithCustomerExcludingAssessedTax -- a priority list
    silently chose the small one and made capex look like 130% of revenue."""
    us = facts.get("facts", {}).get("us-gaap", {})
    best, best_val = {}, 0.0
    for name, node in us.items():
        if "Revenue" not in name and "SalesRevenue" not in name:
            continue
        if any(w in name for w in ("Cost", "Deferred", "Remaining", "EquityMethod", "Segment",
                                   "Disaggregation", "PerformanceObligation")):
            continue
        pts = node.get("units", {}).get("USD", [])
        fy = {p["end"]: p["val"] for p in pts
              if p.get("form", "").startswith("10-K") and p.get("fp") == "FY" and p.get("end")}
        if len(fy) >= 3:
            latest = fy[max(fy)]
            if latest > best_val:
                best, best_val = dict(sorted(fy.items())), latest
    return best


def smoothed_cagr(series):
    """Growth from the average of the first 3 filed years to the average of the
    last 3. A plain endpoint CAGR made RCL look like +56%/yr because its early
    year was a COVID trough -- averaging both ends removes that."""
    if len(series) < 5:
        return None, None
    vals = [v for _, v in series]
    yrs = [y for y, _ in series]
    start, end = sum(vals[:3]) / 3, sum(vals[-3:]) / 3
    span = len(vals) - 3
    if start <= 0 or end <= 0 or span <= 0:
        return None, None
    return (end / start) ** (1 / span) - 1, f"{yrs[0][:4]}-{yrs[2][:4]} vs {yrs[-3][:4]}-{yrs[-1][:4]}"


def beta_vs_spy(ticker, days=500):
    H = {"APCA-API-KEY-ID": CFG["alpaca_api_key"], "APCA-API-SECRET-KEY": CFG["alpaca_secret_key"]}
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    def bars(sym):
        r = requests.get(f"https://data.alpaca.markets/v2/stocks/{sym}/bars",
                         params={"timeframe": "1Day", "start": start, "limit": 10000, "feed": "sip"},
                         headers=H, timeout=30)
        b = pd.DataFrame(r.json().get("bars", []))
        if b.empty:
            return None
        b["t"] = pd.to_datetime(b["t"]).dt.tz_convert("America/New_York").dt.date
        return b.set_index("t")["c"]
    s, m = bars(ticker), bars("SPY")
    if s is None or m is None:
        return None
    df = pd.concat([s.rename("s"), m.rename("m")], axis=1).dropna().pct_change().dropna()
    if len(df) < 120:
        return None
    cov = np.cov(df["s"], df["m"])
    return float(cov[0, 1] / cov[1, 1])


def market_cap(ticker):
    r = requests.get(f"https://api.polygon.io/v3/reference/tickers/{ticker}",
                     params={"apiKey": CFG["polygon_api_key"]}, timeout=30)
    if r.status_code != 200:
        return None
    return (r.json().get("results") or {}).get("market_cap")


def wacc(beta, mcap, debt):
    coe = RISK_FREE + (beta if beta is not None else 1.0) * EQUITY_PREMIUM
    if not mcap:
        return max(WACC_FLOOR, min(coe, WACC_CAP)), coe
    total = mcap + max(debt, 0)
    w = (mcap / total) * coe + (max(debt, 0) / total) * COST_OF_DEBT * (1 - TAX)
    return max(WACC_FLOOR, min(w, WACC_CAP)), coe


def implied_growth(target, cf0, r):
    if not cf0 or cf0 <= 0 or not target or target <= 0 or r <= TERMINAL_G:
        return None
    def pv(g):
        tot, cf = 0.0, cf0
        for t in range(1, YEARS + 1):
            cf *= (1 + g)
            tot += cf / (1 + r) ** t
        return tot + (cf * (1 + TERMINAL_G) / (r - TERMINAL_G)) / (1 + r) ** YEARS
    lo, hi = -0.60, 0.90
    if pv(hi) < target:
        return float("inf")
    if pv(lo) > target:
        return float("-inf")
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pv(mid) < target else (lo, mid)
    return (lo + hi) / 2


def residual_income(book, roe, coe, mcap):
    """Bank/lender value: book value plus the present value of profits above the
    cost of equity, fading to zero over YEARS. Compared with market cap."""
    if not book or book <= 0 or roe is None or not mcap:
        return None, None
    value, bv = book, book
    for t in range(1, YEARS + 1):
        fade = roe + (coe - roe) * (t / YEARS)        # excess return decays to zero
        ri = (fade - coe) * bv
        value += ri / (1 + coe) ** t
        bv *= (1 + fade * 0.5)                        # retain about half of earnings
    return value, mcap / value


def analyse(t, cik):
    facts = sec_facts(cik)
    if not facts:
        return {"ticker": t, "error": "no EDGAR facts"}
    DEFINITIONAL = {"equity", "net_income"}
    A = {k: annual(facts, v, merge=(k not in DEFINITIONAL)) for k, v in FIELDS.items()}
    A["revenue"] = best_revenue(facts) or A["revenue"]
    if not A["ocf"]:
        return {"ticker": t, "error": "no operating cash flow filed"}
    years = sorted(A["ocf"])[-10:]
    fy = years[-1]
    rev, ocf = A["revenue"].get(fy), A["ocf"][fy]
    capex_hist = [(y, A["capex"][y]) for y in years if y in A["capex"]]
    cap_ratio = None
    if capex_hist and A["revenue"]:
        ratios = [c / A["revenue"][y] for y, c in capex_hist if A["revenue"].get(y)]
        ratios = [x for x in ratios if 0 < x < 0.5]        # sanity: >50% of revenue means wrong series
        cap_ratio = statistics.median(ratios) if ratios else None
    capex_now = A["capex"].get(fy)
    fcf_now = ocf - (capex_now or 0)
    fcf_norm = (ocf - cap_ratio * rev) if (cap_ratio and rev) else None
    cash = (A["cash"].get(fy) or 0) + (A["sti"].get(fy) or 0)
    debt = (A["debt_lt"].get(fy) or 0) + (A["debt_cur"].get(fy) or 0)
    nd = debt - cash
    mcap = market_cap(t)
    beta = beta_vs_spy(t)
    r, coe = wacc(beta, mcap, debt)
    ev = (mcap + nd) if mcap else None
    fcf_hist = [(y, A["ocf"][y] - A["capex"][y]) for y in years if y in A["capex"]]
    g_hist, g_window = smoothed_cagr(fcf_hist)
    if g_hist is None and cap_ratio:
        # Not every company files capex under a concept that lines up with every
        # OCF year (RTX's 2020 merger split its history, so only 3 years matched
        # and the growth test came back "no clean history" for a business that
        # plainly has one). Rebuild a normalized FCF series instead: OCF minus
        # the median capex/revenue ratio applied to each year's own revenue.
        norm_hist = [(y, A["ocf"][y] - cap_ratio * A["revenue"][y])
                     for y in years if A["revenue"].get(y)]
        g_hist, g_window = smoothed_cagr(norm_hist)
        if g_window:
            g_window += " (normalized capex)"
    out = dict(ticker=t, fy=fy[:4], years=len(years), rev=rev, ocf=ocf, capex=capex_now,
               capex_pct_rev=cap_ratio, fcf=fcf_now, fcf_norm=fcf_norm, cash=cash, debt=debt,
               net_debt=nd, mcap=mcap, ev=ev, beta=beta, wacc=r, coe=coe, fcf_cagr=g_hist,
               fcf_window=g_window,
               book=A["equity"].get(fy), net_income=A["net_income"].get(fy))
    if t in BANKS:
        # MEDIAN of up to 5 years, not the mean of 3: TFC's 2023 goodwill
        # impairment (-$1.45B) dragged a 3-year mean ROE to 4.7% when the bank
        # earns 7-8% normally, which wrongly disqualified it.
        yrs = sorted(set(A["net_income"]) & set(A["equity"]))[-5:]
        roes = [A["net_income"][y] / A["equity"][y] for y in yrs if A["equity"][y] > 0]
        roe = statistics.median(roes) if roes else None
        val, ratio = residual_income(out["book"], roe, coe, mcap)
        out.update(model="residual income", roe=roe, fair_equity=val, price_to_fair=ratio)
    elif (fcf_norm or fcf_now) and (fcf_norm or fcf_now) > 0:
        base = fcf_norm if fcf_norm else fcf_now
        out.update(model="reverse DCF", base_fcf=base,
                   implied_g=implied_growth(ev, base, r), ev_fcf=(ev / base if ev else None))
    else:
        burn = -(fcf_norm if fcf_norm else fcf_now)
        out.update(model="cash runway", burn=burn,
                   runway_years=(cash / burn if burn and burn > 0 else None))
    return out


def main():
    tickers = sys.argv[1:]
    if not tickers:
        st = json.loads((HERE / "harami_trader_state.json").read_text())
        tickers = sorted({p["ticker"] for p in st["positions"].values() if p.get("phase") == "open"})
    ciks = {v["ticker"]: str(v["cik_str"]).zfill(10)
            for v in (requests.get("https://www.sec.gov/files/company_tickers.json",
                                   headers=UA, timeout=30).json()).values()}
    print(f"holdings: {', '.join(tickers)}")
    print(f"inputs: EDGAR 10-K facts (up to 10y), beta vs SPY on 2y daily, Polygon market cap")
    print(f"assumptions: risk-free {RISK_FREE:.1%}, equity premium {EQUITY_PREMIUM:.1%}, "
          f"pre-tax cost of debt {COST_OF_DEBT:.1%}, tax {TAX:.0%}, terminal growth {TERMINAL_G:.1%}\n")
    rows = [analyse(t, ciks[t]) for t in tickers if t in ciks]
    print(f"{'sym':<6}{'FY':>5}{'rev $B':>8}{'FCF $B':>8}{'norm FCF':>10}{'capex/rev':>11}"
          f"{'net debt $B':>12}{'beta':>6}{'WACC':>7}{'EV/FCF':>8}{'implied g':>10}{'FCF CAGR':>10}")
    for o in rows:
        if o.get("error"):
            print(f"{o['ticker']:<6}  {o['error']}")
            continue
        g = lambda k, s="{:>8.2f}", d=1e9: (s.format(o[k] / d) if o.get(k) is not None else f"{'n/a':>{len(s.format(0))}}")
        print(f"{o['ticker']:<6}{o['fy']:>5}{g('rev')}{g('fcf')}{g('fcf_norm', '{:>10.2f}')}"
              f"{(f'{o[chr(99)+chr(97)+chr(112)+chr(101)+chr(120)+chr(95)+chr(112)+chr(99)+chr(116)+chr(95)+chr(114)+chr(101)+chr(118)]:>10.1%}' if o.get('capex_pct_rev') else f'{chr(110)+chr(47)+chr(97):>10}')} "
              f"{g('net_debt', '{:>12.2f}')}{(f'{o[chr(98)+chr(101)+chr(116)+chr(97)]:>6.2f}' if o.get('beta') else f'{chr(110)+chr(47)+chr(97):>6}')}"
              f"{o['wacc']:>7.1%}"
              f"{(f'{o[chr(101)+chr(118)+chr(95)+chr(102)+chr(99)+chr(102)]:>8.1f}' if o.get('ev_fcf') else f'{chr(110)+chr(47)+chr(97):>8}')}"
              f"{(f'{o[chr(105)+chr(109)+chr(112)+chr(108)+chr(105)+chr(101)+chr(100)+chr(95)+chr(103)]:>10.1%}' if isinstance(o.get('implied_g'), float) and abs(o['implied_g']) < 5 else f'{chr(110)+chr(47)+chr(97):>10}')}"
              f"{(f'{o[chr(102)+chr(99)+chr(102)+chr(95)+chr(99)+chr(97)+chr(103)+chr(114)]:>10.1%}' if o.get('fcf_cagr') else f'{chr(110)+chr(47)+chr(97):>10}')}")
    print("\nverdict per holding:")
    for o in rows:
        if o.get("error"):
            continue
        t, model = o["ticker"], o.get("model")
        if model == "reverse DCF":
            ig, gh = o.get("implied_g"), o.get("fcf_cagr")
            if not isinstance(ig, float) or abs(ig) > 5:
                print(f"  {t:<5} cannot solve -- free cash flow too small for this price")
            elif gh is None:
                print(f"  {t:<5} price needs {ig:+.1%}/yr FCF growth at a {o['wacc']:.1%} WACC; "
                      f"history too short/erratic to judge")
            else:
                gap = ig - gh
                label = "UNDEMANDING" if gap < 0 else ("roughly fair" if gap < 0.05 else "DEMANDING")
                print(f"  {t:<5} price needs {ig:+.1%}/yr vs {gh:+.1%} delivered "
                      f"({o.get('fcf_window') or str(o['years']) + 'y'}) at {o['wacc']:.1%} WACC -> {label}")
        elif model == "residual income":
            pf, roe = o.get("price_to_fair"), o.get("roe")
            if pf:
                print(f"  {t:<5} lender: ROE {roe:.1%} vs {o['coe']:.1%} cost of equity, book "
                      f"${o['book']/1e9:.1f}B -> trades at {pf:.2f}x residual-income value "
                      f"({'cheap' if pf < 0.9 else 'rich' if pf > 1.1 else 'fair'})")
            else:
                print(f"  {t:<5} lender: insufficient book/ROE data")
        else:
            rw = o.get("runway_years")
            print(f"  {t:<5} burns ${o.get('burn', 0)/1e9:.2f}B/yr against ${o['cash']/1e9:.1f}B cash -> "
                  f"{f'{rw:.1f} years runway' if rw else 'runway unknown'}; no DCF is meaningful")
    (HERE / "valuation_output.json").write_text(json.dumps(rows, indent=1, default=str))
    print(f"\nsaved: valuation_output.json")


if __name__ == "__main__":
    main()
