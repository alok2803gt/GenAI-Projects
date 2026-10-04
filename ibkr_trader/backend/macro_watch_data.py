"""Data layer for macro_watch (v2).

Every fetcher merges what it gets into a local cache (macro_watch_cache/, gitignored) and returns the
cached {date: value} series. That gives the percentile scoring the multi-year history it needs even
though FRED times out intermittently, UW price bars only reach back 1 year, UW gamma 250 days, and UW
options net premium has no history endpoint at all (it accumulates one day at a time here).
Fetchers never raise: on failure they fall back to the cache and record the problem in STATUS, which
the report prints, so a degraded source is visible instead of silently missing.
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "macro_watch_cache"
CACHE.mkdir(exist_ok=True)
CFG = json.loads((HERE / "scanner_config.json").read_text())
UA = {"User-Agent": "Mozilla/5.0 (macro_watch)"}
TODAY = date.today()
STATUS: dict[str, str] = {}

TSY_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{yr}/all"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SSGA_URL = "https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
# SSGA weights as of 2026-10-01 -- used only if the holdings download fails and nothing is cached.
DEFAULT_TOP = [("NVDA", 8.45), ("AAPL", 7.30), ("MSFT", 5.76), ("GOOGL", 5.41), ("AMZN", 3.69), ("AVGO", 2.47),
               ("META", 2.42), ("MU", 1.88), ("AMD", 1.52), ("TSLA", 1.51)]


# ---------------------------------------------------------------- plumbing
def load(name, default=None):
    try:
        return json.loads((CACHE / f"{name}.json").read_text())
    except Exception:
        return default


def save(name, obj):
    p = CACHE / f"{name}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(p)


def merge(name, new: dict) -> dict:
    s = load(name, {}) or {}
    s.update(new)
    save(name, s)
    return s


def http(url, params=None, timeout=25, tries=3, as_json=True):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 429:
                last = "HTTP 429"; time.sleep(15 * (i + 1)); continue
            r.raise_for_status()
            return r.json() if as_json else r.text
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(3)
    raise RuntimeError(last)


def _fallback(key, name, err):
    s = load(name, {}) or {}
    STATUS[key] = f"FAILED ({err}); using cache through {max(s) if s else 'none'}"
    return s


# ---------------------------------------------------------------- rates
def treasury() -> dict:
    """{date: {y3m, y2, y10, y30}} -- US Treasury daily par curve (same evening), 3 calendar years.
    Past years are cached permanently; the current year is refetched each run. Massive is the fallback."""
    rows = {}
    for yr in range(TODAY.year - 2, TODAY.year + 1):
        name = f"tsy_{yr}"
        if yr < TODAY.year and load(name):
            rows.update(load(name)); continue
        try:
            txt = http(TSY_URL.format(yr=yr), {"type": "daily_treasury_yield_curve", "field_tdr_date_value": yr,
                                               "_format": "csv"}, timeout=30, as_json=False)
            got = {}
            for x in csv.DictReader(io.StringIO(txt)):
                try:                                             # per-row guard: one bad row must not drop the year
                    m, d, y = x["Date"].split("/")
                    got[f"{y}-{m}-{d}"] = {"y3m": float(x["3 Mo"]), "y2": float(x["2 Yr"]), "y10": float(x["10 Yr"]),
                                           "y30": float(x["30 Yr"])}
                except (KeyError, ValueError):
                    continue
            if not got:
                raise RuntimeError("empty CSV")
            save(name, got); rows.update(got)
        except Exception as e:
            rows.update(_fallback("treasury", name, e))
    if len(rows) < 300:                                          # Treasury down and no cache: Massive (lags ~1 day)
        try:
            pk = CFG["polygon_api_key"]; url = "https://api.polygon.io/fed/v1/treasury-yields"
            prm = {"date.gte": str(TODAY - timedelta(days=1100)), "sort": "date.asc", "limit": 50000, "apiKey": pk}
            while url:
                j = http(url, prm, timeout=60, tries=5)
                for r in j.get("results", []):
                    if r.get("yield_10_year") is not None:
                        rows[r["date"]] = {"y3m": r.get("yield_3_month"), "y2": r.get("yield_2_year"),
                                           "y10": r["yield_10_year"], "y30": r.get("yield_30_year")}
                url, prm = j.get("next_url"), {"apiKey": pk}
            STATUS["treasury"] = "Treasury unavailable; using Massive (1-day lag)"
        except Exception as e:
            STATUS["treasury_massive"] = f"FAILED: {e}"
    STATUS.setdefault("treasury", "ok")
    return rows


def fedfunds() -> dict:
    try:
        j = http("https://api.db.nomics.world/v22/series/FED/H15/RIFSPFF_N.B?observations=1", timeout=60)
        doc = j["series"]["docs"][0]
        new = {p: v for p, v in zip(doc["period"], doc["value"]) if isinstance(v, (int, float)) and p >= str(TODAY.year - 3)}
        STATUS["fedfunds"] = "ok"
        return merge("fedfunds", new)
    except Exception as e:
        return _fallback("fedfunds", "fedfunds", e)


def fred(sid: str, years: int = 3) -> dict:
    try:
        txt = http(FRED_URL, {"id": sid, "cosd": str(TODAY - timedelta(days=365 * years + 40))}, timeout=20, as_json=False)
        new = {}
        for line in txt.strip().splitlines()[1:]:
            d, v = line.split(",")
            if v not in (".", ""):
                new[d] = float(v)
        if not new:
            raise RuntimeError("no rows")
        STATUS[f"FRED {sid}"] = "ok"
        return merge(f"fred_{sid}", new)
    except Exception as e:
        return _fallback(f"FRED {sid}", f"fred_{sid}", e)


def core_cpi() -> dict:
    """{YYYY-MM-01: core CPI index} from Massive (monthly)."""
    try:
        pk = CFG["polygon_api_key"]
        j = http("https://api.polygon.io/fed/v1/inflation",
                 {"date.gte": str(TODAY - timedelta(days=1200)), "sort": "date.asc", "limit": 5000, "apiKey": pk}, tries=5)
        new = {r["date"]: r["cpi_core"] for r in j.get("results", []) if r.get("cpi_core") is not None}
        if not new:
            raise RuntimeError("no rows")
        STATUS["core CPI"] = "ok"
        return merge("corecpi", new)
    except Exception as e:
        return _fallback("core CPI", "corecpi", e)


def auctions(days_ahead=8) -> list:
    """Upcoming coupon auctions (notes/bonds/TIPS) from TreasuryDirect."""
    try:
        j = http("https://www.treasurydirect.gov/TA_WS/securities/upcoming", {"format": "json"}, timeout=20)
        out = []
        for x in j:
            d = (x.get("auctionDate") or "")[:10]
            if x.get("securityType") in ("Note", "Bond", "TIPS") and str(TODAY) < d <= str(TODAY + timedelta(days=days_ahead)):
                out.append((d, x.get("securityTerm", "?"), x.get("securityType")))
        STATUS["auctions"] = "ok"
        return sorted(set(out))
    except Exception as e:
        STATUS["auctions"] = f"FAILED ({e})"
        return []


# ---------------------------------------------------------------- prices
def _ibkr():
    from ib_insync import IB
    ib = IB(); ib.RequestTimeout = 90
    for port in (7497, 7496):
        try:
            ib.connect("127.0.0.1", port, clientId=random.randint(9000, 9999), timeout=15, readonly=True)
            return ib
        except Exception:
            continue
    return None


def prices(tickers: list) -> dict:
    """{ticker: {date: close}} -- IBKR read-only (3y on first fill, 6m updates) when TWS is up, otherwise
    UW regular-session bars (1y max). VIX comes from IBKR or FRED VIXCLS."""
    have = {t: load(f"px_{t}", {}) or {} for t in tickers + ["VIX"]}
    ib = None
    try:
        ib = _ibkr()
        if ib:
            from ib_insync import Index, Stock
            for t in tickers + ["VIX"]:
                c = Index("VIX", "CBOE") if t == "VIX" else Stock(t.replace(".", " "), "SMART", "USD")
                if not ib.qualifyContracts(c):
                    continue
                dur = "3 Y" if len(have[t]) < 500 else "6 M"
                bars = ib.reqHistoricalData(c, "", dur, "1 day", "TRADES", useRTH=True)
                if bars:
                    have[t] = merge(f"px_{t}", {str(b.date): b.close for b in bars})
            STATUS["prices"] = "ok (IBKR)"
    except Exception as e:
        STATUS["prices"] = f"IBKR failed ({type(e).__name__}); trying UW"
    finally:
        if ib:
            ib.disconnect()
    stale = [t for t in tickers if not have[t] or max(have[t]) < _last_weekday()]
    if stale:
        try:
            from unusual_whales_client import UnusualWhalesClient
            uw = UnusualWhalesClient(timeout=30)
            for t in stale:
                try:
                    d = uw._get(f"/api/stock/{t}/ohlc/1d", {"limit": 900})
                    rows = [r for r in (d.get("data", d) if isinstance(d, dict) else d) if r.get("market_time") == "r"]
                    have[t] = merge(f"px_{t}", {r["date"]: float(r["close"]) for r in rows})
                except Exception:
                    pass
                time.sleep(0.4)
            STATUS["prices"] = STATUS.get("prices", "") + f" | UW for {len(stale)} tickers"
        except Exception as e:
            STATUS["prices"] = f"FAILED ({e})"
    if not have["VIX"] or max(have["VIX"]) < _last_weekday():
        have["VIX"] = merge("px_VIX", fred("VIXCLS"))
    return have


def _last_weekday():
    d = TODAY if datetime.now().hour >= 17 else TODAY - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return str(d)


def top_holdings(n=10) -> list:
    """[(ticker, weight%)] -- SPY's top weights from SSGA's daily file, refreshed weekly. GOOG folds into GOOGL."""
    cached = load("holdings")
    if cached and cached.get("fetched", "") >= str(TODAY - timedelta(days=7)):
        return cached["top"]
    try:
        r = requests.get(SSGA_URL, headers=UA, timeout=60); r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        shared = ["".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t"))
                  for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        rows = []
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{ns['m']}}}row"):
            vals = {}
            for cl in row.findall("m:c", ns):
                v = cl.find("m:v", ns)
                if v is not None:
                    vals[re.match(r"[A-Z]+", cl.get("r")).group()] = shared[int(v.text)] if cl.get("t") == "s" else v.text
            rows.append(vals)
        hdr_i = next(i for i, r_ in enumerate(rows) if "Ticker" in r_.values())
        col = {v: k for k, v in rows[hdr_i].items()}
        w = {}
        for r_ in rows[hdr_i + 1:]:
            t, wt = r_.get(col["Ticker"]), r_.get(col["Weight"])
            try:
                t = "GOOGL" if t == "GOOG" else t
                w[t] = w.get(t, 0) + float(wt)
            except (TypeError, ValueError):
                pass
        top = sorted(w.items(), key=lambda kv: -kv[1])[:n]
        save("holdings", {"fetched": str(TODAY), "top": top})
        STATUS["holdings"] = "ok"
        return top
    except Exception as e:
        STATUS["holdings"] = f"FAILED ({type(e).__name__}); using {'cache' if cached else 'built-in list'}"
        return cached["top"] if cached else DEFAULT_TOP


# ---------------------------------------------------------------- Unusual Whales
def uw_all(trading_days: list, asof: str, top: list) -> dict:
    """ETF flows, SPX/SPXW gamma, HYG short interest, options net premium (accumulating cache), SPY
    dark-pool level, economic calendar, leader earnings."""
    out = {"flows": {}, "gamma": {}, "si": {}, "np": {}, "calendar": [], "earnings": {}, "dp_level": None}
    try:
        from unusual_whales_client import UnusualWhalesClient
        c = UnusualWhalesClient(timeout=30)
    except Exception as e:
        STATUS["UW"] = f"FAILED ({e})"
        for t in ("TLT",):
            out["flows"][t] = load(f"flow_{t}", {}) or {}
        for t in ("SPX", "SPXW"):
            out["gamma"][t] = load(f"gamma_{t}", {}) or {}
        out["si"]["HYG"] = load("si_HYG", {}) or {}
        for t in ("QQQ", "SMH", "XLF", "KRE"):
            out["np"][t] = load(f"np_{t}", {}) or {}
        return out
    D = lambda x: x.get("data", x) if isinstance(x, dict) else x
    f = lambda v: float(v or 0)
    errs = []

    def step(label, fn):
        try:
            fn()
        except Exception as e:
            errs.append(f"{label}: {str(e)[:60]}")
        time.sleep(0.35)

    for t in ("TLT",):
        def _flow(t=t):
            out["flows"][t] = merge(f"flow_{t}", {r["date"]: f(r["change_prem"]) for r in D(c._get(f"/api/etfs/{t}/in-outflow"))})
        step(f"flows {t}", _flow)
        out["flows"].setdefault(t, load(f"flow_{t}", {}) or {})
    for t in ("SPX", "SPXW"):
        def _g(t=t):
            out["gamma"][t] = merge(f"gamma_{t}", {r["date"]: f(r["call_gamma"]) + f(r["put_gamma"])
                                                   for r in D(c._get(f"/api/stock/{t}/greek-exposure"))})
        step(f"gamma {t}", _g)
        out["gamma"].setdefault(t, load(f"gamma_{t}", {}) or {})

    def _si():
        rows = D(c._get("/api/shorts/HYG/interest-float/v2"))
        out["si"]["HYG"] = merge("si_HYG", {r["market_date"]: f(r["si_float"]) for r in rows if r.get("si_float")})
    step("HYG short interest", _si)
    out["si"].setdefault("HYG", load("si_HYG", {}) or {})

    # options net premium: no history endpoint, so fetch any of the last 130 trading days not yet cached
    after_close = datetime.now().hour >= 17
    want = [d for d in trading_days[-130:] if d < str(TODAY) or (d == str(TODAY) and after_close)]
    for t in ("QQQ", "SMH", "XLF", "KRE"):
        cache = load(f"np_{t}", {}) or {}
        missing = [d for d in want if d not in cache]
        for d in missing:
            try:
                ticks = D(c._get(f"/api/stock/{t}/net-prem-ticks", {"date": d}))
                # None marks "UW has nothing for this day" so it isn't refetched on every run
                cache[d] = sum(f(x.get("net_call_premium")) - f(x.get("net_put_premium")) for x in ticks) if ticks else None
            except Exception as e:
                errs.append(f"netprem {t} {d}: {str(e)[:40]}")
            time.sleep(0.3)
        if missing:
            save(f"np_{t}", cache)
        out["np"][t] = cache

    def _dp():
        agg = {}
        for d in trading_days[-5:]:
            for r in D(c._get("/api/darkpool/SPY/price-levels", {"date": d})):
                k = int(f(r["price"]) // 2 * 2)
                agg[k] = agg.get(k, 0) + f(r.get("dark_pool_volume"))
            time.sleep(0.3)
        out["dp_level"] = max(agg, key=agg.get) if agg else None
    step("SPY dark pool", _dp)

    def _cal():
        keys = ("CPI", "Consumer Price", "PPI", "PCE", "GDP", "Payroll", "Employment", "Jobless", "JOLTS", "ISM",
                "Retail Sales", "FOMC", "Federal Open", "Fed ", "FRB", "Chair", "Governor", "Michigan", "Beige")
        seen = set()
        for e in sorted(D(c._get("/api/market/economic-calendar")), key=lambda e: e["time"]):
            k = (e["time"][:16], e["event"])
            if k in seen or not (str(TODAY) < e["time"][:10] <= str(TODAY + timedelta(days=8))):
                continue
            seen.add(k)
            if any(w in e["event"] for w in keys):
                out["calendar"].append((e["time"][:10], e["event"]))
    step("calendar", _cal)

    for t, _ in top:
        def _e(t=t):
            rows = D(c._get(f"/api/stock/{t.replace('.', '')}/earnings"))
            fut = sorted(str(r.get("report_date") or "") for r in (rows if isinstance(rows, list) else [])
                         if str(r.get("report_date") or "") >= str(TODAY))
            if fut:
                out["earnings"][t] = fut[0]
        step(f"earnings {t}", _e)

    STATUS["UW"] = "ok" if not errs else f"partial ({len(errs)} errors, e.g. {errs[0]})"
    return out


def collect(top=None) -> dict:
    """Everything the model needs."""
    top = top or top_holdings()
    tickers = sorted({"SPY", "RSP", "QQQ", "SMH", "UUP", "USO"} | {t for t, _ in top})
    px = prices(tickers)
    trading_days = sorted(px["SPY"])
    asof = trading_days[-1]
    return dict(
        asof=asof, trading_days=trading_days, top=top, px=px,
        tsy=treasury(), ff=fedfunds(),
        hy=fred("BAMLH0A0HYM2"), real10=fred("DFII10"), be10=fred("T10YIE"), claims=fred("ICSA", years=4),
        corecpi=core_cpi(), auctions=auctions(),
        uw=uw_all(trading_days, asof, top),
        config=json.loads((HERE / "macro_watch_config.json").read_text()) if (HERE / "macro_watch_config.json").exists() else {},
    )
