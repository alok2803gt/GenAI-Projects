"""Scoring model for macro_watch (v2).

Each metric is turned into a stress value in [-1, +1] by ranking today's reading against that same
metric's own previous 252 trading days (so a $1B TLT flow, a -$100M QQQ premium and a 3.24% HY spread
are each judged against their own normal -- the v1 fixed thresholds made some signals never fire and
others fire 70% of the time). Metrics are averaged within five pillars and the pillars averaged into a
composite, so correlated signals (three flavours of "yields up") can't outvote everything else. The
headline is the composite's percentile against its own replayed past year.

Every value is taken as of its real publication date (FRED daily series +1 day, claims +5, short
interest +10, CPI mid-next-month), so the replay never uses information that wasn't out yet.
"""
from __future__ import annotations

import bisect
import statistics as st
from datetime import date, timedelta

PILLARS = ("RATES", "CREDIT", "BREADTH", "POSITIONING", "MACRO")
WINDOW, MIN_OBS = 252, 60
REPLAY_DAYS = 400
FWD = 18

# key, pillar, label, direction (+1: higher reading = more stress), scale
METRICS = [
    ("r_d10y", "RATES", "10y change 20d", +1, "pct"),
    ("r_y10", "RATES", "10y level", +1, "pct"),
    ("r_dhike", "RATES", "2y-minus-fed-funds change 20d", +1, "pct"),
    ("r_dreal", "RATES", "real 10y change 20d", +1, "pct"),
    ("r_tlt", "RATES", "TLT flows 5d", -1, "pct"),
    ("c_dhy", "CREDIT", "HY spread change 20d", +1, "pct"),
    ("c_hy", "CREDIT", "HY spread level", +1, "pct"),
    ("c_si", "CREDIT", "HYG short interest", +1, "pct"),
    ("b_gap", "BREADTH", "SPY-minus-RSP 20d", +1, "pct"),
    ("b_lead", "BREADTH", "top holdings below 50d", +1, "pct"),
    ("b_offhi", "BREADTH", "SPY distance from high", -1, "pct"),
    ("p_gamma", "POSITIONING", "SPX dealer gamma", -1, "pct"),
    ("p_lead", "POSITIONING", "QQQ+SMH options premium 5d", -1, "pct"),
    ("p_bank", "POSITIONING", "XLF+KRE options premium 5d", -1, "pct"),
    ("p_vix", "POSITIONING", "VIX", +1, "pct"),
    ("m_core", "MACRO", "core CPI 3m annualized", +1, "core"),
    ("m_claims", "MACRO", "jobless claims trend", +1, "pct"),
    ("m_usd", "MACRO", "dollar 20d", +1, "pct"),
    ("m_oil", "MACRO", "oil 20d", +1, "pct"),
]
META = {k: (p, lab, dr, sc) for k, p, lab, dr, sc in METRICS}


class Series:
    """As-of lookups with a publication lag: an observation dated d becomes usable on d + lag days."""

    def __init__(self, d: dict, lag: int = 0, key: str | None = None, avail=None):
        items = []
        for k, v in (d or {}).items():
            v = v.get(key) if key else v
            if v is not None:
                items.append((k, v))
        items.sort()
        self.obs = [k for k, _ in items]
        self.v = [v for _, v in items]
        if avail:
            self.k = [avail(k) for k in self.obs]
        else:
            self.k = [str(date.fromisoformat(k) + timedelta(days=lag)) if lag else k for k in self.obs]

    def idx(self, day):
        return bisect.bisect_right(self.k, day) - 1

    def at(self, day, back=0, max_age=None):
        i = self.idx(day)
        if i < 0 or i - back < 0:
            return None
        if max_age is not None and (date.fromisoformat(day) - date.fromisoformat(self.k[i])).days > max_age:
            return None
        return self.v[i - back]

    def window(self, day, n):
        i = self.idx(day)
        return self.v[max(0, i - n + 1): i + 1] if i >= 0 else []

    def obs_at(self, day, back=0):
        i = self.idx(day) - back
        return self.obs[i] if i >= 0 else None


def _cpi_avail(month_key):          # monthly CPI for month M is released ~13-15th of M+1
    y, m = int(month_key[:4]), int(month_key[5:7])
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return str(date(y, m, 15))


def inputs(data: dict) -> dict:
    uw = data["uw"]
    return dict(
        y10=Series(data["tsy"], key="y10"), y2=Series(data["tsy"], key="y2"), y3m=Series(data["tsy"], key="y3m"),
        ff=Series(data["ff"]), real10=Series(data["real10"], lag=1), be10=Series(data["be10"], lag=1),
        hy=Series(data["hy"], lag=1), claims=Series(data["claims"], lag=5), core=Series(data["corecpi"], avail=_cpi_avail),
        tlt=Series(uw["flows"].get("TLT", {}), lag=1),
        gspx=Series(uw["gamma"].get("SPX", {})), gspxw=Series(uw["gamma"].get("SPXW", {})),
        si=Series(uw["si"].get("HYG", {}), lag=10),
        np={t: uw["np"].get(t, {}) for t in ("QQQ", "SMH", "XLF", "KRE")},
        px={t: Series(v) for t, v in data["px"].items()},
        top=data["top"], days=data["trading_days"],
    )


def _hike(I, d, back):
    """2y yield minus effective fed funds (bp) at the Treasury observation `back` days before d.
    A hike-pricing PROXY: it also contains the 2y term premium."""
    y2 = I["y2"].at(d, back, max_age=5 if back == 0 else None)
    od = I["y2"].obs_at(d, back)
    ff = I["ff"].at(od) if od else None
    return None if y2 is None or ff is None else (y2 - ff) * 100


def _netprem(I, d, names):
    i = bisect.bisect_right(I["days"], d) - 1
    if i < 4:
        return None
    win = I["days"][i - 4:i + 1]
    vals = [I["np"][t].get(x) for t in names for x in win]
    return None if any(v is None for v in vals) else sum(vals)


def values_at(I, d) -> dict:
    m = {}
    a, b = I["y10"].at(d, max_age=5), I["y10"].at(d, 20)
    m["r_y10"] = a
    if a is not None and b is not None:
        m["r_d10y"] = (a - b) * 100
    h0, h20 = _hike(I, d, 0), _hike(I, d, 20)
    m["_hike"] = h0
    if h0 is not None and h20 is not None:
        m["r_dhike"] = h0 - h20
    a, b = I["real10"].at(d, max_age=6), I["real10"].at(d, 20)
    if a is not None and b is not None:
        m["r_dreal"] = (a - b) * 100
    a, b = I["be10"].at(d, max_age=6), I["be10"].at(d, 20)
    if a is not None and b is not None:
        m["_dbe"] = (a - b) * 100
    if I["tlt"].at(d, max_age=6) is not None:
        w = I["tlt"].window(d, 5)
        if len(w) == 5:
            m["r_tlt"] = sum(w)

    a, b = I["hy"].at(d, max_age=6), I["hy"].at(d, 20)
    m["c_hy"] = a
    if a is not None and b is not None:
        m["c_dhy"] = (a - b) * 100
    m["c_si"] = I["si"].at(d, max_age=45)

    spy, rsp = I["px"]["SPY"], I["px"]["RSP"]
    s0, s20, r0, r20 = spy.at(d, max_age=4), spy.at(d, 20), rsp.at(d, max_age=4), rsp.at(d, 20)
    if None not in (s0, s20, r0, r20):
        m["b_gap"] = (s0 / s20 - 1) - (r0 / r20 - 1)
    w = spy.window(d, 252)
    if s0 and len(w) >= 200:
        m["b_offhi"] = s0 / max(w) - 1
    tot = below = 0.0
    names = []
    for t, wt in I["top"]:
        s = I["px"].get(t)
        if not s or s.at(d, max_age=4) is None:
            continue
        win = s.window(d, 50)
        if len(win) == 50:
            tot += wt
            if win[-1] < sum(win) / 50:
                below += wt; names.append(t)
    if tot >= 0.6 * sum(w_ for _, w_ in I["top"]):
        m["b_lead"] = below / tot
        m["_lead_names"] = names

    g1, g2 = I["gspx"].at(d, max_age=4), I["gspxw"].at(d, max_age=4)
    if g1 is not None and g2 is not None:
        m["p_gamma"] = g1 + g2
    m["p_lead"] = _netprem(I, d, ("QQQ", "SMH"))
    m["p_bank"] = _netprem(I, d, ("XLF", "KRE"))
    m["p_vix"] = I["px"]["VIX"].at(d, max_age=5) if "VIX" in I["px"] else None

    c0, c3 = I["core"].at(d), I["core"].at(d, 3)
    if c0 and c3:
        m["m_core"] = ((c0 / c3) ** 4 - 1) * 100
        m["_core_month"] = I["core"].obs_at(d)
    if I["claims"].at(d, max_age=14) is not None:
        cw = I["claims"].window(d, 26)
        if len(cw) == 26:
            m["m_claims"] = (sum(cw[-4:]) / 4) / (sum(cw) / 26) - 1
            m["_claims_last"] = cw[-1]
    for key, t in (("m_usd", "UUP"), ("m_oil", "USO")):
        s = I["px"].get(t)
        if s:
            a, b = s.at(d, max_age=4), s.at(d, 20)
            if a and b:
                m[key] = a / b - 1
    return m


def _pct(hist, v):
    lo = sum(1 for x in hist if x < v); eq = sum(1 for x in hist if x == v)
    return (lo + 0.5 * eq) / len(hist)


def stress_for(key, v, hist):
    """(stress in [-1,1], raw percentile or None)"""
    if v is None:
        return None, None
    _, _, direction, scale = META[key]
    if scale == "core":
        return max(-1.0, min(1.0, (v - 2.75) / 0.75)), None
    if len(hist) < MIN_OBS:
        return None, None
    p = _pct(hist, v)
    return (2 * p - 1) * direction, p


def score(vals_hist: list, i: int) -> dict:
    """Stresses, pillars and composite for day index i, using only days before i as the reference."""
    today = vals_hist[i]
    out = {"metrics": {}, "pillars": {}}
    for key, pillar, label, direction, scale in METRICS:
        hist = [vals_hist[j].get(key) for j in range(max(0, i - WINDOW), i)]
        hist = [x for x in hist if x is not None]
        s, p = stress_for(key, today.get(key), hist)
        if s is not None:
            out["metrics"][key] = {"value": today.get(key), "stress": s, "pct": p}
    v = today
    scare = (v.get("r_d10y") is not None and v["r_d10y"] <= -20 and
             ((v.get("c_dhy") or 0) >= 30 or (v.get("m_claims") or 0) >= 0.08))
    out["growth_scare"] = scare
    for p in PILLARS:
        xs = [x["stress"] for k, x in out["metrics"].items() if META[k][0] == p and not (scare and k == "r_tlt")]
        if xs:
            ps = sum(xs) / len(xs)
            if scare and p == "RATES":
                ps = max(ps, 0.0)          # falling yields in a growth scare are not good news
            out["pillars"][p] = ps
    out["composite"] = (sum(out["pillars"].values()) / len(out["pillars"])) if len(out["pillars"]) >= 3 else None
    return out


def bucket(s):
    return "stressed" if s >= 0.5 else "calm" if s <= -0.5 else "neutral"


def run(data: dict) -> dict:
    I = inputs(data)
    days = I["days"]
    span = days[-(REPLAY_DAYS + WINDOW):]
    vals = [values_at(I, d) for d in span]
    start = len(span) - REPLAY_DAYS
    scored = {}
    for i in range(max(start, 0), len(span)):
        scored[i] = score(vals, i)
    comps = {i: s["composite"] for i, s in scored.items()}
    for i, s in scored.items():
        hist = [comps[j] for j in range(max(min(scored), i - WINDOW), i) if comps.get(j) is not None]
        s["comp_pct"] = _pct(hist, s["composite"]) if s["composite"] is not None and len(hist) >= MIN_OBS else None
    last = len(span) - 1
    today, prev = scored[last], scored.get(last - 1)
    flips = []
    if prev:
        for k, x in today["metrics"].items():
            if k in prev["metrics"] and bucket(x["stress"]) != bucket(prev["metrics"][k]["stress"]):
                flips.append((META[k][1], bucket(prev["metrics"][k]["stress"]), bucket(x["stress"])))

    # self-check: did high composite percentiles precede SPY drawdowns? (overlapping windows; in-sample design)
    spy = I["px"]["SPY"]
    rows = []
    for i, s in scored.items():
        if s.get("comp_pct") is None:
            continue
        d = span[i]; k = spy.idx(d)
        if k + FWD < len(spy.v):
            dd = min(spy.v[k + 1:k + FWD + 1]) / spy.v[k] - 1
            rows.append((s["comp_pct"], dd))
    def grp(cond):
        g = [dd for p, dd in rows if cond(p)]
        return (len(g), sum(1 for x in g if x <= -0.05) / len(g) if g else None,
                sum(1 for x in g if x <= -0.03) / len(g) if g else None)
    selfcheck = {"high": grp(lambda p: p >= 0.8), "mid": grp(lambda p: 0.2 < p < 0.8), "low": grp(lambda p: p <= 0.2)}

    return dict(day=span[last], today=today, prev=prev, values=vals[last], flips=flips, selfcheck=selfcheck,
                replay=[(span[i], scored[i]["composite"], scored[i].get("comp_pct")) for i in sorted(scored)],
                obs={"y10": I["y10"].obs_at(span[last]), "hy": I["hy"].obs_at(span[last]),
                     "si": I["si"].obs_at(span[last]), "claims": I["claims"].obs_at(span[last]),
                     "tlt": I["tlt"].obs_at(span[last]), "gamma": I["gspx"].obs_at(span[last])},
                extra={"y2ff": vals[last].get("_hike"), "dbe": vals[last].get("_dbe"),
                       "y3m": I["y3m"].at(span[last], max_age=5), "lead_names": vals[last].get("_lead_names", []),
                       "core_month": vals[last].get("_core_month"), "claims_last": vals[last].get("_claims_last")})
