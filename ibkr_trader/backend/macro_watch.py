"""Daily macro watch (v2) -> Telegram. Task IBKR-MacroWatch runs it weekdays after the close.

Tracks the fork from the 2026-10-02 macro study -- a 2018-style rate-shock correction vs. yields rolling
over and the market broadening -- as a stress composite: 19 metrics, each ranked against its own past
year, averaged into 5 pillars (rates, credit, breadth, positioning, macro), headline = the composite's
percentile vs its own replayed past year. v2 replaced v1's fixed-threshold vote count after a review
showed v1's typical reading was its "normal" (52nd pct) while it printed "BAD path gaining".

Read-only everywhere (IBKR readonly=True). Data: macro_watch_data.py; scoring: macro_watch_model.py.

    python macro_watch.py              # report for the latest close (skips if already sent), Telegram + history
    python macro_watch.py --dry-run    # print only
    python macro_watch.py --force      # resend even if this close was already reported
    python macro_watch.py --selfcheck  # include the self-check section (always included on Fridays)
"""
import json
import re
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HISTORY = HERE / "macro_watch_history.jsonl"
DRY, FORCE, SELFCHECK = "--dry-run" in sys.argv, "--force" in sys.argv, "--selfcheck" in sys.argv


def telegram(text: str):
    cfg = json.loads((HERE / "scanner_config.json").read_text())
    try:
        from telegram_alert_gate import alert_enabled
        if not alert_enabled("macro_watch"):
            print("macro_watch muted in admin panel"); return
    except Exception:
        pass                                                   # gate fails open, like everywhere else
    r = requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"], "text": text[:4000]}, timeout=20)
    r.raise_for_status()


def short_event(e: str) -> str:
    m = re.match(r"FRB (.+?) President (\S+ \S+)", e)
    if m:
        return f"Fed {m.group(2)} ({m.group(1)}) speaks"
    e = (e.replace("Federal Open Market Committee meeting minutes", "FOMC minutes")
          .replace("Federal Open Market Committee", "FOMC").replace("Report On Business ", "")
          .replace("U. Michigan Prelim Consumer Survey", "UMich sentiment (prelim)"))
    return e if len(e) <= 40 else e[:39].rsplit(" ", 1)[0] + "..."


def auction_label(term: str, typ: str) -> str:
    m = re.match(r"(\d+)-Year(?: (\d+)-Month)?", term or "")
    if not m:
        return f"{term} {typ} auction"
    yrs = int(m.group(1)) + (1 if m.group(2) and int(m.group(2)) >= 6 else 0)   # 9y10m reopening = the 10y
    return f"{yrs}y {'TIPS' if typ == 'TIPS' else typ.lower()} auction"


def band(p):
    if p is None:
        return "n/a (history still building)"
    return ("ELEVATED (bad-path conditions)" if p >= 0.8 else "above normal" if p >= 0.6 else "normal" if p > 0.4
            else "below normal" if p > 0.2 else "LOW (good-path conditions)")


def render(res, data, STATUS):
    t, v, x = res["today"], res["values"], res["extra"]
    M = t["metrics"]
    P = lambda k: f" p{round(M[k]['pct'] * 100)}" if k in M and M[k]["pct"] is not None else ""
    day = date.fromisoformat(res["day"])
    cp = t.get("comp_pct"); pp = res["prev"].get("comp_pct") if res["prev"] else None
    L = [f"MACRO WATCH {day:%a %b %d} close",
         f"STRESS: {f'{round(cp * 100)}th pct' if cp is not None else 'n/a'} of past year -> {band(cp)}"
         + (f" | prev day {round(pp * 100)}th" if pp is not None else "")
         + (f" | composite {t['composite']:+.2f}" if t.get("composite") is not None else ""),
         "Pillars (-1 calm..+1 stressed): " + " | ".join(f"{p.title()} {s:+.2f}" for p, s in t["pillars"].items())]
    if t.get("growth_scare"):
        L.append("! GROWTH-SCARE pattern: yields falling while credit/claims deteriorate -- falling yields NOT scored as good")
    L.append("Changed vs prev day: " + ("; ".join(f"{a} {b}->{c}" for a, b, c in res["flips"][:5]) or "nothing crossed a bucket"))
    L.append("")

    if v.get("r_y10") is not None:
        why = f": real {v['r_dreal']:+.0f}, infl.exp {x['dbe']:+.0f}" if v.get("r_dreal") is not None and x.get("dbe") is not None else ""
        tsy_tag = "" if res["obs"]["y10"] == res["day"] else f" [Treasury {res['obs']['y10']}]"
        s = f"RATES{tsy_tag}: 10y {v['r_y10']:.2f}%{P('r_y10')}"
        if v.get("r_d10y") is not None:
            s += f" | 20d {v['r_d10y']:+.0f}bp{P('r_d10y')} ({why.lstrip(': ')})" if why else f" | 20d {v['r_d10y']:+.0f}bp{P('r_d10y')}"
        if x.get("y2ff") is not None:
            s += f" | 2y-FF {x['y2ff']:+.0f}bp ({v.get('r_dhike', 0):+.0f}/20d; hike-pricing proxy){P('r_dhike')}"
        if x.get("y3m") is not None:
            s += f" | 10y-3m {v['r_y10'] - x['y3m']:+.2f}"
        if v.get("r_tlt") is not None:
            s += f" | TLT flows 5d {v['r_tlt'] / 1e9:+.1f}B{P('r_tlt')}"
        L.append(s)
    else:
        L.append("RATES: n/a")
    if v.get("c_hy") is not None:
        s = f"CREDIT: HY spread {v['c_hy']:.2f}%{P('c_hy')}"
        if v.get("c_dhy") is not None:
            s += f" ({v['c_dhy']:+.0f}bp/20d){P('c_dhy')}"
        if v.get("c_si") is not None:
            s += f" | HYG short interest {v['c_si'] * 100:.0f}% of float ({res['obs']['si']}){P('c_si')}"
        L.append(s)
    else:
        L.append("CREDIT: n/a")
    spy = data["px"]["SPY"].get(res["day"])
    s = f"BREADTH: SPY {spy:.2f}" if spy else "BREADTH:"
    if v.get("b_offhi") is not None:
        s += f" ({v['b_offhi'] * 100:+.1f}% vs high)"
    if v.get("b_gap") is not None:
        s += f" | SPY-RSP 20d {v['b_gap'] * 100:+.1f}pt{P('b_gap')}"
    if v.get("b_lead") is not None:
        s += f" | top-10 below 50d: {v['b_lead'] * 100:.0f}% of weight{P('b_lead')}" + (f" ({', '.join(x['lead_names'])})" if x["lead_names"] else "")
    L.append(s)
    s = "POSITIONING:"
    if v.get("p_gamma") is not None:
        s += f" SPX gamma {'+' if v['p_gamma'] > 0 else '-'}{P('p_gamma')} |"
    if v.get("p_lead") is not None:
        s += f" QQQ+SMH opt prem 5d {v['p_lead'] / 1e6:+.0f}M{P('p_lead')} |"
    if v.get("p_bank") is not None:
        s += f" XLF+KRE {v['p_bank'] / 1e6:+.0f}M{P('p_bank')} |"
    if v.get("p_vix") is not None:
        s += f" VIX {v['p_vix']:.1f}{P('p_vix')} |"
    if data["uw"].get("dp_level"):
        s += f" SPY dark-pool level {data['uw']['dp_level']}"
    L.append(s.rstrip(" |"))
    s = "MACRO:"
    if v.get("m_core") is not None:
        s += f" core CPI 3m {v['m_core']:.1f}% ann. ({(x['core_month'] or '')[:7]}) |"
    if v.get("m_claims") is not None:
        s += f" claims 4wk/26wk {v['m_claims'] * 100:+.0f}% (last {x['claims_last'] / 1000:.0f}k){P('m_claims')} |"
    if v.get("m_usd") is not None:
        s += f" dollar {v['m_usd'] * 100:+.1f}%/20d{P('m_usd')} |"
    if v.get("m_oil") is not None:
        s += f" oil {v['m_oil'] * 100:+.1f}%/20d{P('m_oil')}"
    L.append(s.rstrip(" |"))
    L.append("")

    # calendar: UW economic calendar + Treasury coupon auctions + verified key dates from macro_watch_config.json
    today = date.today()
    ev = {}
    for d, e in data["uw"].get("calendar", []):
        ev.setdefault(d, []).append(short_event(e))
    for d, term, typ in data.get("auctions", []):
        ev.setdefault(d, []).append(auction_label(term, typ))
    for kd in data.get("config", {}).get("key_dates", []):
        if str(today) < kd["date"] <= str(date.fromordinal(today.toordinal() + 8)):
            ev.setdefault(kd["date"], []).append(kd["label"])
    if ev:
        L.append("NEXT 8 DAYS: " + " / ".join(f"{date.fromisoformat(d):%a %m-%d} " + ", ".join(dict.fromkeys(es))
                                              for d, es in sorted(ev.items())))
    later = [kd for kd in data.get("config", {}).get("key_dates", [])
             if str(date.fromordinal(today.toordinal() + 8)) < kd["date"] <= str(date.fromordinal(today.toordinal() + 30))]
    if later:
        L.append("KEY DATES: " + ", ".join(f"{kd['label']} {date.fromisoformat(kd['date']):%a %m-%d}" for kd in later))
    earn = sorted((d, tk) for tk, d in data["uw"].get("earnings", {}).items()
                  if d <= str(date.fromordinal(today.toordinal() + 35)))
    if earn:
        L.append("LEADER EARNINGS (next 35d): " + ", ".join(f"{tk} {d[5:]}" for d, tk in earn))

    if SELFCHECK or day.weekday() == 4:
        sc = res["selfcheck"]
        f = lambda g: f"{g[1] * 100:.0f}% (n={g[0]})" if g[1] is not None else "n/a"
        L.append(f"SELF-CHECK (replay, SPY touched -5% within 18d): stress >=80th pct {f(sc['high'])} | middle {f(sc['mid'])} | "
                 f"<=20th {f(sc['low'])}. Overlapping windows, same period the model was designed on -- weak evidence.")
    bad = {k: s for k, s in STATUS.items() if not s.startswith("ok")}
    L.append("Data issues: " + "; ".join(f"{k}: {s}" for k, s in bad.items())[:500] if bad else "Sources: all ok")
    L.append("pNN = rank vs the metric's own past year (p100 = highest). Context only - not a trade signal.")
    return "\n".join(L)


def history_rows():
    if not HISTORY.exists():
        return []
    rows = []
    for line in HISTORY.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def main():
    import macro_watch_data as dmod
    import macro_watch_model as model
    data = dmod.collect()
    rows = history_rows()
    if not (DRY or FORCE) and any(r.get("date") == data["asof"] and r.get("version") == 2 for r in rows):
        print(f"close {data['asof']} already reported -- nothing new"); return
    res = model.run(data)
    msg = render(res, data, dmod.STATUS)
    print(msg)
    if DRY:
        return
    t = res["today"]
    row = dict(version=2, date=res["day"], run_at=datetime.now().isoformat(timespec="seconds"),
               comp_pct=t.get("comp_pct"), composite=t.get("composite"), pillars=t["pillars"],
               growth_scare=t.get("growth_scare"), metrics={k: round(m["stress"], 3) for k, m in t["metrics"].items()},
               flips=res["flips"])
    rows = [r for r in rows if r.get("date") != res["day"]] + [row]
    HISTORY.write_text("".join(json.dumps(r) + "\n" for r in rows))
    telegram(msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        (HERE / "macro_watch_cache").mkdir(exist_ok=True)
        (HERE / "macro_watch_cache" / "last_error.txt").write_text(f"{datetime.now().isoformat()}\n{tb}")
        print(tb)
        if not DRY:
            try:
                telegram(f"MACRO WATCH FAILED {datetime.now():%a %H:%M}: {type(e).__name__}: {str(e)[:300]}\n"
                         f"Details: ibkr_trader/backend/macro_watch_cache/last_error.txt. No report today -- this is not a quiet day.")
            except Exception:
                pass
        sys.exit(1)
