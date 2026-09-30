"""Phase 2 data pull: dealer positioning (greek-exposure) and dark pool prints.

greek-exposure: ONE call per ticker returns ~250 sessions, but only back to
~2025-10-01 regardless of the date parameter (2024-11-15 returns 0 rows). So it
cannot be explored on the pre-2025-09-16 window and needs its own split.

darkpool: historical, but one call PER ticker-day.
"""
import json, sys, time, glob
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
GREEK = HERE / "uw_greek_cache"; GREEK.mkdir(exist_ok=True)
NOPE = HERE / "uw_nope_cache"; NOPE.mkdir(exist_ok=True)
LEVELS = HERE / "uw_levels_cache"; LEVELS.mkdir(exist_ok=True)
ALERTS = HERE / "uw_alerts_cache"; ALERTS.mkdir(exist_ok=True)
LVLPREV = HERE / "uw_levels_prev_cache"; LVLPREV.mkdir(exist_ok=True)
DARK = HERE / "uw_darkpool_cache"; DARK.mkdir(exist_ok=True)

def pairs():
    out = []
    for f in glob.glob(str(HERE / "uw_flow_cache" / "*.json")):
        stem = Path(f).stem
        t, d = stem.rsplit("_", 1)
        out.append((t, d))
    return sorted(out)

def main():
    import unusual_whales_client as uw
    c = uw.UnusualWhalesClient()
    mode = sys.argv[1] if len(sys.argv) > 1 else "greek"
    ps = pairs()
    if mode == "greek":
        tks = sorted({t for t, _ in ps})
        todo = [t for t in tks if not (GREEK / f"{t}.json").exists()]
        print(f"greek-exposure: {len(tks)} tickers, {len(todo)} to fetch", flush=True)
        for i, t in enumerate(todo, 1):
            rec = {"ticker": t}
            try:
                r = c._get(f"/api/stock/{t}/greek-exposure", {})
                rec["rows"] = r.get("data", r) if isinstance(r, dict) else r
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
            (GREEK / f"{t}.json").write_text(json.dumps(rec))
            time.sleep(0.3)
            if i % 25 == 0: print(f"  {i}/{len(todo)}", flush=True)
        print(f"done: {len(list(GREEK.glob('*.json')))} ticker files", flush=True)
    elif mode == "levels_prev":
        # The same-day levels file is WHOLE-DAY, so splitting it at an intraday
        # price is look-ahead (corr +0.67 with the forward return, verified
        # 2026-09-30). The legitimate test uses the PRIOR session's positioning
        # against today's open, which is genuinely knowable at entry.
        import datetime as _dt
        todo = []
        for t, d in ps:
            dd = _dt.date.fromisoformat(d)
            prev = dd - _dt.timedelta(days=1)
            while prev.weekday() >= 5:
                prev -= _dt.timedelta(days=1)
            if not (LVLPREV / f"{t}_{d}.json").exists():
                todo.append((t, d, prev.isoformat()))
        print(f"levels_prev: {len(todo)} to fetch (~{len(todo)*0.4/60:.0f} min)", flush=True)
        ok = fail = 0
        for i, (t, d, prev) in enumerate(todo, 1):
            rec = {"ticker": t, "date": d, "levels_date": prev}
            try:
                r = c._get(f"/api/stock/{t}/option/stock-price-levels", {"date": prev})
                rec["rows"] = r.get("data", r) if isinstance(r, dict) else r
                ok += 1
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"; fail += 1
            (LVLPREV / f"{t}_{d}.json").write_text(json.dumps(rec))
            time.sleep(0.35)
            if i % 250 == 0: print(f"  {i}/{len(todo)} ok={ok} fail={fail}", flush=True)
        print(f"done: ok={ok} fail={fail}", flush=True)
    elif mode in ("levels", "alerts"):
        # levels: call vs put volume by PRICE LEVEL -> directional positioning
        #         relative to spot.
        # alerts: UW's flagship unusual-options-activity feed. Historical only via
        #         newer_than/older_than (a plain `date` param is silently ignored
        #         and returns today's rows -- verified 2026-09-30).
        D = LEVELS if mode == "levels" else ALERTS
        todo = [(t, d) for t, d in ps if not (D / f"{t}_{d}.json").exists()]
        print(f"{mode}: {len(ps)} ticker-days, {len(todo)} to fetch "
              f"(~{len(todo)*0.45/60:.0f} min)", flush=True)
        ok = fail = 0
        for i, (t, d) in enumerate(todo, 1):
            rec = {"ticker": t, "date": d}
            try:
                if mode == "levels":
                    r = c._get(f"/api/stock/{t}/option/stock-price-levels", {"date": d})
                else:
                    # A NARROW window returns HTTP 500 (verified 2026-09-30);
                    # the full-day range is what the API accepts, so fetch the
                    # day and filter to the opening window client-side, which
                    # also avoids hard-coding an EST/EDT UTC offset.
                    nxt = (__import__("datetime").date.fromisoformat(d)
                           + __import__("datetime").timedelta(days=1)).isoformat()
                    r = c._get("/api/option-trades/flow-alerts",
                               {"ticker_symbol": t, "limit": 200,
                                "newer_than": f"{d}T00:00:00Z",
                                "older_than": f"{nxt}T00:00:00Z"})
                rec["rows"] = r.get("data", r) if isinstance(r, dict) else r
                ok += 1
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"; fail += 1
            (D / f"{t}_{d}.json").write_text(json.dumps(rec))
            time.sleep(0.35)
            if i % 250 == 0: print(f"  {i}/{len(todo)} ok={ok} fail={fail}", flush=True)
        print(f"done: ok={ok} fail={fail}", flush=True)
    elif mode == "nope":
        # NOPE = Net Options Pricing Effect: options delta-hedging pressure
        # relative to stock volume. Purpose-built DIRECTIONAL indicator, and the
        # endpoint returns one row per minute with full 2-year history.
        todo = [(t, d) for t, d in ps if not (NOPE / f"{t}_{d}.json").exists()]
        print(f"nope: {len(ps)} ticker-days, {len(todo)} to fetch "
              f"(~{len(todo)*0.45/60:.0f} min)", flush=True)
        ok = fail = 0
        for i, (t, d) in enumerate(todo, 1):
            rec = {"ticker": t, "date": d}
            try:
                r = c._get(f"/api/stock/{t}/nope", {"date": d})
                rec["ticks"] = r.get("data", r) if isinstance(r, dict) else r
                ok += 1
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"; fail += 1
            (NOPE / f"{t}_{d}.json").write_text(json.dumps(rec))
            time.sleep(0.35)
            if i % 200 == 0: print(f"  {i}/{len(todo)} ok={ok} fail={fail}", flush=True)
        print(f"done: ok={ok} fail={fail}", flush=True)
    else:
        todo = [(t, d) for t, d in ps if not (DARK / f"{t}_{d}.json").exists()]
        print(f"darkpool: {len(ps)} ticker-days, {len(todo)} to fetch "
              f"(~{len(todo)*0.45/60:.0f} min)", flush=True)
        ok = fail = 0
        for i, (t, d) in enumerate(todo, 1):
            rec = {"ticker": t, "date": d}
            try:
                r = c._get(f"/api/darkpool/{t}", {"date": d, "limit": 500})
                rec["prints"] = r.get("data", r) if isinstance(r, dict) else r
                ok += 1
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"; fail += 1
            (DARK / f"{t}_{d}.json").write_text(json.dumps(rec))
            time.sleep(0.35)
            if i % 200 == 0: print(f"  {i}/{len(todo)} ok={ok} fail={fail}", flush=True)
        print(f"done: ok={ok} fail={fail}", flush=True)

if __name__ == "__main__":
    main()
