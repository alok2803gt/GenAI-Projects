"""Pull Unusual Whales intraday options flow + prices for Day Trader candidates.

PURPOSE
-------
direction_search.py exhausted daily-bar features: 11 signed features x 2 panels
captured at most 0.088pp of a 1.726pp available move (5%), against a 0.473pp
fee hurdle (27% needed). Same-day direction is not in daily bars. Options flow
is the next candidate, and it is INTRADAY, which is what a same-day scalp needs.

WHAT IS PULLED, per (ticker, session) -- two calls:
  /api/stock/{t}/net-prem-ticks   390 rows, one per minute: net_call_premium,
                                  net_put_premium, net_delta, bid/ask-side
                                  call and put volume
  /api/stock/{t}/ohlc/5m          5-minute candles, giving a real 09:45 entry
                                  price and the closing price

History is capped at ~2 years on this key (2023-09-18 returns 403), so the
window is 2024-09-16 onward.

SCOPE: the top N candidates per session by composite_score, which mirrors what
the Day Trader would actually have traded rather than the whole universe.

Resumable: every (ticker, date) is cached as its own JSON, so an interrupted
run continues where it stopped and costs no repeat calls.

    ../../venv/bin/python daytrader_research/fetch_uw_flow.py [--top 5] [--limit N]
"""
import json
import sys
import time
from datetime import date as _date
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

CACHE = HERE / "uw_flow_cache"
CACHE.mkdir(exist_ok=True)
START = "2024-09-16"          # UW history limit on this key
TOP_N = 5
PAUSE = 0.25                  # ~4 req/s; the client also honours Retry-After


def candidates(top_n: int) -> list[tuple[str, str]]:
    """(ticker, date) for the top-N Day Trader candidates on each session."""
    from breakout_direction_study import build, add_scores   # identical features
    r = add_scores(build(HERE / "holdout_universe_5y.pkl"))
    r = r[(r.composite_score >= 75) & (r.atr_pct >= 2.5)]
    r = r[r["date"] >= START]
    picks = (r.sort_values("composite_score", ascending=False)
               .groupby("date").head(top_n)[["ticker", "date"]])
    return [(t, d.strftime("%Y-%m-%d")) for t, d in picks.itertuples(index=False)]


def main() -> None:
    import unusual_whales_client as uw
    top_n = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else TOP_N
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    pairs = candidates(top_n)
    if limit:
        pairs = pairs[:limit]
    todo = [(t, d) for t, d in pairs if not (CACHE / f"{t}_{d}.json").exists()]
    print(f"{len(pairs):,} candidate ticker-days from {START}; "
          f"{len(pairs) - len(todo):,} already cached, {len(todo):,} to fetch "
          f"(~{len(todo) * 2 * PAUSE / 60:.0f} min)", flush=True)

    c = uw.UnusualWhalesClient()
    ok = fail = 0
    for i, (tkr, d) in enumerate(todo, 1):
        rec = {"ticker": tkr, "date": d}
        try:
            r = c._get(f"/api/stock/{tkr}/net-prem-ticks", {"date": d})
            rec["ticks"] = r.get("data", r) if isinstance(r, dict) else r
            time.sleep(PAUSE)
            r = c._get(f"/api/stock/{tkr}/ohlc/5m", {"date": d, "limit": 500})
            rec["ohlc"] = r.get("data", r) if isinstance(r, dict) else r
            ok += 1
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            fail += 1
        (CACHE / f"{tkr}_{d}.json").write_text(json.dumps(rec))
        time.sleep(PAUSE)
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}  ok={ok} fail={fail}", flush=True)
    print(f"done: {ok} fetched, {fail} failed, cache holds "
          f"{len(list(CACHE.glob('*.json')))} files", flush=True)


if __name__ == "__main__":
    main()
