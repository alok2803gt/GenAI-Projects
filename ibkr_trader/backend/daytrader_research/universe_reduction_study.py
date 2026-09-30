"""Can a TIGHTER selection earn more than the fee at a $148 position?

CONTEXT (2026-09-29)
--------------------
Account moved to IBKR Tiered pricing, so the per-order minimum fell $1.00 ->
$0.35 and the round-trip hurdle at the live $148 position fell 1.35% -> 0.47%.
gate_threshold_study.py measured the best available edge at +0.174pp (t=1.23).
Question: does concentrating the universe -- higher score cut, fewer names per
day, or a directional filter -- lift excess return above the 0.47pp hurdle?

THE TRAP THIS SCRIPT IS DESIGNED AGAINST
----------------------------------------
Slicing a population until a subset clears a hurdle is how the level-break and
IDR "edges" were manufactured. Three protections:
  1. The grid is fixed BEFORE looking (5 score cuts x 5 top-N x 3 gap regimes).
  2. Every t is date-clustered -- one event per session.
  3. Bonferroni is applied over the FULL grid actually run, and printed.
A subset that clears 0.47pp on a t below the corrected bar is noise, and is
labelled as such rather than recommended.

ONE HYPOTHESIS HERE IS NOT DATA-MINED. The original 500-ticker study (quoted in
daytrader_scanner.py's docstring) found gap direction asymmetric: a gap DOWN
< -1% averaged +0.13% same-day return while a gap UP > +1% averaged -0.03%,
i.e. gap-down reversion was "the single best setup tested". The scanner scores
|gap_pct|, so it surfaces both, while the agent is long only. Testing gap-down
separately is a pre-existing directional hypothesis, not a fishing expedition.

    ../../venv/bin/python daytrader_research/universe_reduction_study.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from gate_threshold_study import (MIN_ATR_PCT, add_scores, build,  # noqa: E402
                                  clustered)

LIVE_POS = 148.0
TIERED_RT = 0.70                       # $0.35 each way
HURDLE = TIERED_RT / LIVE_POS * 100    # 0.47pp

SCORE_CUTS = (75, 80, 85, 90, 95)
TOP_NS = (0, 1, 2, 3, 5)               # 0 = no per-day cap
GAP_REGIMES = (("all gaps", None), ("gap DOWN < -1%", "down"), ("gap UP > +1%", "up"))


def main() -> None:
    r = add_scores(build())
    r = r[r.atr_pct >= MIN_ATR_PCT]
    print(f"panel: {r.ticker.nunique()} tickers x {r.date.nunique()} sessions, "
          f"{len(r):,} ticker-days (ATR% >= {MIN_ATR_PCT})")
    print(f"hurdle at ${LIVE_POS:.0f} on Tiered: round trip ${TIERED_RT:.2f} = {HURDLE:.2f}pp\n")

    n_tests = len(SCORE_CUTS) * len(TOP_NS) * len(GAP_REGIMES)
    # two-sided Bonferroni-corrected ~5% bar, normal approx
    from statistics import NormalDist
    bar = NormalDist().inv_cdf(1 - 0.05 / (2 * n_tests))
    print(f"grid = {len(SCORE_CUTS)} score cuts x {len(TOP_NS)} top-N x "
          f"{len(GAP_REGIMES)} gap regimes = {n_tests} tests")
    print(f"Bonferroni-corrected significance bar: |t| > {bar:.2f}\n")

    print(f"  {'gap regime':<16}{'score':>6}{'topN':>6}{'rows':>7}{'days':>6}"
          f"{'excess%':>10}{'t':>7}{'net':>8}  verdict")
    winners = []
    for gname, gkey in GAP_REGIMES:
        g = r
        if gkey == "down":
            g = r[r.gap_pct < -1.0]
        elif gkey == "up":
            g = r[r.gap_pct > 1.0]
        for cut in SCORE_CUTS:
            s0 = g[g.composite_score >= cut]
            for n in TOP_NS:
                s = s0
                if n:
                    s = s0.sort_values("composite_score", ascending=False) \
                          .groupby("date").head(n)
                if len(s) < 30:
                    continue
                m, t, nd, nr = clustered(s, "excess")
                net = m - HURDLE
                ok = (net > 0) and (abs(t) > bar)
                verdict = "PASSES" if ok else ("clears fee, t too weak" if net > 0 else "")
                if ok:
                    winners.append((gname, cut, n, m, t, nr, nd))
                print(f"  {gname:<16}{cut:>6}{n if n else '-':>6}{nr:>7}{nd:>6}"
                      f"{m:>10.3f}{t:>7.2f}{net:>8.3f}  {verdict}")

    print(f"\n--- does excess rise with score at all? (all gaps, no cap) ---")
    print(f"  {'score bucket':<16}{'rows':>8}{'excess%':>10}{'t':>7}")
    for lo, hi in ((75, 80), (80, 85), (85, 90), (90, 95), (95, 101)):
        s = r[(r.composite_score >= lo) & (r.composite_score < hi)]
        m, t, nd, nr = clustered(s, "excess")
        print(f"  {f'{lo}-{hi}':<16}{nr:>8}{m:>10.3f}{t:>7.2f}")

    print("\n=== CONCLUSION ===")
    if winners:
        print(f"  {len(winners)} subset(s) cleared BOTH the fee and |t|>{bar:.2f}:")
        for w in winners:
            print(f"    {w[0]}, score>={w[1]}, topN={w[2] or '-'}: "
                  f"{w[3]:+.3f}pp t={w[4]:.2f} ({w[5]} rows / {w[6]} days)")
        print("  These are IN-SAMPLE. Any of them needs a pre-registered holdout")
        print("  before a single live dollar -- the standing rule in this project.")
    else:
        print(f"  NO subset clears the {HURDLE:.2f}pp fee with |t| > {bar:.2f}.")
        print("  Reducing the universe does not rescue the strategy: concentrating")
        print("  into fewer names raises variance, not expected excess return.")


if __name__ == "__main__":
    main()
