"""Does the Day Trader's confirmation gate turn "big mover" into "upward mover"?

WHY (2026-09-29)
----------------
gate_threshold_study.py established that the composite score raises the
>=0.5% MOVE hit rate (36.5% -> 42.7%) but leaves the SIGNED excess return at
+0.017pp, t=0.45. The score finds big movers, not upward ones -- and the Day
Trader is long-only. The confirmation gate is the only remaining component
that could supply the sign: it refuses to buy until price has already risen
confirm_pct (0.35%) within confirm_window_min (60) of the signal, with
RVOL >= 1.2.

WHAT THIS CAN AND CANNOT MEASURE
--------------------------------
Only daily OHLC is on disk for a universe this wide (the 1-minute caches under
ict_research/ cover 5 tickers). So confirmation is PROXIED: a day confirms if
high >= open * (1 + 0.35%), and entry is taken AT that level.

Two biases, both stated rather than buried, and both OPTIMISTIC -- so a
negative result here is strong, and a positive one would need intraday data
before being believed:
  1. TIMING. Daily bars cannot say WHEN the high occurred, so this counts
     confirmations that happened at 15:00 as though they happened inside the
     60-minute window. The live gate would have missed those.
  2. PATH. When both the +0.5% target and the -3% stop are inside the day's
     range, the order they were hit is unknowable from daily bars. Resolved as
     STOP FIRST (the pessimistic choice) so this bias runs against the
     strategy, partially offsetting bias 1.

Market-adjusted (excess vs the universe's own mean for the SAME measure that
day) and date-clustered (one event per session), same as the gate study.

    ../../venv/bin/python daytrader_research/confirmation_gate_study.py
"""
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from gate_threshold_study import (SCORE_MIN, MIN_ATR_PCT, ATR_REQ,  # noqa: E402
                                  add_scores, build, clustered)

CONFIRM_PCT = 0.35      # config confirm_pct
TARGET_PCT = 0.5        # config profit_target_pct (live status 2026-09-29)
STOP_PCT = 3.0          # config hard_stop_pct
LIVE_POS = 148.0        # 10% of net liq -- see gate_threshold_study
RT_FEE_PCT = 2.00 / LIVE_POS * 100


def main() -> None:
    r = add_scores(build())
    # build() drops Open/High/Low? -- it keeps them; re-attach explicit names.
    r = r.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c"})
    days = r["date"].nunique()
    print(f"panel: {r.ticker.nunique()} tickers x {days} sessions = {len(r):,} ticker-days")
    print(f"confirmation proxy: high >= open x (1+{CONFIRM_PCT}%), entry AT that level")
    print(f"target +{TARGET_PCT}%  stop -{STOP_PCT}%  ties resolved STOP-FIRST (pessimistic)\n")

    entry = r["o"] * (1 + CONFIRM_PCT / 100)
    r["confirmed"] = r["h"] >= entry
    r["entry_px"] = entry

    # Outcome from the CONFIRMED entry price.
    tgt = entry * (1 + TARGET_PCT / 100)
    stp = entry * (1 - STOP_PCT / 100)
    hit_t = r["h"] >= tgt
    hit_s = r["l"] <= stp
    # stop-first when both are in range
    r["conf_ret"] = ((r["c"] - entry) / entry * 100)                      # ride to close
    # When BOTH target and stop sit inside the day's range, daily bars cannot
    # say which came first. Report the BRACKET rather than one arbitrary
    # choice: stop-first is the floor, target-first the ceiling, and the truth
    # is between. A single tie-break would otherwise masquerade as a result --
    # the first run of this script printed t=-26.68 purely because stop-first
    # assigned -3% to every volatile name that merely traded through it.
    r["both_in_range"] = hit_t & hit_s
    r["conf_ret_pess"] = r["conf_ret"]
    r.loc[hit_t & ~hit_s, "conf_ret_pess"] = TARGET_PCT
    r.loc[hit_s, "conf_ret_pess"] = -STOP_PCT
    r["conf_ret_opt"] = r["conf_ret"]
    r.loc[hit_s & ~hit_t, "conf_ret_opt"] = -STOP_PCT
    r.loc[hit_t, "conf_ret_opt"] = TARGET_PCT

    conf = r[r.confirmed].copy()
    for col in ("conf_ret", "conf_ret_pess", "conf_ret_opt"):
        conf[col + "_ex"] = conf[col] - conf.groupby("date")[col].transform("mean")

    cand = conf[(conf.composite_score >= SCORE_MIN) & (conf.atr_pct >= MIN_ATR_PCT)]
    cand_atr = cand[cand.atr_mult >= ATR_REQ]

    print(f"confirmation rate (whole universe): {r.confirmed.mean() * 100:.1f}% of ticker-days")
    print(f"confirmation rate (score>=75):      "
          f"{r[(r.composite_score >= SCORE_MIN) & (r.atr_pct >= MIN_ATR_PCT)].confirmed.mean() * 100:.1f}%\n")

    print(f"  {'population':<34}{'rows':>8}{'days':>7}{'raw%':>9}{'excess%':>10}{'t':>8}{'net of fee':>12}")
    for label, sub in (("all confirmed ticker-days", conf),
                       ("confirmed & score>=75", cand),
                       (f"confirmed & score>=75 & atr>={ATR_REQ}", cand_atr)):
        for tag, col in (("  ride to close", "conf_ret"),
                         ("  tgt/stop pessimistic", "conf_ret_pess"),
                         ("  tgt/stop optimistic", "conf_ret_opt")):
            m_raw, _, _, _ = clustered(sub, col)
            m_ex, t_ex, nd, nr = clustered(sub, col + "_ex")
            print(f"  {label + tag:<34}{nr:>8}{nd:>7}{m_raw:>9.3f}{m_ex:>10.3f}{t_ex:>8.2f}"
                  f"{m_ex - RT_FEE_PCT:>12.3f}")

    print(f"\n  ties (target AND stop both in range): "
          f"{conf.both_in_range.mean() * 100:.1f}% of confirmed rows -- the spread between")
    print("  the pessimistic and optimistic rows above is entirely this ambiguity.")

    print("\n--- NOT a valid test of the gate (kept as a warning) ---")
    pool = r[(r.composite_score >= SCORE_MIN) & (r.atr_pct >= MIN_ATR_PCT)].copy()
    pool["oc"] = (pool["c"] - pool["o"]) / pool["o"] * 100
    pool["oc_ex"] = pool["oc"] - pool.groupby("date")["oc"].transform("mean")
    yes = pool[pool.confirmed].groupby("date")["oc_ex"].mean()
    no = pool[~pool.confirmed].groupby("date")["oc_ex"].mean()
    pair = pd.concat([yes.rename("conf"), no.rename("unconf")], axis=1, sort=False).dropna()
    d = pair["conf"] - pair["unconf"]
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    print(f"  'confirmed minus unconfirmed' open->close excess: {d.mean():+.3f}pp t={t:+.2f}")
    print("  This is TAUTOLOGICAL, not an edge: confirmation is defined by the SAME day's")
    print("  high, so it selects on an outcome-correlated event. A name that never traded")
    print("  +0.35% above its open has almost by construction closed red. The only")
    print("  legitimate question is the FORWARD return from the confirmation price,")
    print("  which is the 'ride to close' row above -- and that excess is ~0.")

    print(f"\n--- verdict against cost (${LIVE_POS:.0f} position, {RT_FEE_PCT:.2f}% round trip) ---")
    m_ex, t_ex, _, _ = clustered(cand, "conf_ret_ex")
    print(f"  forward excess from the confirmed entry = {m_ex:+.3f}pp   fee = {RT_FEE_PCT:.2f}pp")
    print(f"  net = {m_ex - RT_FEE_PCT:+.3f}pp per trade")
    need = RT_FEE_PCT
    print(f"  the gate would need +{need:.2f}pp of excess just to break even; "
          f"measured {m_ex:+.3f}pp (t={t_ex:.2f})")


if __name__ == "__main__":
    main()
