"""Phase 5: do FEATURE INTERACTIONS carry direction that single features miss?

Every earlier test screened features one at a time. This is the remaining
possibility: that direction lives in a combination -- e.g. "a weak opener ONLY
in a wide opening range ONLY when the market is strong".

THIS IS THE EASIEST PLACE IN THE WHOLE PROJECT TO FOOL ONESELF. With ~1,200 rows
and ~15 features a gradient-booster will fit the noise perfectly and report a
beautiful in-sample number. Three guards, all non-negotiable:

  1. OUT-OF-FOLD ONLY. Every number reported comes from data the model did not
     see. In-sample scores are printed alongside purely to show the gap.
  2. TIME-SERIES CV, never random K-fold. Random folds leak: same-session rows
     land on both sides of the split and same-day candidates are ONE event.
     Folds are contiguous date blocks.
  3. A SHUFFLED-LABEL CONTROL. The identical pipeline is run with the target
     randomly permuted. Whatever score that produces is the floor -- it is what
     this procedure yields from pure noise. A real signal must clear it clearly.

Target: the market-adjusted excess (same-day 09:45 -> close), long only. Scored
in the only currency that matters: mean excess of the names the model ranks in
its top tercile, against the 0.473pp fee.

    ../../venv/bin/python daytrader_research/ml_combination_study.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from micro_features import FEATURES as MICRO, SPLIT, FEE_PP, load  # noqa: E402

N_FOLDS = 5
SEED = 0


def clustered(df: pd.DataFrame, col: str) -> tuple:
    if len(df) < 2:
        return float("nan"), float("nan")
    d = df.groupby("date")[col].mean()
    if len(d) < 2:
        return d.mean(), float("nan")
    return d.mean(), d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))


def run(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, label: str,
        shuffle: bool = False) -> tuple:
    """Contiguous-date-block CV. Returns (top-tercile excess, t, in-sample excess)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(SEED)
    yy = y.values.copy()
    if shuffle:
        rng.shuffle(yy)

    dates = np.array(sorted(meta["date"].unique()))
    blocks = np.array_split(dates, N_FOLDS)
    oof = pd.Series(np.nan, index=meta.index)
    ins = []
    for i in range(1, len(blocks)):                    # expanding window, no lookahead
        tr_dates = np.concatenate(blocks[:i])
        te_dates = blocks[i]
        tr = meta["date"].isin(tr_dates).values
        te = meta["date"].isin(te_dates).values
        if tr.sum() < 150 or te.sum() < 40:
            continue
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=150,
                                          learning_rate=0.05, random_state=SEED)
        m.fit(X[tr], yy[tr])
        oof[te] = m.predict(X[te])
        ins.append(np.corrcoef(m.predict(X[tr]), yy[tr])[0, 1])

    d = meta.copy()
    d["pred"] = oof.values
    d["y"] = yy
    d = d.dropna(subset=["pred"])
    if d.empty:
        return float("nan"), float("nan"), float("nan")
    d["q"] = d.groupby("date")["pred"].transform(
        lambda x: pd.qcut(x, 3, labels=False, duplicates="drop")
        if x.notna().sum() >= 3 else np.nan)
    d = d.dropna(subset=["q"])
    top = d[d.q == d.q.max()]
    m_top, t_top = clustered(top.assign(_y=top["y"]), "_y")
    ic = np.corrcoef(d["pred"], d["y"])[0, 1] if len(d) > 5 else float("nan")
    print(f"  {label:<34}{len(d):>7}{m_top:>11.3f}{t_top:>8.2f}"
          f"{ic:>9.3f}{np.mean(ins) if ins else float('nan'):>12.3f}")
    return m_top, t_top, ic


def main() -> None:
    r = load()
    tr = r[r["date"] < SPLIT].dropna(subset=MICRO + ["excess"]).copy()
    print(f"EXPLORATION only: {len(tr):,} rows, {tr.date.nunique()} sessions, "
          f"{len(MICRO)} features")
    print(f"scored on the model's TOP TERCILE excess vs the {FEE_PP:.3f}pp fee")
    print(f"{N_FOLDS}-block expanding-window CV; all numbers OUT-OF-FOLD\n")
    print(f"  {'model':<34}{'rows':>7}{'top excess':>11}{'t':>8}{'OOF IC':>9}"
          f"{'in-samp IC':>12}")

    X = tr[MICRO].astype(float)
    y = tr["excess"]
    meta = tr[["date", "ticker"]].copy()

    real = run(X, y, meta, "gradient boosting, 11 micro feats")
    ctrl = run(X, y, meta, "SHUFFLED-LABEL control", shuffle=True)

    # a linear model too -- less prone to fitting noise on this sample size
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    def run_ridge(shuffle=False):
        rng = np.random.default_rng(SEED)
        yy = y.values.copy()
        if shuffle:
            rng.shuffle(yy)
        dates = np.array(sorted(meta["date"].unique()))
        blocks = np.array_split(dates, N_FOLDS)
        oof = pd.Series(np.nan, index=meta.index)
        for i in range(1, len(blocks)):
            trm = meta["date"].isin(np.concatenate(blocks[:i])).values
            tem = meta["date"].isin(blocks[i]).values
            if trm.sum() < 150 or tem.sum() < 40:
                continue
            sc = StandardScaler().fit(X[trm])
            mdl = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(X[trm]), yy[trm])
            oof[tem] = mdl.predict(sc.transform(X[tem]))
        d = meta.copy(); d["pred"] = oof.values; d["y"] = yy
        d = d.dropna(subset=["pred"])
        d["q"] = d.groupby("date")["pred"].transform(
            lambda x: pd.qcut(x, 3, labels=False, duplicates="drop")
            if x.notna().sum() >= 3 else np.nan)
        d = d.dropna(subset=["q"])
        top = d[d.q == d.q.max()]
        m, t = clustered(top.assign(_y=top["y"]), "_y")
        ic = np.corrcoef(d["pred"], d["y"])[0, 1]
        print(f"  {'ridge (linear), 11 micro feats' if not shuffle else 'ridge SHUFFLED control':<34}"
              f"{len(d):>7}{m:>11.3f}{t:>8.2f}{ic:>9.3f}{float('nan'):>12}")
        return m, t

    rid = run_ridge(False)
    ridc = run_ridge(True)

    print("\n=== VERDICT ===")
    print(f"  fee hurdle: {FEE_PP:.3f}pp")
    for nm, (m, t) in (("gradient boosting", (real[0], real[1])),
                       ("ridge", rid)):
        ok = (m > FEE_PP) and (t > 2)
        print(f"  {nm:<20}top-tercile {m:+.3f}pp (t {t:+.2f})  "
              f"{'CLEARS fee with t>2' if ok else 'does not clear'}")
    print(f"  shuffled controls: GB {ctrl[0]:+.3f}pp (t {ctrl[1]:+.2f}), "
          f"ridge {ridc[0]:+.3f}pp (t {ridc[1]:+.2f})")
    print("  A real combination must beat BOTH the fee and its own shuffled control.")


if __name__ == "__main__":
    main()
