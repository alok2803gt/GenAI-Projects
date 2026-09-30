"""
SPY 0DTE option-return experiment.

Question: given the price the market is charging for a specific 0DTE
structure at time t, is its forward net P&L predictable from what is
observable at t?  The option trade itself is the prediction target; there is
no regime classifier and no barrier label.

Timing, per candidate
    t_s   signal bar (decision bars 10:00, 10:15, ..., 14:30). Features use SPY
          bars up to and including t_s and option prints in minutes <= t_s.
    t_e   t_s + 1 minute. Entry price = the contract's 1-min VWAP in minute t_e.
    exit  long options: t_e + 30 min, t_e + 60 min, or 15:45; spreads: 15:45.

Prices are traded 1-min VWAPs (Alpaca) used as the mid, plus a MODELED bid/ask
cost (QuoteModel, lambda 0.25) and $0.65/contract commission. A price is used
only if the contract printed in that minute or within `max_stale` minutes
before it; otherwise that P&L is missing -- never filled in by a model.

Contracts come from the contemporaneous chain (review item 7): ATM implied
vol from the nearest-strike prints at t_s sets the expected-move unit
u = S * iv * sqrt(T); targets are 0 / 0.5 / 1.0 u out of the money, and the
listed strike nearest the target that printed near t_s is used.

Models (walk-forward by calendar month, expanding window, 1-session embargo,
training rows strictly from earlier sessions):
    constant  mean training P&L of that exact structure
    ridge     standardized features + structure dummies
    lgbm      small regularized LightGBM, time-ordered early stopping
LightGBM only counts if it beats ridge out of sample. The success criterion
is economic: realized P&L should rise monotonically across predicted-EV
deciles, and the trade rule's P&L should beat the constant model's.

    ./.venv/bin/python option_returns.py                 # build candidates (cached) + report
    ./.venv/bin/python option_returns.py --rebuild       # rebuild the candidate file
"""
import argparse
import logging
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import spy0dte_framework as F

log = logging.getLogger("optret")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

DECISION_TAUS = range(30, 301, 15)          # 10:00 .. 14:30 signal bars
LONG_TARGETS = (0.0, 0.5)                   # OTM distance in expected-move units
LONG_HORIZONS = (30, 60, "eod")
SPREAD_TARGETS = (0.5, 1.0)
SPREAD_WIDTH = 3.0
EOD_BEFORE_CLOSE = 15
LAMBDA = 0.25
COMMISSION = 0.65
MAX_STALE = 2                               # minutes

UNDERLYING_FEATURES = F.FeatureBuilder.MODEL_FEATURES + ["rv30", "ret15", "ret_open", "vwap_dist", "vshock"]
OPTION_FEATURES = ["atm_iv", "atm_iv_chg15", "iv_rv", "skew", "dist_u", "leg_iv_rel", "prem_rel",
                   "leg_vol15", "leg_mom15", "hold_min", "is_call", "target_u"]
FEATURES = UNDERLYING_FEATURES + OPTION_FEATURES


# =============================================================================
# 1. candidates
# =============================================================================

def add_underlying_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Extra causal SPY features (all use bars <= t)."""
    f = feats.copy()
    C, V, sess = f["close"], f["volume"], f["session"]
    f["ret15"] = np.log(C / C.groupby(sess).shift(15))
    f["ret_open"] = np.log(C / f["open"].groupby(sess).transform("first"))
    f["vwap_dist"] = (C - f["vwap"]) / f["atr14"]
    base = V.rolling(F.REGULAR_SESSION_BARS * 10, min_periods=F.REGULAR_SESSION_BARS).mean().shift(15) * 15
    f["vshock"] = np.log(V.rolling(15, min_periods=15).sum() / base)
    return f


class DayChain:
    """One session's option prints as per-minute arrays (minute 0 = 09:30)."""

    def __init__(self, df: pd.DataFrame, session: pd.Timestamp, n_min: int):
        self.px, self.vol = {}, {}
        if df.empty:
            return
        m = ((df.index - (session + pd.Timedelta(hours=9, minutes=30))) / pd.Timedelta(minutes=1)).astype(int)
        d = pd.DataFrame(dict(symbol=df["symbol"].values, m=np.asarray(m),
                              px=np.where(df["vwap"].notna(), df["vwap"], df["close"]),
                              v=df["volume"].values))
        d = d[(d["m"] >= 0) & (d["m"] < n_min) & (d["px"] > 0)]
        for sym, g in d.groupby("symbol"):
            key = (sym[9], int(sym[10:]) / 1000.0)
            a, v = np.full(n_min, np.nan), np.zeros(n_min)
            a[g["m"].values], v[g["m"].values] = g["px"].values, g["v"].values
            self.px[key], self.vol[key] = a, v
        self.strikes = {r: np.array(sorted(k for (rr, k) in self.px if rr == r)) for r in ("C", "P")}

    def price(self, right, strike, m, max_stale=MAX_STALE):
        """(price, staleness_minutes) from prints in minutes m-max_stale .. m, else (nan, nan)."""
        a = self.px.get((right, strike))
        if a is None or m < 0 or m >= len(a):
            return float("nan"), float("nan")
        for j in range(max_stale + 1):
            if m - j >= 0 and np.isfinite(a[m - j]):
                return float(a[m - j]), float(j)
        return float("nan"), float("nan")

    def exit_price(self, right, strike, m, forward=10):
        """Exit: a print in m-MAX_STALE..m, else the FIRST later print within
        `forward` minutes (a real, executable-later price -- never a model).
        Staleness is negative when the fill came after m."""
        p, st = self.price(right, strike, m)
        if np.isfinite(p):
            return p, st
        a = self.px.get((right, strike))
        if a is not None:
            for j in range(1, forward + 1):
                if m + j < len(a) and np.isfinite(a[m + j]):
                    return float(a[m + j]), float(-j)
        return float("nan"), float("nan")

    def nearest_printed(self, right, target, m):
        """Listed strike nearest `target` with a print near minute m."""
        ks = self.strikes.get(right, np.array([]))
        for k in ks[np.argsort(np.abs(ks - target))][:6]:
            if np.isfinite(self.price(right, k, m)[0]):
                return float(k)
        return None

    def iv(self, right, strike, m, S, mins_left):
        p = self.price(right, strike, m)[0]
        return F.implied_vol(p, S, strike, mins_left, right) if np.isfinite(p) else float("nan")

    def atm_iv(self, m, S, mins_left):
        vals = []
        for r in ("C", "P"):
            k = self.nearest_printed(r, S, m)
            if k is not None:
                vals.append(self.iv(r, k, m, S, mins_left))
        vals = [v for v in vals if np.isfinite(v) and v > 0]
        return float(np.mean(vals)) if vals else float("nan")


def build_day(chain: DayChain, day: pd.DataFrame, q: F.QuoteModel, taus=DECISION_TAUS) -> list:
    """All candidates for one session. `day` = that session's feature rows, indexed by tau."""
    rows = []
    n = int(day["session_len"].iloc[0])
    eod = n - EOD_BEFORE_CLOSE
    for tau in taus:
        if tau + 1 >= eod - 5 or tau not in day.index or not bool(day.at[tau, "ready"]):
            continue
        r = day.loc[tau]
        S, T = float(r["close"]), n - tau - 1                     # minutes left at t_s close
        iv0 = chain.atm_iv(tau, S, T)
        if not (np.isfinite(iv0) and iv0 > 0):
            continue
        u = S * iv0 * math.sqrt(T / F.MIN_PER_YEAR)
        S15 = float(day.at[tau - 15, "close"]) if (tau - 15) in day.index else float("nan")
        iv15 = chain.atm_iv(tau - 15, S15, T + 15) if np.isfinite(S15) else float("nan")
        common = {c: float(r[c]) for c in UNDERLYING_FEATURES}
        common.update(session=r["session"], tau=tau, S=S, u=u, atm_iv=iv0, atm_iv_chg15=iv0 - iv15,
                      iv_rv=math.log(iv0 / r["rv30"]) if r["rv30"] > 0 else float("nan"))
        kp, kc = chain.nearest_printed("P", S - 0.5 * u, tau), chain.nearest_printed("C", S + 0.5 * u, tau)
        common["skew"] = ((chain.iv("P", kp, tau, S, T) if kp else np.nan) -
                          (chain.iv("C", kc, tau, S, T) if kc else np.nan))
        te, T_e = tau + 1, n - tau - 1                            # fill minute; minutes left at its open

        def leg_feats(right, k):
            p_s = chain.price(right, k, tau)[0]
            p_15 = chain.price(right, k, tau - 15)[0]
            liv = chain.iv(right, k, tau, S, T)
            return dict(leg_iv_rel=liv / iv0 if np.isfinite(liv) else np.nan,
                        leg_vol15=float(np.log1p(chain.vol[(right, k)][max(0, tau - 14):tau + 1].sum())),
                        leg_mom15=math.log(p_s / p_15) if np.isfinite(p_15) and p_15 > 0 else np.nan,
                        dist_u=(k - S) / u * (1 if right == "C" else -1), p_s=p_s)

        seen = set()
        for right in ("C", "P"):
            sgn = 1 if right == "C" else -1
            for d in LONG_TARGETS:
                k = chain.nearest_printed(right, S + sgn * d * u, tau)
                if k is None:
                    continue
                lf = leg_feats(right, k)
                for H in LONG_HORIZONS:
                    tx = eod if H == "eod" else te + H
                    if tx > eod or ("L", right, k, tx) in seen:
                        continue
                    seen.add(("L", right, k, tx))
                    e_mid, e_st = chain.price(right, k, te)
                    x_mid, x_st = chain.exit_price(right, k, tx)
                    if not np.isfinite(e_mid):
                        continue
                    cost = q.buy(e_mid, T_e)
                    comm = 2 * COMMISSION
                    pnl = (q.sell(x_mid, n - tx) - cost) * 100 - comm if np.isfinite(x_mid) else np.nan
                    rows.append(dict(common, family="long", struct=f"long_{right}_d{d}_H{H}", is_call=float(right == "C"),
                                     target_u=d, hold_min=float(tx - te), strikes=f"{k:g}{right}",
                                     prem_rel=lf["p_s"] / u, **{k_: lf[k_] for k_ in ("leg_iv_rel", "leg_vol15", "leg_mom15", "dist_u")},
                                     entry_mid=e_mid, exit_mid=x_mid, entry_stale=e_st, exit_stale=x_st,
                                     leg1_entry=e_mid, leg1_exit=x_mid, leg2_entry=np.nan, leg2_exit=np.nan,
                                     T_entry=float(T_e), T_exit=float(n - tx),
                                     pnl=pnl, pnl_worst=pnl if np.isfinite(pnl) else -cost * 100 - comm))
            for d in SPREAD_TARGETS:
                ks = chain.nearest_printed(right, S + sgn * d * u, tau)
                if ks is None:
                    continue
                kw = ks + sgn * SPREAD_WIDTH
                if ("S", right, ks) in seen or not np.isfinite(chain.price(right, kw, tau)[0]):
                    continue
                seen.add(("S", right, ks))
                lf = leg_feats(right, ks)
                es, ess = chain.price(right, ks, te)
                ew, ews = chain.price(right, kw, te)
                if not (np.isfinite(es) and np.isfinite(ew)):
                    continue
                credit = q.sell(es, T_e) - q.buy(ew, T_e)
                if credit <= 0:
                    continue
                xs, xss = chain.exit_price(right, ks, eod)
                xw, xws = chain.exit_price(right, kw, eod)
                comm = 4 * COMMISSION
                ok = np.isfinite(xs) and np.isfinite(xw)
                pnl = (credit - (q.buy(xs, n - eod) - q.sell(xw, n - eod))) * 100 - comm if ok else np.nan
                ps_w = chain.price(right, kw, tau)[0]
                rows.append(dict(common, family="spread", struct=f"spread_{right}_d{d}", is_call=float(right == "C"),
                                 target_u=d, hold_min=float(eod - te), strikes=f"{ks:g}/{kw:g}{right}",
                                 prem_rel=(lf["p_s"] - ps_w) / SPREAD_WIDTH,
                                 **{k_: lf[k_] for k_ in ("leg_iv_rel", "leg_vol15", "leg_mom15", "dist_u")},
                                 entry_mid=es - ew, exit_mid=(xs - xw) if ok else np.nan,
                                 entry_stale=max(ess, ews), exit_stale=max(xss, xws) if ok else np.nan,
                                 leg1_entry=es, leg1_exit=xs, leg2_entry=ew, leg2_exit=xw,
                                 T_entry=float(T_e), T_exit=float(n - eod),
                                 pnl=pnl, pnl_worst=pnl if ok else (credit - SPREAD_WIDTH) * 100 - comm))
    return rows


def build_candidates(feats: pd.DataFrame, option_dir: Path, holdout_start: str,
                      taus=DECISION_TAUS) -> pd.DataFrame:
    hold = pd.Timestamp(holdout_start, tz=F.NY)
    q = F.QuoteModel(lam=LAMBDA)
    rows = []
    files = sorted(option_dir.glob("*.parquet"))
    by_sess = {s: g.set_index("tau", drop=False).rename_axis("tau_idx") for s, g in feats.groupby("session")}
    for i, p in enumerate(files):
        session = pd.Timestamp(p.stem, tz=F.NY)
        if session >= hold:                                  # holdout stays locked
            continue
        day = by_sess.get(session)
        if day is None:
            continue
        chain = DayChain(pd.read_parquet(p), session, int(day["session_len"].iloc[0]))
        if chain.px:
            rows += build_day(chain, day, q, taus)
        if (i + 1) % 100 == 0:
            log.info("%d/%d sessions, %d candidates", i + 1, len(files), len(rows))
    out = pd.DataFrame(rows)
    assert (out["session"] < hold).all(), "holdout session in candidate set"
    return out


# =============================================================================
# 2. walk-forward models
# =============================================================================

def _winsor(y, lo, hi):
    return np.clip(y, lo, hi)


def fit_predict_fold(train: pd.DataFrame, test: pd.DataFrame, winsor: bool = True) -> dict:
    from lightgbm import LGBMRegressor, early_stopping
    from sklearn.linear_model import Ridge

    out = {}
    const = train.groupby("struct")["pnl"].mean()
    out["constant"] = test["struct"].map(const).fillna(train["pnl"].mean()).values

    y = train["pnl"].values
    lo, hi = np.percentile(y, [1, 99])
    yw = _winsor(y, lo, hi) if winsor else y      # raw target = tail-risk ablation
    cut = int(len(train) * 0.85)                              # time-ordered validation split

    med = train[FEATURES].median()
    structs = sorted(train["struct"].unique())

    def design(df):
        X = df[FEATURES].fillna(med)
        X = pd.concat([X, pd.get_dummies(df["struct"]).reindex(columns=structs, fill_value=0).astype(float)], axis=1)
        return X.values.astype(float)

    Xtr, Xte = design(train), design(test)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    best = None
    for a in (10.0, 100.0, 1000.0, 10000.0):
        m = Ridge(alpha=a).fit(Ztr[:cut], yw[:cut])
        err = np.mean((m.predict(Ztr[cut:]) - y[cut:]) ** 2)
        best = (err, a) if best is None or err < best[0] else best
    out["ridge"] = Ridge(alpha=best[1]).fit(Ztr, yw).predict(Zte)

    Xg = train[FEATURES].values
    lgbm = LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=300,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, verbose=-1)
    lgbm.fit(Xg[:cut], yw[:cut], eval_set=[(Xg[cut:], y[cut:])], callbacks=[early_stopping(100, verbose=False)])
    out["lgbm"] = lgbm.predict(test[FEATURES].values, num_iteration=lgbm.best_iteration_)
    out["_meta"] = dict(ridge_alpha=best[1], lgbm_trees=int(lgbm.best_iteration_ or 0))
    return out


def walk_forward(c: pd.DataFrame, min_train_sessions: int = 120, embargo: int = 1, winsor: bool = True):
    c = c.sort_values(["session", "tau"]).reset_index(drop=True)
    sessions = list(c["session"].drop_duplicates().sort_values())
    months = pd.PeriodIndex([s.tz_localize(None) for s in sessions], freq="M")
    row_month = c["session"].dt.tz_localize(None).dt.to_period("M").values
    preds = {m: np.full(len(c), np.nan) for m in ("constant", "ridge", "lgbm")}
    folds = []
    for month in months.unique():
        first = np.where(months == month)[0][0]
        if first - embargo < min_train_sessions:
            continue
        train_end = sessions[first - embargo]                  # train on sessions < this (1-session gap)
        test_mask = row_month == month
        for fam in ("long", "spread"):
            tr = c[(c["session"] < train_end) & (c["family"] == fam) & c["pnl"].notna()]
            te_idx = np.where(test_mask & (c["family"] == fam).values)[0]
            if len(te_idx) == 0 or len(tr) < 1000:
                continue
            p = fit_predict_fold(tr, c.iloc[te_idx], winsor=winsor)
            for m in preds:
                preds[m][te_idx] = p[m]
            folds.append(dict(month=str(month), family=fam, n_train=len(tr), n_test=len(te_idx), **p["_meta"]))
    for m, v in preds.items():
        c[f"ev_{m}"] = v
    return c, pd.DataFrame(folds)


# =============================================================================
# 3. evaluation
# =============================================================================

def cluster_mean_se(x: pd.Series, groups: pd.Series):
    """Mean and session-clustered standard error."""
    x = x.astype(float)
    n = len(x)
    if n < 2:
        return x.mean(), float("nan")
    resid = (x - x.mean()).groupby(groups.values).sum()
    g = len(resid)
    return x.mean(), math.sqrt((resid ** 2).sum() * g / max(g - 1, 1)) / n


def scorecard(oos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, g in oos.groupby("family"):
        g = g[g["pnl"].notna()]
        e0 = (g["pnl"] - g["ev_constant"]) ** 2
        for m in ("constant", "ridge", "lgbm"):
            e = (g["pnl"] - g[f"ev_{m}"]) ** 2
            gain = (e0 - e).groupby(g["session"].values).mean()
            rows.append(dict(family=fam, model=m, n=len(g), sessions=g["session"].nunique(),
                             spearman=g[f"ev_{m}"].corr(g["pnl"], method="spearman"),
                             r2_vs_constant=1 - e.sum() / e0.sum(),
                             t_vs_constant=gain.mean() / (gain.std() / math.sqrt(len(gain))) if m != "constant" else np.nan))
    return pd.DataFrame(rows)


def decile_table(oos: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for fam, g in oos.groupby("family"):
        g = g[g["pnl"].notna()].copy()
        g["decile"] = pd.qcut(g[f"ev_{model}"].rank(method="first"), 10, labels=False) + 1
        for d, h in g.groupby("decile"):
            mean, se = cluster_mean_se(h["pnl"], h["session"])
            rows.append(dict(family=fam, decile=d, n=len(h), pred_ev=h[f"ev_{model}"].mean(), realized=mean, se=se))
    t = pd.DataFrame(rows)
    return t


def trade_rule(oos: pd.DataFrame, model: str, threshold: float, one_per_day: bool) -> dict:
    """Best predicted candidate per decision bar; trade it if EV > threshold.
    Candidates whose exit never printed are booked at their worst case."""
    g = oos[oos[f"ev_{model}"].notna()]
    best = g.loc[g.groupby(["session", "tau"])[f"ev_{model}"].idxmax()]
    best = best[best[f"ev_{model}"] > threshold]
    if one_per_day:
        best = best.sort_values("tau").groupby("session").head(1)
    pnl = best["pnl"].where(best["pnl"].notna(), best["pnl_worst"])
    daily = pnl.groupby(best["session"]).sum().reindex(pd.Index(oos["session"].unique()), fill_value=0.0)
    return dict(model=model, threshold=threshold, rule="1/day" if one_per_day else "every bar", trades=len(best),
                missing_exit=int(best["pnl"].isna().sum()), win=(pnl > 0).mean() if len(pnl) else np.nan,
                avg_usd=pnl.mean(), total_usd=pnl.sum(), total_ex_missing=best["pnl"].sum(),
                avg_pred=best[f"ev_{model}"].mean(),
                t_daily=daily.mean() / (daily.std() / math.sqrt(len(daily))) if daily.std() > 0 else np.nan,
                mix=best["struct"].str.extract(r"^(long_[CP]|spread_[CP])")[0].value_counts().to_dict())


# =============================================================================
# 4. report
# =============================================================================

def main():
    HOLDOUT_START = F.HOLDOUT_START
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default="data/SPY_1min_sip.parquet")
    ap.add_argument("--option-bars", default="data/SPY_0dte_option_bars")
    ap.add_argument("--cache", default="data/optret_candidates.parquet")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pd.set_option("display.width", 200)
    t0 = time.time()

    cache = Path(args.cache)
    if cache.exists() and not args.rebuild:
        c = pd.read_parquet(cache)
    else:
        bars = F.prepare_bars(pd.read_parquet(args.bars)[["open", "high", "low", "close", "volume"]])
        feats = add_underlying_features(F.FeatureBuilder().transform(bars))
        c = build_candidates(feats, Path(args.option_bars), HOLDOUT_START)
        c.to_parquet(cache)
    hold = pd.Timestamp(HOLDOUT_START, tz=F.NY)
    assert (c["session"] < hold).all(), "holdout leaked into the candidate set"
    print(f"DEVELOPMENT ONLY: sessions {c.session.min().date()} .. {c.session.max().date()} "
          f"({c.session.nunique()}), holdout from {HOLDOUT_START} locked. Built in {time.time() - t0:.0f}s")
    print("prices: traded 1-min VWAP as mid + modeled bid/ask cost (lambda 0.25) + $0.65/contract\n")

    print("1) CANDIDATES -- unconditional real P&L per structure (every candidate taken, $/contract)")
    u = c.groupby("struct").agg(n=("pnl", "size"), exit_missing=("pnl", lambda s: s.isna().mean()),
                                win=("pnl", lambda s: (s.dropna() > 0).mean()), avg_usd=("pnl", "mean"),
                                median_usd=("pnl", "median"), entry_stale_share=("entry_stale", lambda s: (s > 0).mean()),
                                exit_late_share=("exit_stale", lambda s: (s < 0).mean()))
    u["t_clustered"] = [(lambda m, se: m / se)(*cluster_mean_se(g["pnl"].dropna(), g.loc[g["pnl"].notna(), "session"]))
                        for _, g in c.groupby("struct")]
    print(u.round(3).to_string())

    oos, folds = walk_forward(c)
    oos = oos[oos["ev_constant"].notna()]
    print(f"\n2) WALK-FORWARD: {folds.month.nunique()} monthly folds, OOS {oos.session.min().date()} .. "
          f"{oos.session.max().date()} ({oos.session.nunique()} sessions, {len(oos):,} candidates); "
          f"median LightGBM trees {int(folds.lgbm_trees.median())}, ridge alpha {folds.ridge_alpha.mode()[0]:g}")
    print("   r2_vs_constant > 0 and t_vs_constant > 2 = beats the structure-mean baseline")
    print(scorecard(oos).round(4).to_string(index=False))

    print("\n3) PREDICTED-EV DECILES vs REALIZED P&L ($/contract, session-clustered se) -- want monotonic")
    for m in ("ridge", "lgbm"):
        t = decile_table(oos, m)
        for fam, g in t.groupby("family"):
            rho = g["decile"].corr(g["realized"], method="spearman")
            print(f"\n   {m} / {fam}   (rank corr decile vs realized = {rho:.2f})")
            print(g.drop(columns="family").round(2).to_string(index=False))

    print("\n4) TRADE RULE -- best predicted candidate per decision bar, trade if EV > threshold")
    rows = [trade_rule(oos, m, th, one) for m in ("constant", "ridge", "lgbm") for th in (0.0, 10.0, 25.0)
            for one in (False, True)]
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    oos.to_parquet("data/optret_oos_predictions.parquet")
    folds.to_csv("data/optret_folds.csv", index=False)
    print(f"\nper-candidate OOS predictions: data/optret_oos_predictions.parquet   done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
