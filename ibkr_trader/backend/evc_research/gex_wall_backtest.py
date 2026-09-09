"""
Real historical backtest: does a condor's short strike sitting on a large
negative-gamma wall (or an extreme put/call OI ratio) actually predict a
higher breach rate? Built 2026-08-25 after finding this pattern once, live,
on INTU -- one data point is not evidence, this checks it against many real
historical earnings events instead.

Data sources (all real, all confirmed working against this account's own
credentials in scanner_config.json):
  - CBOE DataShop (api.livevol.com) -- real historical per-strike open
    interest + gamma/delta for the FULL chain, any single trading day.
    Confirmed working for individual equities (not just SPY/index) via a
    live DKS 2026-08-24 test: real underlying_close=$179.33, 550 contracts,
    real open_interest/gamma per strike.
  - yfinance -- real historical earnings dates + realized closing prices
    (same method already used in _evc_pretrade_review's own historical
    containment check).

Method, mirroring EVC's OWN live construction exactly so this tests the
real strategy, not a proxy:
  1. For each ticker's real past earnings date, fetch the CBOE chain for
     the nearest prior trading day (GEX/IV is an end-of-day figure, same
     reasoning fetch_cboe_oi_history.py already established for SPY).
  2. From that REAL chain: find the nearest expiry >= the day after
     earnings (matches _evc_get_chain_params's own "nearest future expiry"
     logic), get the ATM strike, compute EM = ATM call mid + ATM put mid
     (matches _evc_quote_condor exactly), then short_put/short_call at
     1.0x EM and long_put/long_call at 1.5x EM (matches wing_mult=1.5
     default -- NOT tonight's one-off put_cushion_mult=1.3, since that's
     unvalidated and this backtest is what would validate it).
  3. Real GEX wall + real put/call OI ratio computed directly from the
     SAME chain response (no extra API calls needed -- gamma/OI for every
     strike came back in one request).
  4. Real realized move: the actual next-trading-day close (yfinance),
     checked against the short strikes for a real breach/no-breach outcome.

Honest limitations stated up front:
  - Universe is EVC's own real CANDIDATE_POOL, last N earnings per ticker
    (bounded to keep runtime/API load reasonable) -- not necessarily
    representative of every possible future EVC candidate.
  - "Day before earnings" sometimes lands on a low-volume pre-holiday
    session or a day CBOE has no data for -- those are skipped and counted,
    not silently dropped.
  - This tests wing_mult=1.5 / 1.0x-EM-short-strike construction only
    (today's live default), not tonight's put_cushion_mult=1.3 override.
"""
import base64
import json
import sys
import time
from datetime import date, datetime, timedelta

import requests
import yfinance as yf

HERE_CFG = "scanner_config.json"
OUT_FILE = "evc_research/gex_wall_backtest_results.json"
LOG_FILE = "evc_research/gex_wall_backtest_log.txt"
N_EARNINGS_PER_TICKER = 4      # bounded -- keep total API load reasonable
MAX_TICKERS = None             # set via CLI arg for a quick pilot run


def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get_token(cfg):
    basic = base64.b64encode(f"{cfg['cboe_datashop_client_id']}:{cfg['cboe_datashop_client_secret']}".encode()).decode()
    resp = requests.post(
        "https://id.livevol.com/connect/token",
        headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials"}, timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_chain(token, symbol, date_str):
    resp = requests.get(
        "https://api.livevol.com/v1/delayed/allaccess/market/option-and-underlying-quotes",
        headers={"Authorization": f"Bearer {token}"},
        params={"symbol": symbol, "date": date_str}, timeout=60,
    )
    if resp.status_code == 204:
        return None
    if resp.status_code != 200:
        return None
    return resp.json()


def prior_trading_day_str(d: date) -> str:
    prior = d - timedelta(days=1)
    while prior.weekday() >= 5:
        prior -= timedelta(days=1)
    return prior.strftime("%Y-%m-%d")


def analyze_one(token, ticker: str, earnings_date, hist) -> dict | None:
    """Real analysis for one (ticker, earnings_date) pair. Returns None if
    real data wasn't available (skipped, not silently dropped by the caller)."""
    ed_naive = earnings_date.tz_localize(None) if earnings_date.tzinfo else earnings_date
    d_before = prior_trading_day_str(ed_naive.date())

    chain = None
    for attempt_date in (d_before, prior_trading_day_str(datetime.strptime(d_before, "%Y-%m-%d").date())):
        chain = fetch_chain(token, ticker, attempt_date)
        if chain and chain.get("options"):
            d_before = attempt_date
            break
        time.sleep(0.2)
    if not chain or not chain.get("options"):
        return {"skipped": "no CBOE chain data for day before/two days before earnings"}

    spot = chain.get("underlying_close")
    opts = chain["options"]
    if not spot or spot <= 0 or not opts:
        return {"skipped": "no usable spot/options in chain"}

    # Nearest expiry >= day after earnings (matches _evc_get_chain_params)
    day_after = (ed_naive.date() + timedelta(days=1)).strftime("%Y-%m-%d")
    expiries = sorted({o["expiry"] for o in opts if o.get("expiry")})
    future_expiries = [e for e in expiries if e >= day_after]
    if not future_expiries:
        return {"skipped": "no future expiry in chain"}
    expiry = future_expiries[0]
    exp_opts = [o for o in opts if o.get("expiry") == expiry]
    strikes = sorted({o["strike"] for o in exp_opts if o.get("strike")})
    if not strikes:
        return {"skipped": "no strikes for chosen expiry"}

    def nearest(target):
        return min(strikes, key=lambda s: abs(s - target))

    def get_opt(strike, right):
        for o in exp_opts:
            if o.get("strike") == strike and o.get("option_type") == right:
                return o
        return None

    atm_k = nearest(spot)
    atm_c, atm_p = get_opt(atm_k, "C"), get_opt(atm_k, "P")
    if not atm_c or not atm_p or not atm_c.get("option_mid") or not atm_p.get("option_mid"):
        return {"skipped": "no ATM straddle mid available"}
    em = atm_c["option_mid"] + atm_p["option_mid"]
    if em <= 0 or em / spot < 0.02:
        return {"skipped": f"implausible EM {em}"}

    short_put, long_put = nearest(spot - em), nearest(spot - 1.5 * em)
    short_call, long_call = nearest(spot + em), nearest(spot + 1.5 * em)
    if long_put >= short_put or short_call >= long_call:
        return {"skipped": "invalid condor strikes (chain too sparse near this spot)"}

    # Real GEX wall + P/C OI ratio, straight from the already-fetched chain --
    # no extra API calls needed (this is the point of pulling the full chain).
    def dollar_gamma(o):
        oi = o.get("open_interest") or 0
        g = o.get("gamma") or 0
        return g * oi * 100 * (spot ** 2) * 0.01

    strike_gex = {}
    total_call_oi = total_put_oi = 0
    for o in exp_opts:
        k = o.get("strike")
        if k is None:
            continue
        oi = o.get("open_interest") or 0
        if o.get("option_type") == "C":
            total_call_oi += oi
            strike_gex[k] = strike_gex.get(k, 0.0) + dollar_gamma(o)
        elif o.get("option_type") == "P":
            total_put_oi += oi
            strike_gex[k] = strike_gex.get(k, 0.0) - dollar_gamma(o)

    if not strike_gex:
        return {"skipped": "no OI/gamma data in chain"}

    put_wall_gex = strike_gex.get(short_put)
    call_wall_gex = strike_gex.get(short_call)
    biggest_k = max(strike_gex, key=lambda k: abs(strike_gex[k]))
    on_biggest_wall = biggest_k in (short_put, short_call)
    pc_ratio = round(total_put_oi / total_call_oi, 3) if total_call_oi else None

    # Real realized outcome: actual close on/after earnings vs the short strikes
    try:
        idx = hist.index.searchsorted(ed_naive)
        if idx >= len(hist):
            return {"skipped": "no real price data after earnings date yet"}
        react_close = hist["Close"].iloc[idx]
    except Exception as exc:
        return {"skipped": f"price lookup failed: {exc}"}

    breached = react_close <= short_put or react_close >= short_call
    move_pct = round((react_close / spot - 1) * 100, 2)

    return {
        "ticker": ticker, "earnings_date": ed_naive.date().isoformat(),
        "chain_date": d_before, "spot": spot, "em": round(em, 2),
        "im_pct": round(em / spot, 4),
        "short_put": short_put, "long_put": long_put,
        "short_call": short_call, "long_call": long_call,
        "put_wall_gex": put_wall_gex, "call_wall_gex": call_wall_gex,
        "on_biggest_wall": on_biggest_wall, "biggest_wall_strike": biggest_k,
        "pc_oi_ratio": pc_ratio,
        "react_close": round(react_close, 2), "move_pct": move_pct,
        "breached": bool(breached),
    }


def main():
    with open(HERE_CFG) as f:
        cfg = json.load(f)
    token = get_token(cfg)
    log("Got CBOE DataShop token OK")

    import re
    with open("main.py", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r'CANDIDATE_POOL: List\[str\] = \[(.*?)\n\]', content, re.DOTALL)
    universe = re.findall(r'"([A-Z.]+)"', m.group(1))
    universe = sorted(set(universe))
    if MAX_TICKERS:
        universe = universe[:MAX_TICKERS]
    log(f"Real universe: {len(universe)} tickers from EVC's own CANDIDATE_POOL")

    results = []
    for i, ticker in enumerate(universe):
        try:
            t = yf.Ticker(ticker)
            cal = t.get_earnings_dates(limit=N_EARNINGS_PER_TICKER + 1)
            hist = t.history(period="3y", interval="1d")
            hist.index = hist.index.tz_localize(None)
        except Exception as exc:
            log(f"  [{i+1}/{len(universe)}] {ticker}: earnings/history lookup failed: {exc}")
            continue
        if cal is None or not len(hist):
            log(f"  [{i+1}/{len(universe)}] {ticker}: no earnings calendar or history")
            continue

        today = datetime.now()
        past_dates = [ed for ed in cal.index if (ed.tz_localize(None) if ed.tzinfo else ed) < today][:N_EARNINGS_PER_TICKER]
        n_ok = n_skip = 0
        for ed in past_dates:
            r = analyze_one(token, ticker, ed, hist)
            if r is None or r.get("skipped"):
                n_skip += 1
                continue
            results.append(r)
            n_ok += 1
            time.sleep(0.25)
        log(f"  [{i+1}/{len(universe)}] {ticker}: {n_ok} real events analyzed, {n_skip} skipped")

        if (i + 1) % 20 == 0:
            with open(OUT_FILE, "w") as f:
                json.dump(results, f, indent=2)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved {len(results)} real analyzed events to {OUT_FILE}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        MAX_TICKERS = int(sys.argv[1])
    main()
