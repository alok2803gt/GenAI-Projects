"""
Pulls 5 years of daily OHLCV for breakout_scanner.py's own CURATED_TICKERS
universe (112 names, read as reference only -- this script never imports or
modifies breakout_scanner.py itself). Real yfinance data, same source every
other backtest in this codebase uses. Saves one pickle of {ticker: DataFrame}
so every later research iteration reuses this instead of re-fetching.

Read-only reference to the live universe list: CURATED_TICKERS is copied
here verbatim (same convention every other standalone script in this repo
already uses -- "duplicated for self-containment").
"""
import pickle
import time
from pathlib import Path

import yfinance as yf

CURATED_TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "GLD", "TLT", "ARKK",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SMCI",
    "CRM", "NOW", "ADBE", "ORCL", "SNOW", "PANW", "CRWD", "ZS", "DDOG", "NET",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "V", "MA", "AXP", "TFC",
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "DHR", "ISRG", "VRTX", "GILD", "BMY",
    "HD", "MCD", "SBUX", "NKE", "LOW", "TGT", "COST", "BKNG", "LULU",
    "PG", "KO", "PEP", "WMT",
    "XOM", "CVX", "COP", "SLB", "MPC", "VLO", "OXY",
    "BA", "GE", "CAT", "HON", "RTX", "LMT", "FDX", "UPS", "DE", "UAL",
    "DIS", "CMCSA", "VZ", "T",
    "COIN", "PLTR", "UBER", "RIVN", "ROKU", "HOOD", "SOFI", "PYPL", "XYZ", "IBM",
    "RBLX", "RCL", "ABNB",
]))

OUT_PATH = Path(__file__).parent / "universe_5y_ohlcv.pkl"


def main():
    data = {}
    failed = []
    for i, tk in enumerate(CURATED_TICKERS, 1):
        try:
            hist = yf.Ticker(tk).history(period="5y", interval="1d", auto_adjust=False)
            if len(hist) < 200:
                print(f"[{i}/{len(CURATED_TICKERS)}] {tk}: only {len(hist)} rows, skipping")
                failed.append(tk)
                continue
            data[tk] = hist
            print(f"[{i}/{len(CURATED_TICKERS)}] {tk}: {len(hist)} rows, "
                  f"{hist.index[0].date()} to {hist.index[-1].date()}")
        except Exception as e:
            print(f"[{i}/{len(CURATED_TICKERS)}] {tk}: FAILED - {e}")
            failed.append(tk)
        time.sleep(0.15)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved {len(data)}/{len(CURATED_TICKERS)} tickers to {OUT_PATH}")
    if failed:
        print(f"Failed/skipped: {failed}")


if __name__ == "__main__":
    main()
