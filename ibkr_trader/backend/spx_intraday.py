import yfinance as yf

spy = yf.download("SPY", start="2026-07-09", end="2026-07-10", interval="5m", progress=False)
spy.index = spy.index.tz_convert("America/New_York")
spy = spy.between_time("09:30", "16:00")
spy.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in spy.columns]

spx_pc = yf.Ticker("^GSPC").fast_info.previous_close
spy_pc = yf.Ticker("SPY").fast_info.previous_close
ratio  = spx_pc / spy_pc
print(f"SPX/SPY ratio: {ratio:.4f}  (SPX prev {spx_pc:.1f} / SPY prev {spy_pc:.2f})")
print()
print(f"{'Time':<8} {'SPY':>7} {'SPX est':>9} {'Chg':>7}")
print("-" * 36)

prev = None
close_col = [c for c in spy.columns if "close" in c.lower() or "Close" in c][0]
for ts, row in spy.iterrows():
    spx = float(row[close_col]) * ratio
    chg = f"{spx - prev:+.1f}" if prev is not None else "  open"
    print(f"{str(ts)[11:16]:<8} {float(row[close_col]):>7.2f} {spx:>9.1f} {chg:>7}")
    prev = spx
