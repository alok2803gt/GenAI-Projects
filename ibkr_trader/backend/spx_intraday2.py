import yfinance as yf

# Pull July 9 intraday at 1-min for precision around key trade events
spy = yf.download("SPY", start="2026-07-09", end="2026-07-10", interval="1m", progress=False)
spy.index = spy.index.tz_convert("America/New_York")
spy = spy.between_time("09:30", "16:00")
spy.columns = ["_".join(c).strip() if isinstance(c, tuple) else c for c in spy.columns]
close_col = [c for c in spy.columns if "close" in c.lower()][0]

# July 9 prev close ratio (July 8 closing prices)
spx_hist = yf.Ticker("^GSPC").history(start="2026-07-07", end="2026-07-09")
spy_hist  = yf.Ticker("SPY").history(start="2026-07-07", end="2026-07-09")
spx_prev = float(spx_hist["Close"].iloc[-1])
spy_prev  = float(spy_hist["Close"].iloc[-1])
ratio = spx_prev / spy_prev
print(f"July 8 close — SPX: {spx_prev:.1f}  SPY: {spy_prev:.2f}  ratio: {ratio:.4f}")
print()

# Print full day with key moments annotated
print(f"{'Time':<8} {'SPX est':>9}  Note")
print("-" * 50)
prev_spx = None
for ts, row in spy.iterrows():
    spx = float(row[close_col]) * ratio
    t   = ts.strftime("%H:%M")
    note = ""
    # Annotate key moments
    if t == "09:30": note = "<-- OPEN (gap vs prev close 7483)"
    if t == "10:15": note = "<-- INTRADAY LOW"
    if t == "11:10": note = "<-- ~trade 3 entry zone (spot 7465)"
    if t == "13:17": note = "<-- first log entry  P&L already -$1,513"
    if t == "13:48": note = "<-- STOP LOSS fired  P&L -$1,623"
    if note or t in ["09:31","09:45","10:00","10:30","11:00","11:30","12:00","12:30","13:00","13:30","14:00","15:00","15:55"]:
        print(f"{t:<8} {spx:>9.1f}  {note}")
    prev_spx = spx

print()
print("=== Key range during Trade 3 life ===")
t3_data = spy.between_time("11:00", "13:50")
spx_vals = [float(r[close_col]) * ratio for _, r in t3_data.iterrows()]
print(f"Entry zone (11:00 - 11:15): {min([float(r[close_col])*ratio for _,r in spy.between_time('11:00','11:15').iterrows()]):.1f} - {max([float(r[close_col])*ratio for _,r in spy.between_time('11:00','11:15').iterrows()]):.1f}")
print(f"Afternoon peak (12:00-13:48): {max(spx_vals):.1f}")
print(f"Move vs entry: +{max(spx_vals) - 7465.3:.1f} pts")
