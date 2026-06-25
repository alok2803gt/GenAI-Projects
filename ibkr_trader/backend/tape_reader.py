"""
tape_reader.py
─────────────────────────────────────────────────────────────────────────────
Live tape (Time & Sales) reader using IBKR TWS / IB Gateway.
Connects via ib_insync on a separate client ID (default 20) so it won't
conflict with the main ibkr_trader app (client ID 10).

Requirements:
    pip install ib_insync colorama   (both already in requirements.txt)

Usage:
    python tape_reader.py MSFT
    python tape_reader.py NVDA --block 10000
    python tape_reader.py AAPL --port 7496          # live account
    python tape_reader.py MSFT --port 4002          # IB Gateway paper
    python tape_reader.py MSFT --client-id 25       # if 20 is taken

Columns:
    Time         — print timestamp (HH:MM:SS.mmm)
    Price        — execution price
    Size         — shares traded
    Dir          — ▲ uptick  ▼ downtick  ─ unchanged (tick rule)
    Cum Vol      — cumulative session volume seen on this tape
    VWAP         — running volume-weighted average price
    Δ Vol        — running buy vol minus sell vol (uptick = buy)
    Exch         — reporting exchange
    AH           — flag when tick is outside RTH (pre/after market)

Block threshold (--block, default 5000 shares) prints a highlighted banner
instead of a regular row so large institutional prints stand out immediately.
"""

import sys
import asyncio
import argparse
from datetime import datetime

# Windows: force UTF-8 so box-drawing chars don't crash on cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from ib_insync import IB, Stock, util
except ImportError:
    print("\n  ib_insync not found. Run:\n  pip install ib_insync\n")
    sys.exit(1)

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class _Dummy:
        def __getattr__(self, _): return ""
    Fore = Style = _Dummy()

util.logToConsole(None)  # silence ib_insync internal logs


# ── formatting helpers ────────────────────────────────────────────────────────

def c(text, color):
    return f"{color}{text}{Style.RESET_ALL}" if HAS_COLOR else str(text)

def fmt_vol(v: int) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return str(v)

def fmt_delta(delta: int) -> str:
    sign = "+" if delta >= 0 else ""
    col = Fore.GREEN if delta >= 0 else Fore.RED
    return c(f"{sign}{fmt_vol(abs(delta))}", col)

def fmt_price_chg(current, ref) -> str:
    if ref is None or ref == 0:
        return ""
    chg = current - ref
    pct = chg / ref * 100
    col = Fore.GREEN if chg >= 0 else Fore.RED
    return c(f"({chg:+.2f} / {pct:+.2f}%)", col)


# ── tape ──────────────────────────────────────────────────────────────────────

class Tape:
    STATS_EVERY = 30   # print a running-stats divider every N prints
    W = 78             # line width

    def __init__(self, symbol: str, block_size: int):
        self.symbol = symbol
        self.block_size = block_size

        # running accumulators
        self.cum_vol: int = 0
        self.cum_pv: float = 0.0    # sum(price * size) for VWAP
        self.buy_vol: int = 0
        self.sell_vol: int = 0

        self.open_price: float | None = None
        self.last_price: float | None = None
        self.last_dir: int = 0       # +1 / -1 / 0 for tick rule

        self.print_count: int = 0
        self.block_count: int = 0
        self.last_tick_idx: int = 0  # index into ticker.tickByTicks

        self._print_header()

    def _print_header(self):
        print(f"\n{c('━' * self.W, Fore.CYAN)}")
        print(c(f"  LIVE TAPE  ·  {self.symbol}  ·  Ctrl+C to stop  ·  "
                f"Blocks ≥ {self.block_size:,} shares highlighted", Fore.CYAN + Style.BRIGHT))
        print(c('━' * self.W, Fore.CYAN))
        hdr = (
            f"  {'Time':12}  {'Price':>9}  {'Size':>9}  "
            f"{'D':1}  {'CumVol':>7}  {'VWAP':>9}  {'ΔVol':>9}  Exch"
        )
        print(c(hdr, Style.BRIGHT))
        print(c("  " + "─" * (self.W - 2), Fore.CYAN))

    def _stats_bar(self):
        vwap = self.cum_pv / self.cum_vol if self.cum_vol else 0.0
        delta = self.buy_vol - self.sell_vol
        now = datetime.now().strftime("%H:%M:%S")
        last_str = f"${self.last_price:.2f}" if self.last_price else "—"
        chg_str = fmt_price_chg(self.last_price, self.open_price) if self.last_price else ""
        line = (
            f"  ── {c(now, Fore.CYAN)}  "
            f"Last {c(last_str, Fore.CYAN + Style.BRIGHT)} {chg_str}  "
            f"VWAP {c(f'${vwap:.2f}', Fore.YELLOW)}  "
            f"Vol {c(fmt_vol(self.cum_vol), Fore.WHITE)}  "
            f"Δ {fmt_delta(delta)}  "
            f"Blocks {c(str(self.block_count), Fore.MAGENTA)}  "
            f"Prints {self.print_count} ──"
        )
        print(line)

    def _direction(self, price: float) -> int:
        if self.last_price is None:
            return 0
        if price > self.last_price:
            self.last_dir = 1
        elif price < self.last_price:
            self.last_dir = -1
        return self.last_dir

    def _dir_arrow(self, d: int) -> str:
        if d > 0:  return c("▲", Fore.GREEN)
        if d < 0:  return c("▼", Fore.RED)
        return c("─", Fore.YELLOW)

    def process(self, tick) -> None:
        price = float(tick.price)
        size  = int(tick.size)
        if size == 0 or price <= 0:
            return

        ts    = tick.time.strftime("%H:%M:%S.%f")[:-3]
        exch  = (tick.exchange or "—")[:5]
        past_limit = tick.tickAttribLast.pastLimit  # True = pre/AH

        direction = self._direction(price)

        if self.open_price is None:
            self.open_price = price

        # accumulators
        self.cum_vol += size
        self.cum_pv  += price * size
        vwap = self.cum_pv / self.cum_vol

        if direction >= 0:
            self.buy_vol += size
        else:
            self.sell_vol += size

        self.last_price = price
        self.print_count += 1
        is_block = size >= self.block_size
        if is_block:
            self.block_count += 1

        delta = self.buy_vol - self.sell_vol

        if is_block:
            self._print_block(ts, price, size, direction, exch, vwap, delta, past_limit)
        else:
            self._print_tick(ts, price, size, direction, exch, vwap, delta, past_limit)

        if self.print_count % self.STATS_EVERY == 0:
            self._stats_bar()

    def _print_tick(self, ts, price, size, direction, exch, vwap, delta, past_limit):
        col = Fore.GREEN if direction > 0 else Fore.RED if direction < 0 else Fore.YELLOW
        ahr = c(" AH", Fore.MAGENTA) if past_limit else ""
        print(
            f"  {c(ts, col)}  "
            f"{c(f'${price:>8.2f}', col)}  "
            f"{size:>9,}  "
            f"{self._dir_arrow(direction)}  "
            f"{fmt_vol(self.cum_vol):>7}  "
            f"${vwap:>8.2f}  "
            f"{fmt_delta(delta)}  "
            f"{exch}{ahr}"
        )

    def _print_block(self, ts, price, size, direction, exch, vwap, delta, past_limit):
        col = Fore.GREEN if direction >= 0 else Fore.RED
        side = "BUY " if direction >= 0 else "SELL"
        ahr  = " AH" if past_limit else ""
        delta_str = fmt_delta(delta)
        print(
            f"  {c('▐█ BLOCK', col + Style.BRIGHT)}  "
            f"{c(ts, col)}  "
            f"{c(f'${price:.2f}', col + Style.BRIGHT)}  "
            f"×{c(f'{size:,}', col + Style.BRIGHT)}  "
            f"{c(side, col + Style.BRIGHT)}  "
            f"{exch}{ahr}  "
            f"│ VWAP ${vwap:.2f}  Δ {delta_str}"
        )

    def session_summary(self):
        print(f"\n{c('━' * self.W, Fore.CYAN)}")
        print(c(f"  Session Summary  ·  {self.symbol}", Fore.CYAN + Style.BRIGHT))
        print(c('━' * self.W, Fore.CYAN))
        if self.cum_vol == 0:
            print("  No ticks captured.")
            print(c('━' * self.W, Fore.CYAN))
            return
        vwap  = self.cum_pv / self.cum_vol
        delta = self.buy_vol - self.sell_vol
        chg_str = fmt_price_chg(self.last_price, self.open_price)
        rows = [
            ("Prints",    f"{self.print_count:,}"),
            ("Cum vol",   fmt_vol(self.cum_vol)),
            ("VWAP",      f"${vwap:.2f}"),
            ("Buy vol",   c(fmt_vol(self.buy_vol), Fore.GREEN)),
            ("Sell vol",  c(fmt_vol(self.sell_vol), Fore.RED)),
            ("Net delta", fmt_delta(delta)),
            ("Blocks",    c(str(self.block_count), Fore.MAGENTA)),
            ("Open",      f"${self.open_price:.2f}" if self.open_price else "—"),
            ("Last",      f"${self.last_price:.2f} {chg_str}" if self.last_price else "—"),
        ]
        for label, val in rows:
            print(f"  {label:<12} {val}")
        print(c('━' * self.W, Fore.CYAN) + "\n")


# ── connection & streaming loop ───────────────────────────────────────────────

async def run(symbol: str, port: int, client_id: int, block_size: int):
    ib   = IB()
    tape = Tape(symbol.upper(), block_size)

    # connect
    print(f"  Connecting to TWS  port={port}  clientId={client_id} ...", end=" ", flush=True)
    try:
        await ib.connectAsync("127.0.0.1", port, clientId=client_id, timeout=15)
        print(c("connected.", Fore.GREEN))
    except Exception as exc:
        print(c(f"\n  Connection failed: {exc}", Fore.RED))
        print(c("  Make sure TWS or IB Gateway is running with API connections enabled.", Fore.YELLOW))
        sys.exit(1)

    # qualify contract
    contract = Stock(symbol.upper(), "SMART", "USD")
    try:
        await ib.qualifyContractsAsync(contract)
    except Exception as exc:
        print(c(f"\n  Could not qualify {symbol}: {exc}", Fore.RED))
        ib.disconnect()
        sys.exit(1)

    print(f"  Streaming {c(symbol.upper(), Fore.CYAN + Style.BRIGHT)}  "
          f"·  block threshold {block_size:,} shares\n")

    # subscribe to tick-by-tick AllLast (every trade, all exchanges)
    ticker = ib.reqTickByTickData(contract, "AllLast", numberOfTicks=0, ignoreSize=False)

    def on_update(t):
        new_ticks = t.tickByTicks[tape.last_tick_idx:]
        tape.last_tick_idx = len(t.tickByTicks)
        for tick in new_ticks:
            tape.process(tick)

    ticker.updateEvent += on_update

    # run until disconnected or Ctrl+C
    try:
        while ib.isConnected():
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            ib.cancelTickByTickData(contract, "AllLast")
        except Exception:
            pass
        ib.disconnect()
        tape.session_summary()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IBKR live tape reader — Time & Sales via ib_insync"
    )
    parser.add_argument("ticker",
                        help="Stock ticker symbol (e.g. MSFT, NVDA)")
    parser.add_argument("--port", type=int, default=7497,
                        help="TWS/Gateway port  7497=paper TWS (default)  "
                             "7496=live TWS  4002=IB Gateway paper")
    parser.add_argument("--client-id", type=int, default=20,
                        help="IB API client ID — must differ from main app's ID 10 (default: 20)")
    parser.add_argument("--block", type=int, default=5000,
                        help="Block trade alert threshold in shares (default: 5000)")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(
        run(args.ticker, args.port, args.client_id, args.block)
    )

    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
