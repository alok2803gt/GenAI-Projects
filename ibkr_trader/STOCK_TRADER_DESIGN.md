# Stock Trader — Design & Build Notes
*Last updated: 2026-07-02*

## 1. What This Is

A standalone **"Stock Trader"** tab + backend module for momentum breakout entries
on equity stocks — completely independent of the CSP/LEAP auto-trader.
Only shared resource: the IBKR `ib` connection object.

---

## 2. Strategy Parameters (backtest-validated, 18 years / 700 signals)

| Parameter | Value | Source |
|-----------|-------|--------|
| Entry | LIMIT at `last_price × 1.001` | Fills within seconds on liquid stocks |
| Phase 1 stop | 7% GTC hard stop (days 1–5) | Fires on only 14.4% of trades; saves avg +0.82% vs holding |
| Phase 2 stop | 5% IBKR TRAIL order (day 5–30) | IBKR manages trail natively |
| Force close | Market sell at day 30 | 16.7% of trades reach here: WR=88%, avg +6.82% |
| Position size | $3,000 fixed (configurable) | Quarter-Kelly on $50K capital |
| Max positions | 8 concurrent | 90th pct of historical concurrent open = 5 |
| Regime gate | SPY > SMA-200 AND VIX < 25 | Without gate: WR=40.8%, avg -0.135% |
| Signal freshness | 30-minute window | Skip stale alerts from scanner |

**Backtest results (regime-gated):**
- WR: 44.2%, Avg return: +0.827% per trade
- Hard stop: 14.4% of trades, avg -7% (saves avg +0.82% vs no stop)
- Trail stop: 68.9% of trades, WR=41.9%, avg +0.781%
- Max hold: 16.7% of trades, WR=88%, avg +6.82%

---

## 3. Latency Chain

**Before (worst case): ~8 minutes**
```
Scanner cycle (0–3 min wait) → POST /watchlist/alert → AT loop picks up (0–5 min wait) → order
```

**After (with direct trigger): ~10–60 seconds**
```
Scanner cycle (0–60 sec wait, reduced from 3 min) → POST /stock-trader/signal → order placed (<1 sec)
```

---

## 4. Order Lifecycle

```
ENTRY
  shares = floor(position_size / last_price)
  LIMIT BUY qty=shares  lmtPrice=round(last_price×1.001, 2)  tif=DAY

  → Monitor detects fill (phase 0 → 1):
    stop = round(fill_price × 0.93, 2)
    STP SELL qty=shares  auxPrice=stop  tif=GTC

PHASE TRANSITION (trading day ≥ 5, phase 1 → 2)
  Cancel STP order
  TRAIL SELL qty=shares  trailingPercent=5.0  tif=GTC

FORCE CLOSE (trading day ≥ 30, phase 2 → closed)
  Cancel TRAIL order
  MKT SELL qty=shares  tif=DAY

EXIT DETECTION (every 60s)
  stop_order_id not in ib.openTrades() → check ib.fills() → record close
```

---

## 5. Files Changed

| File | Change |
|------|--------|
| `backend/main.py` | +~350 lines: state, helpers, monitor loop, 5 endpoints |
| `backend/breakout_scanner.py` | +~15 lines: direct POST /stock-trader/signal on BREAKOUT |
| `frontend/index.html` | +~350 lines: StockTraderTab component + tab registration |
| `backend/stock_state.json` | New — auto-created, gitignored |
| `ibkr_trader/.gitignore` | Add stock_state.json |

---

## 6. New API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/stock-trader/signal` | Scanner fast-path → places LIMIT buy immediately |
| `GET` | `/stock-trader/status` | All positions, config, closed today, decisions |
| `POST` | `/stock-trader/config` | Update any config key |
| `POST` | `/stock-trader/close/{ticker}` | Manual market-close a position |
| `GET` | `/stock-trader/history?days=30` | Closed trades from trade_journal |

---

## 7. New State Keys in `state["stock_trader"]`

```json
{
  "enabled": false,
  "config": {
    "position_size":        3000,
    "max_positions":        8,
    "hard_stop_pct":        7.0,
    "trail_pct":            5.0,
    "max_hold_days":        30,
    "signal_freshness_min": 30,
    "limit_buffer_pct":     0.10
  },
  "positions": {
    "NVDA": {
      "entry_date":        "2026-07-02",
      "entry_price":       135.50,
      "shares":            22,
      "cost":              2981.00,
      "buy_order_id":      10001,
      "stop_order_id":     10002,
      "stop_type":         "STOP",
      "stop_price":        126.02,
      "trading_days_held": 3,
      "phase":             1,
      "alert_fired_at":    "2026-07-02T10:31:44"
    }
  },
  "closed_today": [],
  "decisions":    []
}
```

Position `phase` values:
- `0` = buy order placed, waiting for fill
- `1` = filled, hard stop (STP GTC) active
- `2` = trailing stop (TRAIL GTC) active
- `3` = market sell placed (force close in flight)

---

## 8. Eligibility Checks (`POST /stock-trader/signal`)

| Check | On fail |
|-------|---------|
| `enabled == True` | Skip, log "disabled" |
| Market hours 09:30–15:50 ET | Skip, log "outside hours" |
| `alert_fired_at` within `freshness_min` | Skip, log "stale signal" |
| Ticker not already in `positions` | Skip, log "already open" |
| `len(positions) < max_positions` | Skip, log "at capacity" |
| IBKR connected | Return 503 |

No averaging-down check needed — structurally impossible with this setup
(18-year backtest found zero losing-position duplicate scenarios).

---

## 9. Frontend Tab Layout

```
[● ENABLED]   Stock Trader   $6,000 deployed / $50,000 capital
3 positions   Today: +$142   All-time: +$1,840

[Config panel — collapsible]
  Position $[3000]  Max positions [8]  Hard stop [7]%
  Trail % [5]  Max hold [30]d  Freshness [30]min

[Open Positions]
  Ticker  Entry   Shares  Cost    P&L$    P&L%   Phase  Day  Stop
  NVDA    135.50  22      $2,981  +$125   +4.2%  TRAIL  8    —
  META    532.00  5       $2,660  -$18    -0.7%  STOP   3    $494.76  [Close]

[Decision Log]
  10:32 NVDA ENTERED 135.50×22sh stop@126.02
  10:31 AAPL SKIPPED already open

[Closed Today]
  KLAC  entry 141.20 → exit 131.32  -$218  hard_stop  day 2
```

---

## 10. Build Order

1. Design doc (this file) ✓
2. `stock_state.json` (initial file) + `.gitignore` update
3. `main.py` — constants, state dict, Order import
4. `main.py` — helper functions + monitor loop
5. `main.py` — 5 REST endpoints + Pydantic model
6. `main.py` — lifespan() startup
7. `breakout_scanner.py` — direct trigger
8. `frontend/index.html` — StockTraderTab + tab registration
