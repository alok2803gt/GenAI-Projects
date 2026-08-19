"""Backfill 7/14 EVC journal rows from state file (one-time repair)."""
import json, sqlite3

with open('earnings_vol_crush_state.json') as f:
    ev = json.load(f)

con = sqlite3.connect('trade_journal.db')

updated = 0
skipped = 0
for r in ev['closed_today']:
    ticker     = r['ticker']
    close_cost = r['close_cost']
    pnl        = r['pnl']
    pnl_pct    = r['pnl_pct']
    win        = r['win']
    reason     = r['exit_reason']

    row = con.execute(
        "SELECT id, entry_price FROM trade_journal "
        "WHERE ticker=? AND strategy_type='earnings_vol_crush' AND exit_reason='orphaned' "
        "AND date(opened_at)='2026-07-14' AND exit_price IS NULL "
        "ORDER BY id ASC LIMIT 1",
        (ticker,)).fetchone()

    if row:
        con.execute(
            "UPDATE trade_journal SET exit_price=?, "
            "pnl=?, pnl_pct=?, win=?, exit_reason=? WHERE id=?",
            (round(close_cost, 2), pnl, pnl_pct, win, reason, row[0]))
        print(f"  Updated id={row[0]} {ticker}: exit_price={close_cost:.2f} pnl={pnl:+.2f} ({reason})")
        updated += 1
    else:
        print(f"  SKIP {ticker}: no open journal row found")
        skipped += 1

con.commit()
con.close()
print(f"\nDone: {updated} updated, {skipped} skipped")
