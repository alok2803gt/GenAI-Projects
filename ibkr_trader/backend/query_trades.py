import sqlite3, os

for db in ["trade_journal.db", "trades.db"]:
    path = os.path.join(os.path.dirname(__file__), db)
    conn = sqlite3.connect(path)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"\n{db} tables: {tables}")
    for t in tables:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})").fetchall()]
        print(f"  {t}: {cols}")
        rows = conn.execute(f"SELECT * FROM {t} WHERE date(opened_at) = '2026-07-09' ORDER BY opened_at ASC").fetchall()
        for r in rows:
            for c, v in zip(cols, r):
                print(f"    {c}: {v}")
            print("    ---")
    conn.close()
