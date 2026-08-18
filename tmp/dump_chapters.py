import sqlite3, sys

db = sqlite3.connect("/data/short_term.db")
db.row_factory = sqlite3.Row
cur = db.cursor()

# List tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("TABLES:", [r[0] for r in cur.fetchall()])

# Dump chapter_drafts
try:
    cur.execute("PRAGMA table_info(chapter_drafts)")
    print("COLUMNS:", [r["name"] for r in cur.fetchall()])
    cur.execute("SELECT * FROM chapter_drafts ORDER BY novel_id, chapter_seq, version")
    rows = cur.fetchall()
    print("ROWS:", len(rows))
    for r in rows:
        print("=" * 80)
        print("novel_id:", r["novel_id"], "| seq:", r["chapter_seq"], "| version:", r["version"], "| title:", r["title"], "| status:", r["status"])
        content = r["content"]
        print("content length:", len(content))
        print(content)
except Exception as e:
    print("ERR chapter_drafts:", e)

db.close()
