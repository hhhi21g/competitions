import sqlite3

SQLITE_PATH = "dataset_1/session_kv.sqlite"

conn = sqlite3.connect(SQLITE_PATH)
cur = conn.cursor()

cur.execute("PRAGMA table_info(sessions);")
print(cur.fetchall())

conn.close()
