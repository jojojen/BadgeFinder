import sqlite3

DB_PATH = 'badges.db'

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT id, image_hash, source_work, character FROM badges")
rows = cursor.fetchall()

for row in rows:
    print(f"ID: {row[0]}")
    print(f"Hash: {row[1]}")
    print(f"Work: {row[2]}")
    print(f"Character: {row[3]}")
    print("------")

conn.close()
