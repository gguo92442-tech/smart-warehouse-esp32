import sqlite3

conn = sqlite3.connect('warehouse.db')
cursor = conn.cursor()

# ── 建立 cargo_rules 表（HSV 顏色辨識規則）──
cursor.execute('''
    CREATE TABLE IF NOT EXISTS cargo_rules (
        name TEXT PRIMARY KEY,
        h_min INTEGER, s_min INTEGER, v_min INTEGER,
        h_max INTEGER, s_max INTEGER, v_max INTEGER
    )
''')

# ── 建立 inventory 表（庫存數量）──
cursor.execute('''
    CREATE TABLE IF NOT EXISTS inventory (
        cargo_name TEXT PRIMARY KEY,
        quantity INTEGER DEFAULT 0
    )
''')

# ── 四種貨物初始化（已移除藍色）──
cargo_names = [
    "Cargo A (Pink)",
    "Cargo B (Orange)",
    "Cargo C (Yellow)",
    "Cargo D (Green)",
]
for name in cargo_names:
    cursor.execute(
        "INSERT OR IGNORE INTO inventory (cargo_name, quantity) VALUES (?, 0)", (name,))

# 移除藍色（如果之前存在）
cursor.execute("DELETE FROM inventory WHERE cargo_name = 'Cargo F (Blue)'")

conn.commit()
conn.close()
print("✅ 初始化完成（四色版，cargo_rules + inventory 表已建立）")
