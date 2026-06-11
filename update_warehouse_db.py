import sqlite3

conn = sqlite3.connect('warehouse.db')
cursor = conn.cursor()

# ════════════════════════════════════════════════════════════════
#  HSV 規則（此場地校準版）
#  四色完全分離策略：
#    Pink:   H:46~54  S:52~100   S 最低（S切開Yellow）
#    Yellow: H:46~53  S:101~129  S 中高（S切開Pink）
#    Green:  H:54~61  S:63~104   H 最高
#    Orange: H:38~45  S:102~161  S 最高
# ════════════════════════════════════════════════════════════════

calibrated_rules = [
    # 名稱              H_min S_min V_min  H_max S_max V_max
    ("Cargo A (Pink)",    46,   52,  118,   54,  100,  160),
    ("Cargo C (Yellow)",  46,  101,  132,   53,  129,  182),
    ("Cargo D (Green)",   54,   63,  143,   61,  104,  194),
    ("Cargo B (Orange)",  38,  102,  112,   45,  161,  165),
]

cursor.execute('DELETE FROM cargo_rules')
for rule in calibrated_rules:
    cursor.execute('''
        INSERT INTO cargo_rules (name, h_min, s_min, v_min, h_max, s_max, v_max)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', rule)

conn.commit()
conn.close()

print("HSV 規則已更新")
print()
print("  Pink:   H:46~54  S:52~100   S最低")
print("  Yellow: H:46~53  S:101~129  S中高（S值切開Pink）")
print("  Green:  H:54~61  S:63~104   H最高")
print("  Orange: H:38~45  S:102~161  S最高")
print()
print("  四色完全分離 ✓")