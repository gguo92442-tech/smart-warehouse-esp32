"""
package.py  ── 一鍵打包「智慧倉儲系統」可攜式資料夾
================================================================
執行方式：python package.py
產出：smart_warehouse/ 資料夾，整個複製到 USB 隨身碟即可帶走

使用方式（到新電腦上）：
  1. 安裝 Python 3.10+
  2. 進入 smart_warehouse 資料夾
  3. 執行 setup.bat（Windows）或 setup.sh（Mac/Linux）
  4. 執行 start.bat 啟動系統
================================================================
"""

import os
import shutil

PKG_DIR = "smart_warehouse"

# 要打包的檔案
FILES = [
    "catch_camera.py",
    "auto_hsv_sampler.py",
    "init_inventory.py",
    "update_warehouse_db.py",
    "warehouse.db",
]

def main():
    # 建立資料夾
    if os.path.exists(PKG_DIR):
        shutil.rmtree(PKG_DIR)
    os.makedirs(PKG_DIR)

    # 複製檔案
    copied = []
    missing = []
    for f in FILES:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(PKG_DIR, f))
            copied.append(f)
        else:
            missing.append(f)

    # 建立 requirements.txt
    with open(os.path.join(PKG_DIR, "requirements.txt"), "w") as f:
        f.write("opencv-python>=4.8\nnumpy>=1.24\n")

    # 建立 setup.bat（Windows）
    with open(os.path.join(PKG_DIR, "setup.bat"), "w") as f:
        f.write('@echo off\n')
        f.write('echo ========================================\n')
        f.write('echo   智慧倉儲系統 - 環境安裝\n')
        f.write('echo ========================================\n')
        f.write('python -m venv .venv\n')
        f.write('call .venv\\Scripts\\activate.bat\n')
        f.write('pip install -r requirements.txt\n')
        f.write('python init_inventory.py\n')
        f.write('python update_warehouse_db.py\n')
        f.write('echo.\n')
        f.write('echo 安裝完成！請執行 start.bat 啟動系統\n')
        f.write('pause\n')

    # 建立 start.bat（Windows）
    with open(os.path.join(PKG_DIR, "start.bat"), "w") as f:
        f.write('@echo off\n')
        f.write('call .venv\\Scripts\\activate.bat\n')
        f.write('python catch_camera.py\n')
        f.write('pause\n')

    # 建立 calibrate.bat（校準工具）
    with open(os.path.join(PKG_DIR, "calibrate.bat"), "w") as f:
        f.write('@echo off\n')
        f.write('call .venv\\Scripts\\activate.bat\n')
        f.write('python auto_hsv_sampler.py\n')
        f.write('pause\n')

    # 建立 setup.sh（Mac/Linux）
    with open(os.path.join(PKG_DIR, "setup.sh"), "w") as f:
        f.write('#!/bin/bash\n')
        f.write('echo "========================================"\n')
        f.write('echo "  智慧倉儲系統 - 環境安裝"\n')
        f.write('echo "========================================"\n')
        f.write('python3 -m venv .venv\n')
        f.write('source .venv/bin/activate\n')
        f.write('pip install -r requirements.txt\n')
        f.write('python init_inventory.py\n')
        f.write('python update_warehouse_db.py\n')
        f.write('echo "安裝完成！請執行: source .venv/bin/activate && python catch_camera.py"\n')

    # 建立 README.txt
    with open(os.path.join(PKG_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write("""
╔══════════════════════════════════════════════════════════╗
║           智慧倉儲庫存系統 - 可攜式部署包                  ║
╚══════════════════════════════════════════════════════════╝

【系統需求】
  - Python 3.10 以上
  - ESP32-CAM（IP: 172.20.10.2，可在 catch_camera.py 修改）
  - 同一區域網路（手機熱點或 WiFi）

【首次使用】
  Windows：雙擊 setup.bat → 等待安裝完成 → 雙擊 start.bat
  Mac/Linux：終端執行 bash setup.sh → python catch_camera.py

【日常使用】
  雙擊 start.bat 即可啟動

【快捷鍵】
  q / ESC  = 離開
  r        = 重載庫存
  c        = 清空所有庫存（歸零）

【倉庫規則】
  每種貨物上限 8 個，滿了會顯示 FULL 並拒絕入庫
  按 c 可清空重新開始

【換場地校準】
  1. 雙擊 calibrate.bat
  2. 每色各採樣 1 次（按 1~5 對應顏色）
  3. 按 6 輸出結果
  4. 修改 update_warehouse_db.py 中的數值
  5. 執行 python update_warehouse_db.py
  6. 重啟 start.bat

【四種貨物】
  粉色 + 正方形 = Cargo A (Pink)
  橘色 + 三角形 = Cargo B (Orange)
  黃色 + 六角星 = Cargo C (Yellow)
  綠色 + X 叉叉 = Cargo D (Green)

【修改 ESP32-CAM IP】
  開啟 catch_camera.py，修改第 23 行：
  ESP32_HOST = "你的IP"
""")

    print("=" * 50)
    print("  ✅ 打包完成！")
    print("=" * 50)
    print(f"  資料夾：{PKG_DIR}/")
    print(f"  已複製：{', '.join(copied)}")
    if missing:
        print(f"  ⚠️ 缺少：{', '.join(missing)}")
    print()
    print("  檔案清單：")
    for f in os.listdir(PKG_DIR):
        size = os.path.getsize(os.path.join(PKG_DIR, f))
        print(f"    {f:30s} {size:>8,} bytes")
    print()
    print("  將整個 smart_warehouse 資料夾複製到 USB 即可帶走！")


if __name__ == "__main__":
    main()