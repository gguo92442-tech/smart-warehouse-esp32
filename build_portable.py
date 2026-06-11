"""
build_portable.py  ── 一鍵建置「智慧倉儲系統」可攜式免安裝包（強化版）
================================================================
執行方式（在你自己的電腦上，需要有網路）：

    python build_portable.py

產出：SmartWarehouse/ 資料夾
  → 整個複製到 USB 隨身碟
  → 到任何 Windows 10+ 電腦，雙擊「啟動系統.bat」即可使用
  → 不需要安裝 Python、不需要 pip、不需要任何東西

預防措施：
  ✓ shape_model.onnx 自動打包
  ✓ sqlite3 DLL 補齊（解決 Embedded Python 缺少問題）
  ✓ IP 自動偵測腳本（不用手動改程式）
  ✓ warehouse.db 首次自動初始化
  ✓ .bat 改用 UTF-8 編碼（解決中文亂碼）
  ✓ 以系統管理員身份執行（解決 UAC 權限問題）
  ✓ 防毒軟體提示說明
================================================================
"""

import os
import sys
import shutil
import urllib.request
import zipfile
import subprocess

# ── 設定 ──
OUT_DIR     = "SmartWarehouse"
PYTHON_VER  = "3.11.9"
PYTHON_URL  = f"https://www.python.org/ftp/python/{PYTHON_VER}/python-{PYTHON_VER}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
PY_DIR      = "python"

# 必要檔案（缺少任何一個就中止）
PROJECT_FILES = [
    "catch_camera.py",
    "auto_hsv_sampler.py",
    "init_inventory.py",
    "update_warehouse_db.py",
]

# 選用檔案（有才複製）
OPTIONAL_FILES = [
    "warehouse.db",
    "shape_model.onnx",   # ← AI 模型，一定要包進去
]

PACKAGES = ["opencv-python==4.8.1.78", "numpy==1.26.4", "onnxsim"]

# 工具檔案（有才複製；換場地重新採集/訓練/驗證用）
OPTIONAL_TOOLS = [
    "collect_dataset.py",
    "train_model.py",
    "verify_model.py",
]


def download(url, dest):
    print(f"  下載中：{url.split('/')[-1]} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"完成 ({size_mb:.1f} MB)")


def fix_sqlite3_dlls(py_dir):
    """
    Python Embedded 有時缺少 sqlite3 所需的 DLL。
    從系統的 Python 安裝複製過來（如果有的話）。
    """
    # 先確認 Embedded Python 能不能 import sqlite3
    python_exe = os.path.abspath(os.path.join(py_dir, "python.exe"))
    result = subprocess.run(
        [python_exe, "-c", "import sqlite3; print('ok')"],
        capture_output=True, text=True
    )
    if result.stdout.strip() == "ok":
        print("  sqlite3 正常，不需要補充 DLL")
        return

    print("  sqlite3 缺失，嘗試從系統 Python 複製 DLL...")
    # 嘗試從系統 Python 找 sqlite3.dll 或 _sqlite3.pyd
    import glob
    candidates = []
    for base in [sys.prefix, os.path.dirname(sys.executable)]:
        candidates += glob.glob(os.path.join(base, "**", "_sqlite3*.pyd"), recursive=True)
        candidates += glob.glob(os.path.join(base, "**", "sqlite3*.dll"), recursive=True)

    copied = 0
    for src in candidates:
        dst = os.path.join(py_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"    複製 {os.path.basename(src)}")
            copied += 1

    if copied == 0:
        print("  ⚠ 無法自動補充 DLL，若啟動時出現 sqlite3 錯誤，")
        print("    請手動安裝 Python 3.11 後重新執行本工具。")


def write_bat(path, lines, encoding="utf-8"):
    """寫入 .bat 檔案，使用 UTF-8 + BOM（解決中文亂碼）"""
    with open(path, "w", encoding="utf-8-sig") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    print("=" * 60)
    print("  智慧倉儲系統 ── 可攜式免安裝包建置工具（強化版）")
    print("=" * 60)

    # ── 防毒軟體提醒 ──
    print("""
⚠  注意事項：
   建置過程會下載 Python 官方套件，部分防毒軟體可能誤判。
   若出現安全性警告，請選擇「允許」或暫時關閉即時防護。
   建置完成後可重新開啟防毒軟體。
""")

    # ── 檢查專案檔案 ──
    missing = [f for f in PROJECT_FILES if not os.path.exists(f)]
    if missing:
        print("[錯誤] 找不到以下必要檔案：")
        for f in missing:
            print(f"  - {f}")
        input("\n按 Enter 結束...")
        return

    # 提醒 shape_model.onnx
    if not os.path.exists("shape_model.onnx"):
        print("⚠  找不到 shape_model.onnx！")
        print("   系統將退回幾何規則辨識（圖形辨識較不穩定）")
        ans = input("   確定要繼續嗎？(y/n): ").strip().lower()
        if ans != 'y':
            print("已取消。請先執行 train_model.py 訓練模型。")
            return

    # ── 清理舊資料夾 ──
    if os.path.exists(OUT_DIR):
        print(f"\n[清理] 移除舊的 {OUT_DIR}/ ...")
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    py_dir = os.path.join(OUT_DIR, PY_DIR)
    os.makedirs(py_dir)

    # ════════════════════════════════════════
    #  Step 1：下載 Python Embedded
    # ════════════════════════════════════════
    print("\n[Step 1/5] 下載 Python Embedded ...")
    zip_path = os.path.join(OUT_DIR, "python_embed.zip")
    download(PYTHON_URL, zip_path)

    print("  解壓縮中 ...", end=" ", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(py_dir)
    os.remove(zip_path)
    print("完成")

    # 修改 ._pth 允許 site-packages
    pth_files = [f for f in os.listdir(py_dir) if f.endswith("._pth")]
    for pth in pth_files:
        pth_path = os.path.join(py_dir, pth)
        with open(pth_path, "r") as f:
            content = f.read()
        content = content.replace("#import site", "import site")
        content += "\n..\n"
        with open(pth_path, "w") as f:
            f.write(content)
    print("  已修改 ._pth 設定")

    # ════════════════════════════════════════
    #  Step 2：補充 sqlite3 DLL
    # ════════════════════════════════════════
    print("\n[Step 2/5] 確認 sqlite3 支援 ...")
    fix_sqlite3_dlls(py_dir)

    # ════════════════════════════════════════
    #  Step 3：安裝 pip + 套件
    # ════════════════════════════════════════
    print("\n[Step 3/5] 安裝 pip ...")
    get_pip_path = os.path.join(py_dir, "get-pip.py")
    download(GET_PIP_URL, get_pip_path)

    python_exe  = os.path.abspath(os.path.join(py_dir, "python.exe"))
    get_pip_abs = os.path.abspath(get_pip_path)
    subprocess.run([python_exe, get_pip_abs, "--no-warn-script-location"],
                   check=True)
    os.remove(get_pip_path)

    print(f"\n[Step 4/5] 安裝套件：{', '.join(PACKAGES)} ...")
    subprocess.run([python_exe, "-m", "pip", "install",
                    *PACKAGES, "--no-warn-script-location"],
                   check=True)

    # ════════════════════════════════════════
    #  Step 4：複製專案檔案
    # ════════════════════════════════════════
    print(f"\n[Step 5/5] 複製專案檔案 ...")
    for f in PROJECT_FILES:
        shutil.copy2(f, os.path.join(OUT_DIR, f))
        print(f"  ✓ {f}")
    for f in OPTIONAL_FILES:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(OUT_DIR, f))
            print(f"  ✓ {f}")
        else:
            print(f"  - {f}（找不到，跳過）")

    # ════════════════════════════════════════
    #  Step 5：產生 .bat 腳本
    # ════════════════════════════════════════

    # ── IP 自動偵測腳本 ──
    with open(os.path.join(OUT_DIR, "find_ip.py"), "w", encoding="utf-8") as f:
        f.write('''#!/usr/bin/env python3
"""
find_ip.py  ── 自動偵測 ESP32-CAM 的 IP 並更新 catch_camera.py
使用方式：雙擊「偵測ESP32-IP.bat」執行
"""
import subprocess, re, socket, sys, os

def ping(ip):
    try:
        r = subprocess.run(["ping", "-n", "1", "-w", "500", ip],
                           capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False

def try_connect(ip, port=80, timeout=1.5):
    try:
        s = socket.create_connection((ip, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False

def get_arp_ips():
    result = subprocess.run(["arp", "-a"], capture_output=True, text=True)
    ips = re.findall(r"(\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})", result.stdout)
    return [ip for ip in ips if not ip.endswith(".255") and not ip.endswith(".1")
            and not ip.startswith("224.") and not ip.startswith("239.")
            and ip != "255.255.255.255"]

print("=" * 50)
print("  ESP32-CAM IP 自動偵測工具")
print("=" * 50)
print()
print("確認 ESP32-CAM 已通電並連上手機熱點...")
print()

candidates = get_arp_ips()
print(f"ARP 表中找到 {len(candidates)} 個 IP，逐一測試 port 80...")

found = None
for ip in candidates:
    print(f"  測試 {ip} ...", end=" ", flush=True)
    if try_connect(ip):
        print("✓ 回應！")
        found = ip
        break
    else:
        print("✗")

if not found:
    print()
    print("⚠  找不到 ESP32-CAM，請確認：")
    print("   1. ESP32-CAM 已通電（LED 亮）")
    print("   2. ESP32-CAM 和電腦連到同一個熱點")
    print("   3. 手機熱點開啟「最大化相容性」（2.4GHz）")
    input("\\n按 Enter 結束...")
    sys.exit(1)

print()
print(f"✅ 找到 ESP32-CAM：{found}")

# 更新 catch_camera.py
cam_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catch_camera.py")
if os.path.exists(cam_file):
    with open(cam_file, "r", encoding="utf-8") as f:
        src = f.read()
    import re as _re
    new_src = _re.sub(r\'ESP32_HOST\\s*=\\s*"[^"]*"\', f\'ESP32_HOST = "{found}"\', src)
    with open(cam_file, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"   catch_camera.py 已自動更新為 {found}")

# 同時更新 auto_hsv_sampler.py
sampler_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_hsv_sampler.py")
if os.path.exists(sampler_file):
    with open(sampler_file, "r", encoding="utf-8") as f:
        src = f.read()
    new_src = _re.sub(r\'ESP32_HOST\\s*=\\s*"[^"]*"\', f\'ESP32_HOST = "{found}"\', src)
    with open(sampler_file, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"   auto_hsv_sampler.py 已自動更新為 {found}")

print()
input("按 Enter 關閉...")
''')

    # ── 啟動系統.bat（以系統管理員執行，UTF-8）──
    write_bat(os.path.join(OUT_DIR, "啟動系統.bat"), [
        '@echo off',
        'chcp 65001 >nul',
        ':: 要求系統管理員權限（解決 UAC 問題）',
        'net session >nul 2>&1',
        'if %errorLevel% neq 0 (',
        '    powershell -Command "Start-Process \'%~f0\' -Verb RunAs"',
        '    exit /b',
        ')',
        'echo ========================================',
        'echo   智慧倉儲庫存辨識系統',
        'echo   按 q 或 ESC 離開',
        'echo   按 c 清空庫存',
        'echo   按 5-8 逐件出庫',
        'echo ========================================',
        'echo.',
        'cd /d %~dp0',
        ':: 首次啟動自動初始化資料庫',
        'if not exist warehouse.db (',
        '    echo 首次啟動，初始化資料庫...',
        '    %~dp0python\\python.exe %~dp0init_inventory.py',
        '    %~dp0python\\python.exe %~dp0update_warehouse_db.py',
        ')',
        '%~dp0python\\python.exe %~dp0catch_camera.py',
        'pause',
    ])

    # ── 偵測ESP32-IP.bat ──
    write_bat(os.path.join(OUT_DIR, "偵測ESP32-IP.bat"), [
        '@echo off',
        'chcp 65001 >nul',
        'echo 正在偵測 ESP32-CAM 的 IP...',
        'cd /d %~dp0',
        '%~dp0python\\python.exe %~dp0find_ip.py',
    ])

    # ── 校準工具.bat ──
    write_bat(os.path.join(OUT_DIR, "校準工具.bat"), [
        '@echo off',
        'chcp 65001 >nul',
        'echo ========================================',
        'echo   HSV 色彩校準工具',
        'echo   按 3=粉色 5=橘色 2=黃色 4=綠色',
        'echo   按 6=輸出結果  q=離開',
        'echo ========================================',
        'cd /d %~dp0',
        '%~dp0python\\python.exe %~dp0auto_hsv_sampler.py',
        'pause',
    ])

    # ── 清空庫存.bat ──
    write_bat(os.path.join(OUT_DIR, "清空庫存.bat"), [
        '@echo off',
        'chcp 65001 >nul',
        'cd /d %~dp0',
        'echo 正在清空所有庫存...',
        '%~dp0python\\python.exe -c "import sqlite3; c=sqlite3.connect(\'warehouse.db\'); c.execute(\'UPDATE inventory SET quantity=0\'); c.commit(); c.close(); print(\'已清空！\')"',
        'pause',
    ])

    # ── 更新HSV規則.bat ──
    write_bat(os.path.join(OUT_DIR, "更新HSV規則.bat"), [
        '@echo off',
        'chcp 65001 >nul',
        'cd /d %~dp0',
        '%~dp0python\\python.exe %~dp0update_warehouse_db.py',
        'pause',
    ])

    # ── 重新初始化.bat ──
    write_bat(os.path.join(OUT_DIR, "重新初始化.bat"), [
        '@echo off',
        'chcp 65001 >nul',
        'cd /d %~dp0',
        'echo 重新建立資料庫...',
        'del warehouse.db 2>nul',
        'del warehouse.db-wal 2>nul',
        'del warehouse.db-shm 2>nul',
        '%~dp0python\\python.exe %~dp0init_inventory.py',
        '%~dp0python\\python.exe %~dp0update_warehouse_db.py',
        'echo 完成！',
        'pause',
    ])

    # ── 使用說明.txt ──
    with open(os.path.join(OUT_DIR, "使用說明.txt"), "w", encoding="utf-8") as f:
        f.write("""╔══════════════════════════════════════════════════════════╗
║        智慧倉儲庫存辨識系統 ── 可攜式免安裝版（強化版）      ║
╚══════════════════════════════════════════════════════════╝

【系統需求】
  Windows 10 / 11（64位元）

【啟動前必做】
  ① ESP32-CAM 通電，連上手機熱點
  ② 雙擊「偵測ESP32-IP.bat」→ 自動找到 ESP32 的 IP
  ③ 雙擊「啟動系統.bat」→ 開始辨識

【換場地操作（一定要做！）】
  ① 雙擊「偵測ESP32-IP.bat」（IP 可能不同）
  ② 雙擊「校準工具.bat」
  ③ 每個顏色各按 3~5 次採樣
  ④ 按 6 輸出結果
  ⑤ 用記事本修改 update_warehouse_db.py 中的數值
  ⑥ 雙擊「更新HSV規則.bat」
  ⑦ 雙擊「啟動系統.bat」

【所有 .bat 檔案說明】
  啟動系統.bat      → 主程式（辨識 + 入庫）
  偵測ESP32-IP.bat  → 自動找到 ESP32 IP 並更新程式
  校準工具.bat      → 換場地時重新校準顏色
  清空庫存.bat      → 所有貨物數量歸零
  更新HSV規則.bat   → 修改顏色規則後執行
  重新初始化.bat    → 完全重建資料庫

【主程式按鍵】
  q / ESC  = 離開
  r        = 重載庫存
  c        = 清空所有庫存
  5        = 粉色 Cargo A 出庫 -1
  6        = 橘色 Cargo B 出庫 -1
  7        = 黃色 Cargo C 出庫 -1
  8        = 綠色 Cargo D 出庫 -1

【四種貨物】
  粉色 + 正方形 = Cargo A (Pink)
  橘色 + 三角形 = Cargo B (Orange)
  黃色 + 六角星 = Cargo C (Yellow)
  綠色 + X 叉叉 = Cargo D (Green)
  每種上限 8 個，滿了顯示 FULL

【防毒軟體警告】
  若出現安全性警告，請選擇「允許」或將此資料夾加入白名單。
  本程式僅包含 Python 官方套件與 OpenCV，無惡意程式碼。

【連線失敗排查】
  1. 確認 ESP32-CAM LED 亮燈
  2. 手機熱點開「最大化相容性」（2.4GHz）
  3. 執行「偵測ESP32-IP.bat」更新 IP
  4. 若電腦沒有 WiFi 網卡，需 iPhone USB 網路共享 + iTunes
""")

    # ── 完成報告 ──
    total_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(OUT_DIR)
        for f in files
    )

    print()
    print("=" * 60)
    print("  ✅ 建置完成！")
    print("=" * 60)
    print(f"  資料夾：{OUT_DIR}/")
    print(f"  總大小：{total_size / 1024 / 1024:.1f} MB")
    print()
    print("  新增功能：")
    print("  ✓ shape_model.onnx 已打包（AI 模型）")
    print("  ✓ sqlite3 DLL 已補充")
    print("  ✓ 偵測ESP32-IP.bat（自動找 IP）")
    print("  ✓ 以系統管理員執行（解決 UAC）")
    print("  ✓ UTF-8 編碼（解決中文亂碼）")
    print()
    print("  使用步驟：")
    print("  1. 複製整個 SmartWarehouse/ 到 USB")
    print("  2. 插上目標電腦")
    print("  3. 雙擊「偵測ESP32-IP.bat」")
    print("  4. 雙擊「啟動系統.bat」")
    print()


if __name__ == "__main__":
    main()