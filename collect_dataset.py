"""
collect_dataset.py  ── AI 訓練資料收集工具（優化版）
================================================================
用途：
  從 ESP32-CAM 串流中擷取色紙 ROI，自動儲存到對應類別資料夾。

操作方式：
  1. 把色紙對準鏡頭（系統會自動框出色紙）
  2. 按下對應數字鍵開始連拍：
       1 = square    (粉色 - 正方形)
       2 = triangle  (橘色 - 三角形)
       3 = hexagram  (黃色 - 六角星)
       4 = cross     (綠色 - X)
  3. 按下後系統會連續拍 50 張
  4. 拍攝期間請慢慢轉動 / 移動色紙

  其他按鍵：
       p / 空白鍵 = 暫停 / 繼續採集
       u = 復原（刪除最近一張）
       q = 離開
================================================================
"""

import cv2
import numpy as np
import socket
import threading
import time
import os
import sqlite3
from datetime import datetime

# ────────────────────────────────────────────────
ESP32_HOST   = "172.20.10.3"
ESP32_PORT   = 80
STREAM_PATH  = "/"
DB_PATH      = "warehouse.db"
DATASET_DIR  = "dataset"
SAMPLE_COUNT = 50
SAMPLE_DELAY = 0.15
ROI_SIZE     = 128
RECV_CHUNK   = 65536
TARGET_PER_CLASS = {
    "square":   600,
    "triangle": 600,
    "hexagram": 600,
    "cross":    600,
    "circle":   600,
}      # 目標每類張數
MIN_BLACK_PX = 50           # ROI 內最少黑色像素（過濾無圖案的純色紙）
# ────────────────────────────────────────────────

CLASS_KEYS = {
    ord('1'): "square",
    ord('2'): "triangle",
    ord('3'): "hexagram",
    ord('4'): "cross",
    ord('5'): "circle",
}

CLASS_DESC = {
    "square":   "粉色 - 正方形",
    "triangle": "橘色 - 三角形",
    "hexagram": "黃色 - 六角星",
    "cross":    "綠色 - X 叉叉",
    "circle":   "任意 - 圓形（負樣本）",
}

CLASS_COLORS = {
    "square":   (180, 105, 255),
    "triangle": (  0, 165, 255),
    "hexagram": (  0, 215, 255),
    "cross":    (  0, 200, 100),
    "circle":   (100, 100, 255),
}


# ════════════════════════════════════════════════
#  MJPEG 串流讀取
# ════════════════════════════════════════════════
class MJPEGStreamReader(threading.Thread):
    SOI = b'\xff\xd8'
    EOI = b'\xff\xd9'

    def __init__(self, host, port, path):
        super().__init__(daemon=True, name="StreamReader")
        self.host, self.port, self.path = host, port, path
        self._lock = threading.Lock()
        self._latest = None
        self._quit = threading.Event()
        self.connected = False

    def get_latest_frame(self):
        with self._lock:
            f = self._latest
            self._latest = None
        return f

    def request_stop(self):
        self._quit.set()

    def run(self):
        while not self._quit.is_set():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self.host, self.port))
                sock.sendall((
                    f"GET {self.path} HTTP/1.1\r\n"
                    f"Host: {self.host}:{self.port}\r\n"
                    f"Connection: keep-alive\r\n\r\n"
                ).encode())
                buf = bytearray()
                self.connected = True
                while not self._quit.is_set():
                    try:
                        chunk = sock.recv(RECV_CHUNK)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf.extend(chunk)
                    end = buf.rfind(self.EOI)
                    if end == -1:
                        if len(buf) > RECV_CHUNK * 8:
                            buf = buf[-(RECV_CHUNK * 4):]
                        continue
                    start = buf.rfind(self.SOI, 0, end)
                    if start == -1:
                        buf = buf[end + 2:]
                        continue
                    jpeg = bytes(buf[start: end + 2])
                    buf = buf[end + 2:]
                    if len(jpeg) > 100 and jpeg[:2] == self.SOI and jpeg[-2:] == self.EOI:
                        with self._lock:
                            self._latest = jpeg
                sock.close()
                self.connected = False
            except Exception as e:
                self.connected = False
                print(f"[Stream] 重連中: {e}")
                time.sleep(2.0)


# ════════════════════════════════════════════════
#  資料庫
# ════════════════════════════════════════════════
def load_rules():
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(
            "SELECT name,h_min,s_min,v_min,h_max,s_max,v_max FROM cargo_rules")
        rules = {}
        for row in cur.fetchall():
            c, hl, sl, vl, hh, sh, vh = row
            rules[c] = {
                "lower": np.array([hl, sl, vl], dtype=np.uint8),
                "upper": np.array([hh, sh, vh], dtype=np.uint8),
            }
        con.close()
        return rules
    except Exception as e:
        print(f"[DB] 載入規則失敗: {e}")
        return {}


# ════════════════════════════════════════════════
#  ROI 處理
# ════════════════════════════════════════════════
def extract_roi(frame, contour, target_size=ROI_SIZE):
    """從色彩輪廓抓出 ROI 並產生彩色 + 黑白二值化雙版本。"""
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < 30 or bh < 30:
        return None, None

    pad_x = max(5, int(bw * 0.10))
    pad_y = max(5, int(bh * 0.10))
    fh, fw = frame.shape[:2]
    rx1, ry1 = max(0, x - pad_x), max(0, y - pad_y)
    rx2, ry2 = min(fw, x + bw + pad_x), min(fh, y + bh + pad_y)
    if rx2 <= rx1 or ry2 <= ry1:
        return None, None

    roi_bgr = frame[ry1:ry2, rx1:rx2].copy()

    h, w = roi_bgr.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    yoff = (side - h) // 2
    xoff = (side - w) // 2
    canvas[yoff:yoff+h, xoff:xoff+w] = roi_bgr
    roi_bgr = cv2.resize(canvas, (target_size, target_size))

    _, _, v_ch = cv2.split(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV))
    _, mask = cv2.threshold(v_ch, 110, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return roi_bgr, mask


# ════════════════════════════════════════════════
#  輔助：只算 mask 檔案數（避免 color/mask 混算）
# ════════════════════════════════════════════════
def count_samples(class_name):
    folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith("_mask.png")])


# ════════════════════════════════════════════════
#  主程式
# ════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  AI 訓練資料收集工具（優化版）")
    print("=" * 60)
    for k, cls in CLASS_KEYS.items():
        print(f"  {chr(k)} = {CLASS_DESC[cls]}")
    print(f"  目標：每類 600 張")
    print("  P/空白 = 暫停  U = 復原  Q = 離開")
    print("=" * 60)

    for cname in CLASS_KEYS.values():
        os.makedirs(os.path.join(DATASET_DIR, cname), exist_ok=True)

    color_rules = load_rules()
    if not color_rules:
        print("[錯誤] 找不到色彩規則，請先執行 update_warehouse_db.py")
        input("按 Enter 結束...")
        return

    reader = MJPEGStreamReader(ESP32_HOST, ESP32_PORT, STREAM_PATH)
    reader.start()

    capture_target = None
    capture_remaining = 0
    paused = False
    last_capture_ts = 0.0
    last_saved_files = []          # [(color_path, mask_path), ...] 供 undo 使用

    # 只計算 mask 檔案數
    class_counts = {c: count_samples(c) for c in CLASS_KEYS.values()}

    print(f"\n[Main] 目前資料數：{class_counts}")
    print("[Main] 等待 ESP32-CAM ...\n")

    # 預先建立空白畫面（用於連線中）
    last_frame = None

    while True:
        jpeg = reader.get_latest_frame()
        key = cv2.waitKey(1) & 0xFF

        # ── 按鍵處理 ──
        if key == ord('q') or key == 27:
            break

        if key == ord('p') or key == ord(' '):
            paused = not paused
            print(f"[暫停] {'已暫停' if paused else '已繼續'}")

        if key == ord('u'):
            # 復原最近一張
            if last_saved_files:
                cp, mp = last_saved_files.pop()
                cls = os.path.basename(os.path.dirname(cp))
                try:
                    if os.path.exists(cp): os.remove(cp)
                    if os.path.exists(mp): os.remove(mp)
                    class_counts[cls] = max(0, class_counts[cls] - 1)
                    print(f"[復原] 已刪除 {cls} 最近一張，剩餘 {class_counts[cls]}")
                except Exception as e:
                    print(f"[復原] 失敗：{e}")
            else:
                print("[復原] 沒有可復原的紀錄")

        if key in CLASS_KEYS and capture_remaining == 0 and not paused:
            capture_target = CLASS_KEYS[key]
            capture_remaining = SAMPLE_COUNT
            print(f"\n[採集] 開始連拍「{capture_target}」x {SAMPLE_COUNT} 張")
            print(f"       請慢慢轉動 / 移動色紙以涵蓋各種角度")

        # ── 解碼 ──
        if jpeg is None:
            if last_frame is not None:
                cv2.imshow("AI Training Data Collector", last_frame)
                cv2.waitKey(1)
            time.sleep(0.01)
            continue

        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # ── 色紙偵測 ──
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_f, w_f = hsv.shape[:2]
        hsv[:, :int(w_f * 0.15)] = 0
        hsv[:, int(w_f * 0.85):] = 0

        best_area, best_cnt, best_box, best_color = 0, None, None, ""
        for name, r in color_rules.items():
            mask = cv2.inRange(hsv, r["lower"], r["upper"])
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                if a > 3500 and a > best_area:
                    best_area, best_cnt, best_box, best_color = a, c, cv2.boundingRect(c), name

        # ── 採集邏輯（含品質檢查） ──
        now_ts = time.time()
        save_status = ""
        if (capture_remaining > 0 and best_cnt is not None and not paused
                and now_ts - last_capture_ts >= SAMPLE_DELAY):
            roi_bgr, roi_mask = extract_roi(frame, best_cnt)

            if roi_bgr is None:
                save_status = "ROI too small"
            else:
                # 品質檢查：黑色像素必須足夠（過濾沒畫圖案的空白色紙）
                black_px = cv2.countNonZero(roi_mask)
                if black_px < MIN_BLACK_PX:
                    save_status = f"no shape ({black_px} px)"
                else:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    folder = os.path.join(DATASET_DIR, capture_target)
                    bgr_path  = os.path.join(folder, f"{ts}_color.png")
                    mask_path = os.path.join(folder, f"{ts}_mask.png")
                    cv2.imwrite(bgr_path, roi_bgr)
                    cv2.imwrite(mask_path, roi_mask)
                    last_saved_files.append((bgr_path, mask_path))
                    if len(last_saved_files) > 20:
                        last_saved_files.pop(0)

                    capture_remaining -= 1
                    last_capture_ts = now_ts
                    class_counts[capture_target] += 1
                    save_status = "saved"

                    if capture_remaining == 0:
                        print(f"[採集] 完成！「{capture_target}」累計 "
                              f"{class_counts[capture_target]}/{TARGET_PER_CLASS.get(capture_target, 600)}")
                        if class_counts[capture_target] >= TARGET_PER_CLASS.get(capture_target, 600):
                            print(f"       ✓ 已達目標！")
                        capture_target = None

        # ════════════════════════════════════
        #  UI 繪製
        # ════════════════════════════════════
        if best_box:
            x, y, bw, bh = best_box
            # 框線：採集中閃爍亮綠，否則灰綠
            box_col = (0, 255, 100) if capture_remaining > 0 else (0, 180, 80)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), box_col, 2)
            # 標籤顯示偵測到的顏色
            cv2.putText(frame, best_color, (x, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 1)
        else:
            # 沒偵測到色紙時的警告
            cv2.putText(frame, "No color detected",
                        (10, h_f - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (80, 80, 220), 1)

        # 頂部狀態列
        if capture_target:
            progress = (SAMPLE_COUNT - capture_remaining) / SAMPLE_COUNT
            bar_w = int(w_f * progress)
            cv2.rectangle(frame, (0, 0), (w_f, 40), (40, 40, 40), -1)
            cv2.rectangle(frame, (0, 0), (bar_w, 40),
                          CLASS_COLORS[capture_target], -1)
            cv2.putText(frame,
                        f"Capturing: {capture_target}  "
                        f"{SAMPLE_COUNT - capture_remaining}/{SAMPLE_COUNT}",
                        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            # 第二行顯示儲存狀態
            if save_status:
                col = (200,255,200) if save_status == "saved" else (150,200,255)
                cv2.putText(frame, f"-> {save_status}",
                            (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        if paused:
            cv2.putText(frame, "[ PAUSED ]", (w_f//2 - 60, h_f//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,180,255), 2)

        # ── 右側面板 ──
        panel = np.zeros((h_f, 320, 3), dtype=np.uint8)
        cv2.putText(panel, "DATASET COUNT", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,220), 2)
        cv2.line(panel, (8,33), (312,33), (50,50,50), 1)

        # 連線狀態指示
        conn_col = (0,200,80) if reader.connected else (60,60,200)
        cv2.circle(panel, (300, 22), 5, conn_col, -1)

        for i, (cname, cnt) in enumerate(class_counts.items()):
            iy = 60 + i * 42
            bgr = CLASS_COLORS[cname]
            key_label = f"{i+1}={cname}"
            mark = "✓" if cnt >= TARGET_PER_CLASS.get(cname, 600) else " "
            cv2.putText(panel, f"{mark} {key_label}", (10, iy-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200,255,200) if cnt >= TARGET_PER_CLASS.get(cname, 600) else (180,180,180), 1)

            # 進度條
            bar_max = TARGET_PER_CLASS.get(cname, 600)
            bar = min(1.0, cnt / bar_max)
            cv2.rectangle(panel, (10, iy), (10 + int(280 * bar), iy + 12), bgr, -1)
            cv2.rectangle(panel, (10, iy), (290, iy + 12), (40,40,40), 1)
            cv2.putText(panel, f"{cnt}/{TARGET_PER_CLASS.get(cname, 600)}",
                        (115, iy + 11), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (240,240,240), 1)

        # 統計與提示
        total = sum(class_counts.values())
        target_total = sum(TARGET_PER_CLASS.values())
        done_classes = sum(1 for c, v in class_counts.items() if v >= TARGET_PER_CLASS.get(c, 600))
        cv2.line(panel, (8, h_f-80), (312, h_f-80), (50,50,50), 1)
        cv2.putText(panel, f"Total: {total}/{target_total}",
                    (10, h_f-58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,220,220), 1)
        cv2.putText(panel, f"Classes done: {done_classes}/{len(CLASS_KEYS)}",
                    (10, h_f-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100,255,100) if done_classes == len(CLASS_KEYS) else (180,180,180), 1)
        cv2.putText(panel, "1-4=capture P=pause", (10, h_f-22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140,140,140), 1)
        cv2.putText(panel, "U=undo Q=quit", (10, h_f-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140,140,140), 1)

        display = np.hstack((frame, panel))
        last_frame = display
        cv2.imshow("AI Training Data Collector", display)

    reader.request_stop()
    cv2.destroyAllWindows()

    # 結束報告
    print("\n" + "=" * 60)
    print("  資料收集統計：")
    all_done = True
    for cname, cnt in class_counts.items():
        if cnt >= TARGET_PER_CLASS.get(cname, 600):
            status = f"✓ 充足"
        else:
            status = f"✗ 不足，還差 {TARGET_PER_CLASS.get(cname, 600) - cnt} 張"
            all_done = False
        print(f"    {cname:10s} : {cnt:4d}/{TARGET_PER_CLASS.get(cname, 600)}  {status}")
    print(f"\n  資料儲存位置：{os.path.abspath(DATASET_DIR)}/")
    if all_done:
        print("\n  🎉 所有類別都達到目標！可以開始訓練：python train_model.py")
    else:
        print("\n  ⚠ 還有類別不足，建議再執行一次本工具補齊資料")
    print("=" * 60)


if __name__ == "__main__":
    main()
