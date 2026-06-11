"""
verify_model.py  ── AI 模型即時驗證工具
================================================================
用途：
  在不啟動整個倉儲系統的情況下，單獨測試 shape_model.onnx
  的辨識準確度。可以對著色紙慢慢轉動，看 AI 在不同角度的反應。

執行：
  python verify_model.py

操作：
  - 把色紙對準鏡頭，系統會即時顯示 AI 的預測結果
  - 同時顯示信心度（越高越穩定）
  - 數字 1-4：標記你目前正在拿哪種圖形（用來計算準確率）
  - q：離開
================================================================
"""

import cv2
import numpy as np
import socket
import threading
import time
import os
import sqlite3

# ────────────────────────────────────────────────
ESP32_HOST   = "172.20.10.3"
ESP32_PORT   = 80
STREAM_PATH  = "/"
DB_PATH      = "warehouse.db"
MODEL_PATH   = "shape_model.onnx"
RECV_CHUNK   = 65536

AI_CLASSES        = ["square", "triangle", "hexagram", "cross"]
AI_INPUT_SIZE     = 224
AI_CONFIDENCE_TH  = 0.75

V_THRESH     = 110
MIN_PX       = 60
# ────────────────────────────────────────────────

GT_KEYS = {
    ord('1'): "square",
    ord('2'): "triangle",
    ord('3'): "hexagram",
    ord('4'): "cross",
    ord('0'): None,           # 重置標記
}

SHAPE_COLOR = {
    "square":   (180, 105, 255),
    "triangle": (  0, 165, 255),
    "hexagram": (  0, 215, 255),
    "cross":    (  0, 200, 100),
    "unknown":  (100, 100, 100),
}


# ════════════════════════════════════════════════
#  MJPEG 串流
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
            except Exception as e:
                print(f"[Stream] 重連中: {e}")
                time.sleep(2.0)


# ════════════════════════════════════════════════
#  輔助函式
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
    except Exception:
        return {}


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def extract_mask(frame, contour):
    """擷取色紙 ROI 並做黑色線條二值化。"""
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < 20 or bh < 20:
        return None, None
    pad_x = max(3, int(bw * 0.05))
    pad_y = max(3, int(bh * 0.05))
    fh, fw = frame.shape[:2]
    rx1, ry1 = max(0, x+pad_x), max(0, y+pad_y)
    rx2, ry2 = min(fw, x+bw-pad_x), min(fh, y+bh-pad_y)
    if rx2 <= rx1 or ry2 <= ry1:
        return None, None
    roi = frame[ry1:ry2, rx1:rx2].copy()
    _, _, v_ch = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    _, mask = cv2.threshold(v_ch, V_THRESH, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return roi, mask


def predict(net, mask):
    """執行 ONNX 推論。回傳 (預測類別, 信心度, 所有類別機率)。"""
    img = cv2.resize(mask, (AI_INPUT_SIZE, AI_INPUT_SIZE), cv2.INTER_AREA)
    img_f = img.astype(np.float32) / 255.0
    img_f = (img_f - 0.5) / 0.5
    blob = img_f.reshape(1, 1, AI_INPUT_SIZE, AI_INPUT_SIZE)
    net.setInput(blob)
    logits = net.forward().flatten()
    probs  = softmax(logits)
    idx    = int(np.argmax(probs))
    return AI_CLASSES[idx], float(probs[idx]), probs


# ════════════════════════════════════════════════
#  主程式
# ════════════════════════════════════════════════
def main():
    print("=" * 56)
    print("  AI 模型即時驗證工具")
    print("=" * 56)

    if not os.path.exists(MODEL_PATH):
        print(f"\n[錯誤] 找不到模型：{MODEL_PATH}")
        print("       請先執行 train_model.py 訓練模型")
        input("\n按 Enter 結束...")
        return

    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    print(f"[OK] 已載入：{MODEL_PATH}")

    color_rules = load_rules()
    if not color_rules:
        print("[錯誤] 找不到色彩規則")
        return

    reader = MJPEGStreamReader(ESP32_HOST, ESP32_PORT, STREAM_PATH)
    reader.start()

    # 統計用
    gt_label = None    # 你目前手持的真實圖形（標記用）
    stats = {c: {"correct": 0, "total": 0} for c in AI_CLASSES}

    print("\n操作：1=square 2=triangle 3=hexagram 4=cross 0=取消標記 q=離開")
    print("拿著色紙慢慢轉動，按對應數字鍵標記真實圖形，會自動計算準確率。\n")

    while True:
        jpeg = reader.get_latest_frame()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key in GT_KEYS:
            gt_label = GT_KEYS[key]
            if gt_label:
                print(f"[標記] 真實類別 = {gt_label}")
            else:
                print("[標記] 已取消")

        if jpeg is None:
            time.sleep(0.005)
            continue

        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h_f, w_f = frame.shape[:2]

        # 找色紙
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :int(w_f*0.15)] = 0
        hsv[:, int(w_f*0.85):] = 0

        best_area, best_cnt, best_box = 0, None, None
        for name, r in color_rules.items():
            mask = cv2.inRange(hsv, r["lower"], r["upper"])
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                if a > 3500 and a > best_area:
                    best_area, best_cnt, best_box = a, c, cv2.boundingRect(c)

        # 預測
        pred_label = "—"
        pred_conf  = 0.0
        all_probs  = None
        roi_mask   = None

        if best_cnt is not None:
            _, roi_mask = extract_mask(frame, best_cnt)
            if roi_mask is not None and cv2.countNonZero(roi_mask) >= MIN_PX:
                pred_label, pred_conf, all_probs = predict(net, roi_mask)

        # 更新統計（如果有標記）
        if gt_label and pred_label in AI_CLASSES:
            stats[gt_label]["total"] += 1
            if pred_label == gt_label:
                stats[gt_label]["correct"] += 1

        # ── UI ──
        # 色紙框
        if best_box:
            x, y, bw, bh = best_box
            col = SHAPE_COLOR.get(pred_label, (100,100,100))
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), col, 2)
            # 顯示預測
            cv2.putText(frame, f"AI: {pred_label}",
                        (x, y-22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            # 信心度條
            bar_w = int(bw * pred_conf)
            cv2.rectangle(frame, (x, y-12), (x+bar_w, y-6), col, -1)
            cv2.rectangle(frame, (x, y-12), (x+bw, y-6), (60,60,60), 1)
            cv2.putText(frame, f"{pred_conf*100:.0f}%",
                        (x+bw+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        # 右側面板
        panel = np.zeros((h_f, 320, 3), dtype=np.uint8)
        cv2.putText(panel, "AI MODEL VERIFY", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,220), 2)
        cv2.line(panel, (8,33), (312,33), (50,50,50), 1)

        # 四類機率長條圖
        if all_probs is not None:
            for i, cls in enumerate(AI_CLASSES):
                iy = 55 + i * 28
                p  = float(all_probs[i])
                col = SHAPE_COLOR[cls]
                cv2.putText(panel, cls, (10, iy-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1)
                cv2.putText(panel, f"{p*100:5.1f}%", (240, iy-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
                cv2.rectangle(panel, (10, iy), (10+int(220*p), iy+8), col, -1)
                cv2.rectangle(panel, (10, iy), (230,         iy+8), (40,40,40), 1)
        else:
            cv2.putText(panel, "no shape detected", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)

        # 標記狀態
        cv2.line(panel, (8, 180), (312, 180), (50,50,50), 1)
        gt_txt = f"GT label: {gt_label if gt_label else '(none)'}"
        cv2.putText(panel, gt_txt, (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0,220,100) if gt_label else (120,120,120), 1)

        # 統計
        cv2.putText(panel, "Live Accuracy:", (10, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
        for i, cls in enumerate(AI_CLASSES):
            iy = 250 + i * 18
            tot = stats[cls]["total"]
            cor = stats[cls]["correct"]
            acc = cor / tot if tot > 0 else 0
            col = SHAPE_COLOR[cls]
            if tot > 0:
                txt = f"{cls:10s} {acc*100:5.1f}%  ({cor}/{tot})"
            else:
                txt = f"{cls:10s}    -    (0/0)"
            cv2.putText(panel, txt, (10, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        # 操作提示
        cv2.putText(panel, "1-4=mark GT  0=clear  q=quit",
                    (10, h_f-12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120,120,120), 1)

        # 顯示
        cv2.imshow("AI Model Verifier", np.hstack((frame, panel)))

        # 同時顯示模型輸入的 mask（左下小窗）
        if roi_mask is not None:
            cv2.imshow("Model Input (64x64)",
                       cv2.resize(roi_mask, (256, 256), cv2.INTER_NEAREST))

    reader.request_stop()
    cv2.destroyAllWindows()

    # 結束報告
    print("\n" + "=" * 56)
    print("  驗證統計：")
    overall_cor, overall_tot = 0, 0
    for cls in AI_CLASSES:
        tot = stats[cls]["total"]
        cor = stats[cls]["correct"]
        overall_cor += cor
        overall_tot += tot
        acc = cor / tot if tot > 0 else 0
        mark = "✓" if acc >= 0.9 else "⚠" if acc >= 0.7 else "✗"
        print(f"  {mark} {cls:10s} : {acc*100:5.1f}%  ({cor}/{tot})")
    if overall_tot > 0:
        print(f"\n  整體準確率：{overall_cor/overall_tot*100:.1f}%")
    print("=" * 56)


if __name__ == "__main__":
    main()
