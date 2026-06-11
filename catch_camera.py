"""
catch_camera.py  ── 智慧倉儲庫存系統 v5（穩定版）
================================================================
修復：
  - threading._stop 命名衝突 (TypeError: 'Event' not callable)
  - JPEG 損毀容錯
  - solidity 策略圖形辨識
================================================================
"""

import cv2
import numpy as np
import sqlite3
import threading
import queue
import socket
import time
from datetime import datetime

# ════════════════════════════════════════════════════════════════
#  設定區
# ════════════════════════════════════════════════════════════════
ESP32_HOST      = "172.20.10.3"
ESP32_PORT      = 80
STREAM_PATH     = "/"
DB_PATH         = "warehouse.db"

CONFIRM_FRAMES  = 10
MAX_STOCK       = 8           # 每種貨物的倉庫上限
COOLDOWN_SEC    = 2.0
RECONNECT_DELAY = 2.0
RECV_CHUNK      = 65536

INIT_V_THRESH   = 110
INIT_POLY_EPS   = 5
INIT_MIN_PX     = 60

WIN_MAIN  = "Smart Warehouse"
WIN_DEBUG = "Shape Inspector"
TB_THRESH = "V-Thresh"
TB_EPS    = "PolyDP x100"
TB_MINPX  = "Min Pixels"

SHAPE_RULES = {
    "Cargo A (Pink)":   "square",
    "Cargo B (Orange)": "triangle",
    "Cargo C (Yellow)": "hexagram",
    "Cargo D (Green)":  "cross",
}

CARGO_BGR = {
    "Cargo A (Pink)":   (180, 105, 255),
    "Cargo B (Orange)": (  0, 165, 255),
    "Cargo C (Yellow)": (  0, 215, 255),
    "Cargo D (Green)":  (  0, 200, 100),
}

SHAPE_ICON = {
    "square":"[]", "triangle":"^",
    "hexagram":"*", "cross":"X",
}


# ════════════════════════════════════════════════════════════════
#  Thread-1：MJPEG 串流讀取
#  注意：停止旗標改名為 _quit，避免與 threading.Thread._stop 衝突
# ════════════════════════════════════════════════════════════════
class MJPEGStreamReader(threading.Thread):
    SOI = b'\xff\xd8'
    EOI = b'\xff\xd9'

    def __init__(self, host, port, path):
        super().__init__(daemon=True, name="StreamReader")
        self.host, self.port, self.path = host, port, path
        self._lock    = threading.Lock()
        self._latest  = None
        self._quit    = threading.Event()   # ← _stop 改為 _quit
        self.recv_fps = 0.0
        self._fps_cnt = 0
        self._fps_ts  = time.time()

    def get_latest_frame(self):
        with self._lock:
            f = self._latest
            self._latest = None
        return f

    def request_stop(self):          # ← stop() 改為 request_stop()
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
                    buf  = buf[end + 2:]

                    # JPEG 基本完整性檢查（避免 corrupt data 進入辨識）
                    if len(jpeg) > 100 and jpeg[:2] == self.SOI and jpeg[-2:] == self.EOI:
                        with self._lock:
                            self._latest = jpeg

                    self._fps_cnt += 1
                    now = time.time()
                    if now - self._fps_ts >= 1.0:
                        self.recv_fps = self._fps_cnt / (now - self._fps_ts)
                        self._fps_cnt = 0
                        self._fps_ts  = now

                sock.close()
            except Exception as e:
                print(f"[Stream] 連線失敗: {e}，{RECONNECT_DELAY}s 後重試")
                time.sleep(RECONNECT_DELAY)


# ════════════════════════════════════════════════════════════════
#  Thread-3：非同步 SQLite 寫入
#  注意：停止旗標改名為 _quit，避免與 threading.Thread._stop 衝突
# ════════════════════════════════════════════════════════════════
class DBWriter(threading.Thread):
    def __init__(self, db_path):
        super().__init__(daemon=True, name="DBWriter")
        self.db_path = db_path
        self.q       = queue.Queue()
        self._quit   = threading.Event()    # ← _stop 改為 _quit

    def enqueue(self, cargo_name, delta=1):
        try:
            self.q.put_nowait((cargo_name, delta))
        except queue.Full:
            pass

    def request_stop(self):                 # ← stop() 改為 request_stop()
        self._quit.set()

    def run(self):
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        cur = con.cursor()

        while not self._quit.is_set():
            try:
                name, delta = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            now = datetime.now().strftime("%H:%M:%S")
            cur.execute(
                "UPDATE inventory SET quantity=quantity+? WHERE cargo_name=?",
                (delta, name))
            con.commit()
            print(f"[DB] {name} +{delta} ({now})")

        con.close()


# ════════════════════════════════════════════════════════════════
#  資料庫載入
# ════════════════════════════════════════════════════════════════
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

def remove_one(cargo_name):
    """從指定貨物中出庫一個（數量 -1，最低為 0）。"""
    try:
        con = sqlite3.connect(DB_PATH, timeout=1)
        cur = con.cursor()
        cur.execute(
            "UPDATE inventory SET quantity = MAX(0, quantity - 1) WHERE cargo_name = ?",
            (cargo_name,))
        con.commit()
        con.close()
    except Exception as e:
        print(f"[DB] 出庫失敗: {e}")


def clear_inventory():
    """清空所有庫存歸零。"""
    try:
        con = sqlite3.connect(DB_PATH, timeout=1)
        cur = con.cursor()
        cur.execute("UPDATE inventory SET quantity = 0")
        con.commit()
        con.close()
        print("[DB] 所有庫存已清空！")
    except Exception as e:
        print(f"[DB] 清空失敗: {e}")


def load_inventory():
    try:
        con = sqlite3.connect(DB_PATH, timeout=1)
        cur = con.cursor()
        cur.execute("SELECT cargo_name, quantity FROM inventory")
        inv = {r[0]: r[1] for r in cur.fetchall()}
        con.close()
        return inv
    except Exception:
        return {}



# ════════════════════════════════════════════════════════════════
#  AI 模型載入（如果有 shape_model.onnx 則使用 AI，否則退回幾何判定）
# ════════════════════════════════════════════════════════════════
import os as _os

MODEL_PATH       = "shape_model.onnx"
AI_CLASSES       = ["square", "triangle", "hexagram", "cross", "circle"]
AI_INPUT_SIZE    = 224
AI_CONFIDENCE_TH = 0.75   # AI 預測信心度門檻，低於此值視為 unknown

_ai_net = None
_use_ai = False
if _os.path.exists(MODEL_PATH):
    try:
        _ai_net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        _use_ai = True
        print(f"[AI] 已載入模型：{MODEL_PATH}")
    except Exception as _e:
        print(f"[AI] 模型載入失敗，退回幾何判定：{_e}")
        _use_ai = False
else:
    print(f"[AI] 找不到 {MODEL_PATH}，使用幾何判定（規則式）")


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _classify_with_ai(mask):
    """
    用 ONNX 模型預測圖形類別。
    輸入：黑色線條二值化 mask（任意尺寸）
    回傳：(shape_name, confidence) 或 ("unknown", 0.0)
    """
    global _ai_net
    if _ai_net is None:
        return "unknown", 0.0

    # 縮放到 64x64
    img = cv2.resize(mask, (AI_INPUT_SIZE, AI_INPUT_SIZE),
                     interpolation=cv2.INTER_AREA)
    # 標準化到 [-1, 1]（與訓練時的 Normalize([0.5], [0.5]) 一致）
    img_f = img.astype(np.float32) / 255.0
    img_f = (img_f - 0.5) / 0.5

    # 形狀：(1, 1, 64, 64)
    blob = img_f.reshape(1, 1, AI_INPUT_SIZE, AI_INPUT_SIZE)
    _ai_net.setInput(blob)
    logits = _ai_net.forward().flatten()
    probs  = _softmax(logits)
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])

    if conf < AI_CONFIDENCE_TH:
        return "unknown", conf
    # circle 是負樣本，預測到 circle 視為 unknown（拒絕入庫）
    if AI_CLASSES[idx] == "circle":
        return "unknown", conf
    return AI_CLASSES[idx], conf


# ════════════════════════════════════════════════════════════════
#  圖形辨識核心（Solidity 策略）
#
#  策略：對二值化遮罩膨脹填實空心線條 → 計算 solidity
#
#  solidity = 輪廓面積 / 凸包面積，有凹角時偏低：
#    circle   → ~1.00  (無凹角)
#    square   → ~0.95  (無凹角)
#    triangle → ~0.93  (無凹角)
#    hexagram → ~0.55~0.78  (6個凹角)
#    cross(X) → ~0.50~0.70  (4個凹角) + 對角分佈
# ════════════════════════════════════════════════════════════════

def _get_feat(mask, eps_k):
    """
    膨脹填實空心線條後提取幾何特徵（旋轉不變版）。

    關鍵改動：
      - 使用 minAreaRect（最小外接旋轉矩形）取代 boundingRect
        → 菱形不會被判成「歪斜的矩形」，而是被判成「旋轉45度的正方形」
      - 新增 rot_ar：旋轉後的長寬比（旋轉不變）
      - 新增 rect_fill：輪廓面積 / minAreaRect 面積
        → 正方形/長方形 ≈ 0.95，菱形也 ≈ 0.95（因為菱形其實是正方形）
        → 圓形 ≈ 0.78（π/4），三角形 ≈ 0.5
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    filled = cv2.dilate(mask, k, iterations=2)
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt  = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 80:
        return None
    peri = cv2.arcLength(cnt, True)
    if peri < 1:
        return None
    ci       = 4 * np.pi * area / (peri ** 2)
    hull     = cv2.convexHull(cnt)
    h_area   = cv2.contourArea(hull)
    solidity = area / h_area if h_area > 0 else 1.0
    approx   = cv2.approxPolyDP(cnt, eps_k * peri, True)
    v        = len(approx)

    # 一般外接矩形（用於 cross_score 旋轉檢測）
    _, _, bw, bh = cv2.boundingRect(cnt)
    ar = bw / bh if bh > 0 else 1.0

    # 最小外接旋轉矩形（旋轉不變）
    rect = cv2.minAreaRect(cnt)  # ((cx,cy), (w,h), angle)
    (rw, rh) = rect[1]
    if rw < 1 or rh < 1:
        rot_ar     = 1.0
        rect_fill  = 0.0
    else:
        # rot_ar 永遠 >= 1（長/短）
        rot_ar    = max(rw, rh) / min(rw, rh)
        rect_fill = area / (rw * rh)

    return {
        "ci": ci, "v": v, "sol": solidity, "ar": ar,
        "rot_ar": rot_ar,        # 旋轉不變的長寬比（菱形也 ≈ 1.0）
        "rect_fill": rect_fill,  # 對 minAreaRect 的填充率
        "cnt": cnt, "hull": hull,
    }


def _cross_score(mask):
    """
    X/+ 形判定（旋轉不變版）。

    把 mask 分成 3x3 格，計算：
      cross_x = 對角四角的像素佔比（純 X 形時最高）
      cross_p = 上下左右四邊中點的像素佔比（純 + 形時最高）
    回傳 max(cross_x, cross_p)，這樣 X 旋轉成 + 也能偵測到。
    """
    h, w = mask.shape[:2]
    if h < 9 or w < 9:
        return 0.0
    gh, gw = h // 3, w // 3
    def px(r1, c1, r2, c2):
        return float(np.sum(mask[r1:r2, c1:c2] > 0))
    corner = (px(0,    0,    gh,   gw) + px(0,    2*gw, gh,   w) +
              px(2*gh, 0,    h,    gw) + px(2*gh, 2*gw, h,    w))
    edge   = (px(0, gw, gh, 2*gw) + px(gh, 0, 2*gh, gw) +
              px(gh, 2*gw, 2*gh, w) + px(2*gh, gw, h, 2*gw))
    total  = corner + edge
    if total < 20:
        return 0.0
    cross_x = corner / total   # X 形對角分佈
    cross_p = edge   / total   # + 形十字分佈
    return max(cross_x, cross_p)


def _classify(feat, mask):
    """
    旋轉不變版圖形分類。

    核心策略：使用 rot_ar（旋轉後長寬比）取代普通 aspect_ratio
      → 菱形旋轉45度後 rot_ar ≈ 1.0，等同正方形
      → 半月形 rot_ar 較大且 rect_fill 低

    判定順序（不可交換）：
      ① 三角形 : v=3 + rect_fill ≈ 0.5（三角形佔外接矩形一半）
      ② X / + 形 : cross_score 高 + 凹形
      ③ 六角星 : 凹形 + 不是 X 形分佈
      ④ 正方形/菱形 : rot_ar 接近 1 + 高 solidity
      ⑤ 長方形 : rot_ar 較大但 rect_fill 高
      ⑥ 半月/弧形 : rect_fill 低 + 凸形
    """
    ci        = feat["ci"]
    v         = feat["v"]
    sol       = feat["sol"]
    rot_ar    = feat["rot_ar"]     # ≥1，旋轉不變
    rect_fill = feat["rect_fill"]
    xs        = _cross_score(mask)

    # ① 三角形：3 頂點 OR rect_fill 約 0.5（三角形塞滿外接矩形的一半）
    if v == 3 and sol > 0.78:
        return "triangle"
    if v <= 4 and 0.40 <= rect_fill <= 0.62 and sol > 0.85:
        return "triangle"

    # ② X 形 / + 形：cross_score 高 + 有凹角（條件更嚴格，避免六角星誤判）
    if xs > 0.55 and sol < 0.75:
        return "cross"

    # ③ 六角星：solidity 低（有凹角），條件放寬（現場 sol=0.81 可正確辨識）
    if sol < 0.95:
        return "hexagram"

    # ④ 正方形/菱形：rot_ar 接近 1（旋轉45度的菱形也算）+ 無凹角
    if rot_ar <= 1.60 and sol > 0.85 and rect_fill > 0.80:
        return "square"

    # ⑤ 長方形也歸類為 square（容忍正方形畫得不太方正）
    if rot_ar <= 2.20 and sol > 0.85 and rect_fill > 0.78 and 4 <= v <= 8:
        return "square"

    # ⑥ 半月/弧形：填充率低 + 凸形 → 不歸入任何已知類別（避免誤判）
    #    返回 unknown 讓使用者明白「圖案太歪需要重畫」

    return "unknown"


def detect_shape(contour, frame, v_thresh, eps_k, min_px, draw_frame=None):
    """回傳 (shape_str, debug_pkg)"""
    empty = {"roi": None, "mask": None, "hull": None, "feat": None, "n": 0}

    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < 20 or bh < 20:
        return "unknown", empty

    pad_x = max(3, int(bw * 0.05))
    pad_y = max(3, int(bh * 0.05))
    fh, fw = frame.shape[:2]
    rx1, ry1 = max(0, x+pad_x),     max(0, y+pad_y)
    rx2, ry2 = min(fw, x+bw-pad_x), min(fh, y+bh-pad_y)
    if rx2 <= rx1 or ry2 <= ry1:
        return "unknown", empty

    roi = frame[ry1:ry2, rx1:rx2].copy()

    _, _, v_ch = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    _, mask = cv2.threshold(v_ch, v_thresh, 255, cv2.THRESH_BINARY_INV)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

    black_px = cv2.countNonZero(mask)
    roi_px   = mask.shape[0] * mask.shape[1]
    # 黑色像素太少（沒有圖形）→ unknown
    if black_px < min_px:
        return "unknown", {"roi": roi, "mask": mask, "hull": None, "feat": None, "n": 0}
    # 黑色像素佔比太高（>60%）→ 整片都是暗色，不是圖形
    if black_px > roi_px * 0.6:
        return "unknown", {"roi": roi, "mask": mask, "hull": None, "feat": None, "n": 0}

    feat = _get_feat(mask, eps_k)
    if feat is None:
        return "unknown", {"roi": roi, "mask": mask, "hull": None, "feat": None, "n": 0}

    # ── 圖形分類：優先用 AI，否則退回幾何判定 ──
    ai_conf = 0.0
    if _use_ai:
        shape, ai_conf = _classify_with_ai(mask)
    else:
        shape = _classify(feat, mask)

    hull_disp = feat["hull"].reshape(-1, 1, 2)

    if draw_frame is not None:
        offset = np.array([[[rx1, ry1]]])
        raw_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in raw_cnts:
            if cv2.contourArea(c) > 5:
                cv2.drawContours(draw_frame, [c + offset], -1, (200, 50, 200), 1)
        cv2.drawContours(draw_frame,
                         [(feat["cnt"] + offset.reshape(1, 2))], -1, (0, 230, 200), 2)

    return shape, {
        "roi":  roi,
        "mask": mask,
        "hull": hull_disp,
        "feat": {"v": feat["v"], "ci": feat["ci"],
                 "ar": feat["ar"], "sol": feat["sol"],
                 "ai_conf": ai_conf, "use_ai": _use_ai},
        "n": 1,
    }


# ════════════════════════════════════════════════════════════════
#  除錯視窗（預配置畫布）
# ════════════════════════════════════════════════════════════════
_DW, _DH = 120, 120
_DGAP    = 4
_DC_W    = _DW * 3 + _DGAP * 2
_DC_H    = _DH + 50
_dbg_buf = np.zeros((_DC_H, _DC_W, 3), dtype=np.uint8)


def _draw_debug(pkg, det_shape, req_shape, tb_v, tb_e, tb_m):
    global _dbg_buf
    _dbg_buf[:] = 25

    def put(img, ox, oy):
        ih, iw = img.shape[:2]
        _dbg_buf[oy:oy+ih, ox:ox+iw] = img

    if pkg["roi"] is not None:
        put(cv2.resize(pkg["roi"], (_DW, _DH)), 0, 0)
    else:
        cv2.putText(_dbg_buf, "no color det", (2, _DH//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60,60,60), 1)

    if pkg["mask"] is not None:
        bm = cv2.cvtColor(cv2.resize(pkg["mask"], (_DW, _DH)), cv2.COLOR_GRAY2BGR)
        put(bm, _DW + _DGAP, 0)
    else:
        cv2.putText(_dbg_buf, "no mask", (_DW+_DGAP+2, _DH//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60,60,60), 1)

    ox2 = (_DW + _DGAP) * 2
    if pkg["hull"] is not None and pkg["roi"] is not None:
        tmp = np.zeros((_DH, _DW, 3), dtype=np.uint8)
        oh, ow = pkg["roi"].shape[:2]
        sc  = np.array([_DW/max(ow,1), _DH/max(oh,1)], dtype=np.float32)
        sh  = (pkg["hull"].reshape(-1,2) * sc).astype(np.int32).reshape(-1,1,2)
        cv2.polylines(tmp, [sh], True, (0,230,200), 2)
        if pkg["feat"]:
            f = pkg["feat"]
            cv2.putText(tmp, f"v={f['v']} sol={f['sol']:.2f}",
                        (2, _DH-30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,0), 1)
            cv2.putText(tmp, f"rotAR={f.get('rot_ar',0):.2f} fill={f.get('rect_fill',0):.2f}",
                        (2, _DH-18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,0), 1)
            # sol 進度條：綠=高(無凹角) 紅=低(有凹角)
            sol_w = int(f["sol"] * (_DW-4))
            col   = (0,180,0) if f["sol"] > 0.82 else (0,80,220)
            cv2.rectangle(tmp, (2, _DH-10), (2+sol_w, _DH-4), col, -1)
            cv2.rectangle(tmp, (2, _DH-10), (_DW-2,   _DH-4), (60,60,60), 1)
        put(tmp, ox2, 0)
    else:
        cv2.putText(_dbg_buf, "no hull", (ox2+2, _DH//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60,60,60), 1)

    for i, lbl in enumerate(["ROI", "BkMask", "Hull+sol"]):
        cv2.putText(_dbg_buf, lbl, (i*(_DW+_DGAP)+2, _DH+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120,120,120), 1)

    ok  = det_shape == req_shape
    col = (0,200,80) if ok else (0,80,255)
    cv2.putText(_dbg_buf, f"Det:{det_shape}", (_DC_W-168, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    cv2.putText(_dbg_buf, f"Req:{req_shape}", (_DC_W-168, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110,110,110), 1)

    cv2.putText(_dbg_buf, f"Thresh:{tb_v} Eps:{tb_e/100:.2f} MinPx:{tb_m}",
                (4, _DH+30), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0,190,190), 1)
    cv2.putText(_dbg_buf, "sol<0.82=concave  sol>0.88=convex",
                (4, _DH+44), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (70,70,70), 1)

    cv2.imshow(WIN_DEBUG, _dbg_buf)


# ════════════════════════════════════════════════════════════════
#  主程式
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 56)
    print("  智慧倉儲系統 v5  ──  顏色 + 圖形雙重驗證")
    mode = "AI 模型辨識" if _use_ai else "幾何規則辨識"
    print(f"  圖形辨識模式：{mode}")
    print("  q/ESC=離開  r=重載庫存  c=清空倉庫")
    print("  出庫：5=粉色-1  6=橘色-1  7=黃色-1  8=綠色-1")
    print(f"  倉庫上限：每種貨物最多 {MAX_STOCK} 個")
    print("=" * 56)

    color_rules = load_rules()
    if not color_rules:
        print("[錯誤] 找不到色彩規則，請先執行 update_warehouse_db.py")
        input("按 Enter 結束...")
        return

    # 建立除錯視窗與 Trackbar
    _ph = np.full((_DC_H, _DC_W, 3), 25, dtype=np.uint8)
    cv2.putText(_ph, "Waiting...", (4, _DC_H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1)
    cv2.imshow(WIN_DEBUG, _ph)
    cv2.waitKey(1)
    cv2.createTrackbar(TB_THRESH, WIN_DEBUG, INIT_V_THRESH, 200, lambda v: None)
    cv2.createTrackbar(TB_EPS,    WIN_DEBUG, INIT_POLY_EPS, 15,  lambda v: None)
    cv2.createTrackbar(TB_MINPX,  WIN_DEBUG, INIT_MIN_PX,   300, lambda v: None)

    db_writer = DBWriter(DB_PATH)
    db_writer.start()

    reader = MJPEGStreamReader(ESP32_HOST, ESP32_PORT, STREAM_PATH)
    reader.start()

    inventory     = load_inventory()
    pass_counters = {c: 0   for c in color_rules}
    cooldowns     = {c: 0.0 for c in color_rules}
    last_color    = ""

    last_pkg = {"roi": None, "mask": None, "hull": None, "feat": None, "n": 0}
    last_det = "---"
    last_req = "---"

    proc_n, proc_ts, proc_fps = 0, time.time(), 0.0
    ui_h = 240

    print("[Main] 等待攝影機連線...")

    while True:
        # 讀 Trackbar
        try:
            tb_v = cv2.getTrackbarPos(TB_THRESH, WIN_DEBUG)
            tb_e = cv2.getTrackbarPos(TB_EPS,    WIN_DEBUG)
            tb_m = cv2.getTrackbarPos(TB_MINPX,  WIN_DEBUG)
        except Exception:
            tb_v, tb_e, tb_m = INIT_V_THRESH, INIT_POLY_EPS, INIT_MIN_PX
        eps_k = max(0.01, tb_e / 100.0)

        # 按鍵偵測（每次迴圈必須執行，否則 GUI 無回應）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            inventory = load_inventory()
            print("[Main] 庫存已重載")
        if key == ord('c'):
            clear_inventory()
            inventory = load_inventory()
            print("[Main] 倉庫已清空，所有貨物歸零")

        # 出庫：按 5~8 對應貨物各減 1
        remove_map = {
            ord('5'): "Cargo A (Pink)",
            ord('6'): "Cargo B (Orange)",
            ord('7'): "Cargo C (Yellow)",
            ord('8'): "Cargo D (Green)",
        }
        if key in remove_map:
            rname = remove_map[key]
            cur_qty = inventory.get(rname, 0)
            if cur_qty > 0:
                remove_one(rname)
                inventory[rname] = cur_qty - 1
                print(f"[出庫] {rname} -1 → 剩餘 {inventory[rname]}/{MAX_STOCK}")
            else:
                print(f"[出庫] {rname} 已經是 0，無法出庫")

        jpeg = reader.get_latest_frame()
        if jpeg is None:
            _draw_debug(last_pkg, last_det, last_req, tb_v, tb_e, tb_m)
            time.sleep(0.005)
            continue

        arr   = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        ui_h, w_f = frame.shape[:2]

        # Phase 1：HSV 顏色辨識
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :int(w_f*0.15)] = 0
        hsv[:, int(w_f*0.85):] = 0

        best_name, best_area, best_cnt, best_box = "", 0, None, None
        for name, r in color_rules.items():
            mask = cv2.inRange(hsv, r["lower"], r["upper"])
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                if a > 3500 and a > best_area:
                    best_area, best_name, best_cnt, best_box = \
                        a, name, c, cv2.boundingRect(c)

        now_ts = time.time()

        # Phase 2：圖形辨識
        shape_ok  = False
        det_shape = "---"
        req_shape = "---"
        cur_pkg   = last_pkg

        if best_name and best_cnt is not None:
            req_shape = SHAPE_RULES.get(best_name, "")
            if req_shape:
                det_shape, cur_pkg = detect_shape(
                    best_cnt, frame, tb_v, eps_k, tb_m, draw_frame=frame)
                last_pkg = cur_pkg
                last_det = det_shape
                last_req = req_shape
                shape_ok = (det_shape == req_shape)
            else:
                shape_ok = True

        # Phase 3：計數與入庫
        if best_name and best_box:
            x, y, bw, bh = best_box
            box_col = (0,220,80) if shape_ok else (0,80,255)
            cv2.rectangle(frame, (x,y), (x+bw,y+bh), box_col, 2)
            cv2.putText(frame, best_name,
                        (x, y-22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 2)
            cv2.putText(frame, f"Shape:{det_shape} {'PASS' if shape_ok else 'FAIL'}",
                        (x, y-8),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_col, 1)

            if cur_pkg["feat"]:
                f = cur_pkg["feat"]
                if f.get("use_ai"):
                    txt = f"AI: {det_shape} (conf={f.get('ai_conf',0):.2f})"
                    tcol = (0, 255, 100) if f.get('ai_conf', 0) > 0.85 else (0, 200, 220)
                else:
                    txt = f"v={f['v']} sol={f['sol']:.2f} rotAR={f.get('rot_ar',0):.2f}"
                    tcol = (200, 200, 0)
                cv2.putText(frame, txt,
                            (x, y+bh+16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, tcol, 1)

            if shape_ok:
                if best_name == last_color:
                    pass_counters[best_name] += 1
                else:
                    pass_counters = {c: 0 for c in color_rules}
                    pass_counters[best_name] = 1
                    last_color = best_name

                prog = pass_counters[best_name] / CONFIRM_FRAMES
                cv2.rectangle(frame, (x, y+bh+3),
                              (x+int(bw*prog), y+bh+9), (0,220,120), -1)
                cv2.rectangle(frame, (x, y+bh+3),
                              (x+bw, y+bh+9), (60,60,60), 1)

                if (pass_counters[best_name] >= CONFIRM_FRAMES
                        and now_ts > cooldowns[best_name]):
                    cur_qty = inventory.get(best_name, 0)
                    if cur_qty >= MAX_STOCK:
                        # 已滿，不入庫，顯示警告
                        cv2.putText(frame, "FULL!", (x, y+bh+26),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        pass_counters[best_name] = 0
                        cooldowns[best_name]     = now_ts + COOLDOWN_SEC
                        print(f"[FULL] {best_name} 已滿（{MAX_STOCK}/{MAX_STOCK}），拒絕入庫")
                    else:
                        db_writer.enqueue(best_name, 1)
                        cooldowns[best_name]     = now_ts + COOLDOWN_SEC
                        pass_counters[best_name] = 0
                        inventory[best_name]     = cur_qty + 1
                        print(f"[OK] {best_name} 入庫！累計={inventory[best_name]}/{MAX_STOCK}")
        else:
            last_color    = ""
            pass_counters = {c: 0 for c in color_rules}

        # FPS
        proc_n += 1
        if now_ts - proc_ts >= 1.0:
            proc_fps = proc_n / (now_ts - proc_ts)
            proc_n, proc_ts = 0, now_ts

        # UI 面板
        panel = np.zeros((ui_h, 300, 3), dtype=np.uint8)
        cv2.putText(panel, "LIVE INVENTORY", (10,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,220), 2)
        cv2.line(panel, (8,32), (292,32), (50,50,50), 1)

        for i, (cname, sname) in enumerate(SHAPE_RULES.items()):
            iy   = 50 + i*16
            bgr  = CARGO_BGR.get(cname, (180,180,180))
            icon = SHAPE_ICON.get(sname, "?")
            cv2.putText(panel, f"{icon} {cname}", (10,iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, bgr, 1)

        cv2.line(panel, (8,140), (292,140), (50,50,50), 1)

        bar_max = MAX_STOCK
        for i, (cname, qty) in enumerate(inventory.items()):
            iy  = 158 + i*26
            bgr = CARGO_BGR.get(cname, (180,180,180))
            bw2 = int(270 * qty / bar_max)
            cv2.rectangle(panel, (10,iy-12), (10+bw2,iy-3), bgr, -1)
            cv2.rectangle(panel, (10,iy-12), (280,   iy-3), (40,40,40), 1)
            short = cname.replace("Cargo ","").replace(" ","")
            full_mark = " FULL" if qty >= MAX_STOCK else ""
            txt_col = (0,0,255) if qty >= MAX_STOCK else (220,220,220)
            cv2.putText(panel, f"{short}:{qty}/{MAX_STOCK}{full_mark}", (12,iy-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1)

        cv2.putText(panel, f"Net:{reader.recv_fps:.1f} Sys:{proc_fps:.1f} fps",
                    (10,ui_h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80,80,80), 1)
        cv2.putText(panel, "q=quit r=reload c=clear 5-8=remove",
                    (10,ui_h-6),  cv2.FONT_HERSHEY_SIMPLEX, 0.30, (60,60,60), 1)

        cv2.imshow(WIN_MAIN, np.hstack((frame, panel)))
        _draw_debug(last_pkg, last_det, last_req, tb_v, tb_e, tb_m)

    # 清理
    print("[Main] 正在關閉...")
    reader.request_stop()
    db_writer.request_stop()
    db_writer.join(timeout=2)
    cv2.destroyAllWindows()
    print("[Main] 已結束。")


if __name__ == "__main__":
    main()