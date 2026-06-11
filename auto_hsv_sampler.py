"""
auto_hsv_sampler.py  ── 自動化 HSV 色彩採樣與校準系統
================================================================
功能：
  - MJPEG 串流即時顯示（多執行緒跳幀，零積壓）
  - 十字準星 + 5×5 ROI 區域採樣
  - 連續 100 幀採樣，顯示進度條與倒數
  - 自動計算 H/S/V 上下限 + 寬容度緩衝
  - 按 6 輸出可直接貼上至 update_warehouse_db.py 的程式碼

操作說明：
  1-5  → 開始採樣對應顏色
  6    → 輸出校準結果
  r    → 清除所有採樣資料重來
  q    → 離開

修正紀錄 v2：
  - 修正 _stop / stop() → _quit / request_stop()，避免 threading 衝突
  - 修正 ESP32_URL 未使用 → 統一改為 ESP32_HOST
  - 修正 dir() bug → 直接使用 ESP32_HOST
  - 修正採樣輸出 label 格式：Cargo (Pink) → Cargo A (Pink)，與資料庫一致
================================================================
"""

import cv2
import numpy as np
import threading
import socket
import time
import sys

# ─────────────────────────────────────────────
#  設定區
# ─────────────────────────────────────────────
ESP32_HOST   = "172.20.10.3"    # ← 與其他程式統一使用 ESP32_HOST
STREAM_PATH  = "/"
ESP32_PORT   = 80

ROI_HALF      = 5             # ROI 半徑：畫面中心 ± 5 px → 11×11 區域
SAMPLE_FRAMES = 100           # 每次採樣的幀數
H_BUFFER      = 2             # H 值上下緩衝
SV_BUFFER     = 15            # S、V 值上下緩衝

RECV_CHUNK   = 65536

# 顏色定義：按鍵 → (Cargo 完整名稱, BGR顯示色)
# 名稱格式與 update_warehouse_db.py / cargo_rules 表完全一致
COLOR_MAP = {
    ord('1'): ("Cargo D (Green)",  (  0, 200, 100)),
    ord('2'): ("Cargo C (Yellow)", (  0, 215, 255)),
    ord('3'): ("Cargo A (Pink)",   (180, 105, 255)),
    ord('4'): ("Cargo B (Orange)", (  0, 165, 255)),
    ord('5'): ("Cargo X (Blue)",   (255, 140,   0)),   # 備用，一般不用
}
KEY_LABELS = {
    ord('1'): "1=Green(D)",
    ord('2'): "2=Yellow(C)",
    ord('3'): "3=Pink(A)",
    ord('4'): "4=Orange(B)",
    ord('5'): "5=Blue(X)",
}
# ─────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════
#  Thread-1：MJPEG 串流讀取（跳幀設計）
#  修正：_stop / stop() → _quit / request_stop()
# ═══════════════════════════════════════════════════════════════

class MJPEGStreamReader(threading.Thread):
    SOI = b'\xff\xd8'
    EOI = b'\xff\xd9'

    def __init__(self, host, port, path):
        super().__init__(daemon=True, name="StreamReader")
        self.host, self.port, self.path = host, port, path
        self._lock   = threading.Lock()
        self._latest = None
        self._quit   = threading.Event()   # ← 修正：_stop → _quit
        self.connected = False

    def get_latest_frame(self):
        with self._lock:
            f = self._latest
            self._latest = None
        return f

    def request_stop(self):                # ← 修正：stop() → request_stop()
        self._quit.set()

    def _connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((self.host, self.port))
        req = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Connection: keep-alive\r\n\r\n"
        )
        sock.sendall(req.encode())
        return sock

    def run(self):
        while not self._quit.is_set():     # ← 修正：_stop → _quit
            try:
                sock = self._connect()
                self.connected = True
                buf = bytearray()

                while not self._quit.is_set():   # ← 修正
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

                    with self._lock:
                        self._latest = jpeg

                sock.close()
                self.connected = False
            except Exception as e:
                self.connected = False
                print(f"[StreamReader] 連線失敗: {e}，2 秒後重試…")
                time.sleep(2.0)


# ═══════════════════════════════════════════════════════════════
#  採樣狀態機
# ═══════════════════════════════════════════════════════════════

class SamplerState:
    IDLE     = "IDLE"
    SAMPLING = "SAMPLING"

    def __init__(self):
        self.state        = self.IDLE
        self.color_name   = ""
        self.color_bgr    = (255, 255, 255)
        self.samples_h    = []
        self.samples_s    = []
        self.samples_v    = []
        self.frame_count  = 0
        self.results      = {}   # cargo_name → (h_lo,s_lo,v_lo,h_hi,s_hi,v_hi)

    def start_sampling(self, cargo_name, color_bgr):
        self.state       = self.SAMPLING
        self.color_name  = cargo_name
        self.color_bgr   = color_bgr
        self.samples_h   = []
        self.samples_s   = []
        self.samples_v   = []
        self.frame_count = 0
        print(f"\n[採樣] 開始採樣「{cargo_name}」→ 請將色紙對準十字準星…")

    def feed_roi(self, roi_bgr):
        """傳入 ROI BGR 小圖，計算平均 HSV 並記錄。"""
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mean    = cv2.mean(roi_hsv)[:3]  # (H_avg, S_avg, V_avg)
        self.samples_h.append(mean[0])
        self.samples_s.append(mean[1])
        self.samples_v.append(mean[2])
        self.frame_count += 1

        if self.frame_count >= SAMPLE_FRAMES:
            self._finalize()

    def _finalize(self):
        h_arr = np.array(self.samples_h)
        s_arr = np.array(self.samples_s)
        v_arr = np.array(self.samples_v)

        h_lo = max(0,   int(h_arr.min()) - H_BUFFER)
        h_hi = min(179, int(h_arr.max()) + H_BUFFER)
        s_lo = max(0,   int(s_arr.min()) - SV_BUFFER)
        s_hi = min(255, int(s_arr.max()) + SV_BUFFER)
        v_lo = max(0,   int(v_arr.min()) - SV_BUFFER)
        v_hi = min(255, int(v_arr.max()) + SV_BUFFER)

        self.results[self.color_name] = (h_lo, s_lo, v_lo, h_hi, s_hi, v_hi)

        print(f"\n[完成] {self.color_name}")
        print(f"  H: {h_arr.min():.1f}~{h_arr.max():.1f}  → 加緩衝後 [{h_lo}, {h_hi}]")
        print(f"  S: {s_arr.min():.1f}~{s_arr.max():.1f}  → 加緩衝後 [{s_lo}, {s_hi}]")
        print(f"  V: {v_arr.min():.1f}~{v_arr.max():.1f}  → 加緩衝後 [{v_lo}, {v_hi}]")

        self.state = self.IDLE

    def reset(self):
        self.__init__()
        print("[重置] 所有採樣資料已清除。")

    def print_results(self):
        if not self.results:
            print("\n[輸出] 尚無任何採樣結果！請先按 1-5 進行採樣。")
            return

        print("\n" + "═" * 60)
        print("  ✅  校準結果 ── 可直接複製貼上至 update_warehouse_db.py")
        print("═" * 60)
        print("calibrated_rules = [")
        for cargo_name, (h_lo, s_lo, v_lo, h_hi, s_hi, v_hi) in self.results.items():
            print(f'    ("{cargo_name}", {h_lo:>4}, {s_lo:>4}, {v_lo:>4},  '
                  f'{h_hi:>4}, {s_hi:>4}, {v_hi:>4}),')
        print("]")
        print()
        print("# 確認格式符合 update_warehouse_db.py 中的 calibrated_rules")
        print("═" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
#  UI 繪製函式
# ═══════════════════════════════════════════════════════════════

def draw_ui(frame, sampler: SamplerState, cx: int, cy: int):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── 半透明頂部資訊列 ────────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 52), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # ── 按鍵提示 ────────────────────────────────────────────
    hint_x = 6
    for key, label in KEY_LABELS.items():
        cargo_name = COLOR_MAP[key][0]
        bgr = COLOR_MAP[key][1]
        done_mark = "✓ " if cargo_name in sampler.results else "  "
        text = done_mark + label
        cv2.putText(frame, text, (hint_x, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1, cv2.LINE_AA)
        hint_x += 100

    cv2.putText(frame, "6=Export  r=Reset  q=Quit",
                (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # ── 十字準星 ────────────────────────────────────────────
    cross_color = (0, 255, 255)
    cross_len   = 18
    cv2.line(frame, (cx - cross_len, cy), (cx + cross_len, cy), cross_color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - cross_len), (cx, cy + cross_len), cross_color, 1, cv2.LINE_AA)

    # ── ROI 方框 ────────────────────────────────────────────
    r = ROI_HALF
    roi_color = sampler.color_bgr if sampler.state == SamplerState.SAMPLING else (0, 255, 255)
    cv2.rectangle(frame, (cx - r, cy - r), (cx + r, cy + r), roi_color, 1, cv2.LINE_AA)

    # ROI 中心點即時 HSV 顯示
    roi_patch = frame[cy - r: cy + r + 1, cx - r: cx + r + 1]
    if roi_patch.size > 0:
        hsv_patch = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2HSV)
        mean_hsv  = cv2.mean(hsv_patch)[:3]
        hsv_text  = f"H:{mean_hsv[0]:.0f} S:{mean_hsv[1]:.0f} V:{mean_hsv[2]:.0f}"
        cv2.putText(frame, hsv_text, (cx + r + 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    # ── 採樣進度條 ───────────────────────────────────────────
    if sampler.state == SamplerState.SAMPLING:
        progress = sampler.frame_count / SAMPLE_FRAMES
        bar_w    = w - 12
        bar_y    = h - 30
        bar_fill = int(bar_w * progress)

        cv2.rectangle(frame, (6, bar_y), (6 + bar_w, bar_y + 14), (40, 40, 40), -1)
        if bar_fill > 0:
            cv2.rectangle(frame, (6, bar_y), (6 + bar_fill, bar_y + 14),
                          sampler.color_bgr, -1)
        cv2.rectangle(frame, (6, bar_y), (6 + bar_w, bar_y + 14), (120, 120, 120), 1)

        remaining = SAMPLE_FRAMES - sampler.frame_count
        prog_text = (f"採樣中「{sampler.color_name}」："
                     f"{sampler.frame_count}/{SAMPLE_FRAMES} 幀  "
                     f"剩 {remaining} 幀")
        cv2.putText(frame, prog_text, (6, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, sampler.color_bgr, 1, cv2.LINE_AA)

    elif sampler.state == SamplerState.IDLE:
        done_list = list(sampler.results.keys())
        status = f"已完成: {done_list}  │  按 1-5 採樣，6 輸出結果"
        cv2.putText(frame, status, (6, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 200, 160), 1, cv2.LINE_AA)

    # ── 顯示已完成顏色的小色塊 ──────────────────────────────
    if sampler.results:
        swatch_x = w - 10
        swatch_y = 60
        for cname, vals in sampler.results.items():
            h_mid = (vals[0] + vals[3]) // 2
            s_mid = (vals[1] + vals[4]) // 2
            v_mid = (vals[2] + vals[5]) // 2
            hsv_pixel = np.uint8([[[h_mid, s_mid, v_mid]]])
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
            bgr_color = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
            swatch_x -= 28
            cv2.rectangle(frame, (swatch_x, swatch_y),
                          (swatch_x + 22, swatch_y + 22), bgr_color, -1)
            cv2.rectangle(frame, (swatch_x, swatch_y),
                          (swatch_x + 22, swatch_y + 22), (200, 200, 200), 1)
            cv2.putText(frame, cname[6],  # 'A'/'B'/'C'/'D'
                        (swatch_x + 7, swatch_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)

    return frame


# ═══════════════════════════════════════════════════════════════
#  主程式
# ═══════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  HSV 色彩採樣校準工具  ─  auto_hsv_sampler.py  v2")
    print("═" * 60)
    print(f"  連線目標：{ESP32_HOST}:{ESP32_PORT}")
    print(f"  ROI 大小：{ROI_HALF*2+1} × {ROI_HALF*2+1} px")
    print(f"  採樣幀數：{SAMPLE_FRAMES} 幀/色彩")
    print(f"  緩衝值　：H ±{H_BUFFER}，S/V ±{SV_BUFFER}")
    print("─" * 60)
    print("  按鍵：1=Green(D)  2=Yellow(C)  3=Pink(A)  4=Orange(B)")
    print("        6=輸出結果  r=重置  q=離開")
    print("═" * 60 + "\n")

    # 修正：直接使用 ESP32_HOST，不再用 dir() 判斷
    reader = MJPEGStreamReader(ESP32_HOST, ESP32_PORT, STREAM_PATH)
    reader.start()

    sampler     = SamplerState()
    window_name = "HSV 採樣校準工具"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 520)

    print("[Main] 等待串流連線…", end="", flush=True)
    wait_start = time.time()
    while True:
        f = reader.get_latest_frame()
        if f is not None:
            break
        if time.time() - wait_start > 10:
            print("\n[錯誤] 無法取得影像，請確認 ESP32-CAM IP 與連線。")
            reader.request_stop()           # ← 修正：stop() → request_stop()
            sys.exit(1)
        time.sleep(0.1)
        print(".", end="", flush=True)
    print(" 連線成功！\n")

    placeholder = None

    while True:
        jpeg = reader.get_latest_frame()
        if jpeg is not None:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                placeholder = frame.copy()
        else:
            if placeholder is None:
                time.sleep(0.01)
                continue
            frame = placeholder.copy()

        fh, fw = frame.shape[:2]
        cx, cy = fw // 2, fh // 2

        if sampler.state == SamplerState.SAMPLING:
            r = ROI_HALF
            roi = frame[max(0, cy - r): cy + r + 1,
                        max(0, cx - r): cx + r + 1]
            if roi.size > 0:
                sampler.feed_roi(roi)

        frame = draw_ui(frame, sampler, cx, cy)

        display = cv2.resize(frame, (fw * 2, fh * 2),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key in COLOR_MAP:
            if sampler.state == SamplerState.SAMPLING:
                print(f"[警告] 正在採樣「{sampler.color_name}」，請等待完成。")
            else:
                cargo_name, bgr = COLOR_MAP[key]
                sampler.start_sampling(cargo_name, bgr)

        elif key == ord('6'):
            sampler.print_results()

        elif key == ord('r'):
            sampler.reset()

    reader.request_stop()              # ← 修正：stop() → request_stop()
    cv2.destroyAllWindows()
    print("\n[Main] 已結束。")


if __name__ == "__main__":
    main()
