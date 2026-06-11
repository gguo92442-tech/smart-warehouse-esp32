\# Smart Warehouse — ESP32-CAM + CNN 智慧倉儲辨識系統



基於 ESP32-CAM 與 MobileNetV2 之智慧倉儲貨物辨識系統，透過「HSV 色彩辨識 + CNN 圖形驗證」雙重防呆機制實現自動化入庫管理。



> 第十八屆全國大專院校行動通訊專題創意競賽作品

> 龍華科技大學 資訊網路工程系｜指導老師：王友俊｜組員：郭庭佑、林文生



\## 系統特色



\- \*\*雙重驗證防呆\*\*：顏色（HSV）+ 圖案（CNN）皆通過、且連續 10 幀穩定，才觸發入庫

\- \*\*三執行緒流水線\*\*：串流跳幀讀取 / 辨識推論 / SQLite WAL 非同步寫入，主迴圈零等待

\- \*\*負樣本拒絕機制\*\*：circle 類別作為負樣本，防止圓形物體誤判入庫

\- \*\*可攜式部署\*\*：一鍵打包含 Embedded Python 的免安裝部署包

\- \*\*換場地快速校準\*\*：HSV 規則存於 SQLite，附自動採樣校準工具



\## 硬體



| 項目 | 規格 |

|---|---|

| 影像端 | ESP32-CAM（AI-Thinker）OV2640 |

| 解析度 | QVGA 320×240，JPEG Quality 15 |

| 傳輸 | WiFi MJPEG over HTTP（TCP Socket 直收）|

| 處理端 | Windows PC + Python 3.12 |



\## 環境安裝



```bash

python -m venv .venv

.venv\\Scripts\\activate

pip install numpy==1.26.4 opencv-python==4.8.1.78 pillow onnx==1.16.2 onnxsim

pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu

```



\## 快速開始



```bash

python init\_inventory.py

python update\_warehouse\_db.py

python catch\_camera.py

```



\## License



MIT

