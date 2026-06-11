# Smart Warehouse — ESP32-CAM + CNN 智慧倉儲辨識系統

基於 ESP32-CAM 與 MobileNetV2 之智慧倉儲貨物辨識系統，透過「HSV 色彩辨識 + CNN 圖形驗證」雙重防呆機制實現自動化入庫管理。

## 系統特色

- 雙重驗證防呆：顏色（HSV）+ 圖案（CNN）皆通過、且連續 10 幀穩定，才觸發入庫
- 三執行緒流水線：串流跳幀讀取 / 辨識推論 / SQLite WAL 非同步寫入
- 負樣本拒絕機制：circle 類別作為負樣本，防止圓形物體誤判入庫

## 硬體

ESP32-CAM（AI-Thinker）OV2640，QVGA 320x240

## 快速開始

1. python init_inventory.py
2. python update_warehouse_db.py
3. python catch_camera.py

## License

MIT
