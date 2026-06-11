"""
train_model.py  ── AI 圖形辨識模型訓練腳本
================================================================
功能：
  讀取 dataset/ 資料夾的圖形照片，訓練一個 CNN 分類器，
  輸出 shape_model.onnx（OpenCV 可直接載入，無需 PyTorch 環境）。

執行：
  python train_model.py

訓練前置（首次需要安裝）：
  pip install torch torchvision pillow onnx

輸出：
  shape_model.onnx   ← 訓練好的模型（給 catch_camera.py 用）
  training_log.txt   ← 訓練過程記錄
  confusion_matrix.png ← 驗證集混淆矩陣（觀察哪類最容易誤判）

電腦規格：
  CPU 訓練：約 5~15 分鐘（看資料量）
  GPU 訓練：約 1~3 分鐘（會自動偵測 CUDA）
================================================================
"""

import os
import sys
import glob
import random
import time

# ── 檢查依賴 ──
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    import numpy as np
except ImportError as e:
    print("=" * 60)
    print("  缺少必要套件！請執行：")
    print("  pip install torch torchvision pillow onnx")
    print("=" * 60)
    print(f"\n  錯誤詳情：{e}")
    sys.exit(1)


# ════════════════════════════════════════════════
#  設定
# ════════════════════════════════════════════════
DATASET_DIR   = "dataset"
MODEL_PATH    = "shape_model.onnx"
LOG_PATH      = "training_log.txt"

CLASSES       = ["square", "triangle", "hexagram", "cross", "circle"]
IMG_SIZE      = 224          # MobileNetV2 標準輸入尺寸
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 0.0005      # MobileNetV2 fine-tune 用較小 lr
VAL_RATIO     = 0.2          # 驗證集比例

# 用 mask 黑白版訓練（輪廓資訊最關鍵）
USE_MASK      = True


# ════════════════════════════════════════════════
#  資料集類別
# ════════════════════════════════════════════════
class ShapeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


# ════════════════════════════════════════════════
#  CNN 模型架構
# ════════════════════════════════════════════════
class ShapeCNN(nn.Module):
    """
    輸入：1×64×64 灰階圖
    輸出：4 類別 logits
    參數量：約 540K，CPU 推論 < 10ms
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64 → 32
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32 → 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 16 → 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)




# ════════════════════════════════════════════════
#  MobileNetV2（灰階輸入版）
# ════════════════════════════════════════════════
class MobileNetV2Gray(nn.Module):
    """
    MobileNetV2 修改版：
    - 第一層 Conv2d 從 3 通道改為 1 通道（接受灰階輸入）
    - 最後分類層改為 n 類
    - 使用 ImageNet 預訓練權重
    """
    def __init__(self, num_classes=5):
        super().__init__()
        from torchvision import models
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        orig_conv = base.features[0][0]
        new_conv  = nn.Conv2d(
            1, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )
        import torch
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                orig_conv.weight.mean(dim=1, keepdim=True)
            )
        base.features[0][0] = new_conv
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

# ════════════════════════════════════════════════
#  資料載入
# ════════════════════════════════════════════════
def load_samples():
    samples = []
    counts = {}
    suffix = "_mask.png" if USE_MASK else "_color.png"

    for idx, cls in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(folder):
            print(f"[警告] 找不到資料夾：{folder}")
            counts[cls] = 0
            continue
        files = glob.glob(os.path.join(folder, f"*{suffix}"))
        for f in files:
            samples.append((f, idx))
        counts[cls] = len(files)
    return samples, counts


def split_train_val(samples, val_ratio=0.2):
    random.seed(42)
    random.shuffle(samples)
    n_val = int(len(samples) * val_ratio)
    return samples[n_val:], samples[:n_val]


# ════════════════════════════════════════════════
#  訓練流程
# ════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total   += imgs.size(0)
    return total_loss / total, correct / total


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total   += imgs.size(0)
            for t, p in zip(labels.cpu().numpy(), pred.cpu().numpy()):
                confusion[t, p] += 1
    return total_loss / total, correct / total, confusion


# ════════════════════════════════════════════════
#  主程式
# ════════════════════════════════════════════════
def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 60)
    log("  AI 圖形辨識模型訓練")
    log("=" * 60)

    # ── 1. 掃描資料 ──
    log("\n[1/5] 掃描資料集 ...")
    samples, counts = load_samples()
    log(f"     資料夾：{os.path.abspath(DATASET_DIR)}/")
    for cls, n in counts.items():
        mark = "✓" if n >= 600 else "⚠"
        log(f"     {mark} {cls:10s} : {n} 張")

    if not samples or min(counts.values()) < 20:
        log("\n[錯誤] 資料不足！每類至少需要 20 張。")
        log("       建議數量：每類 100~200 張")
        log("       請先執行 collect_dataset.py 採集更多資料。")
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        return

    log(f"\n     總計 {len(samples)} 張，分割為訓練 / 驗證集 ...")
    train_samples, val_samples = split_train_val(samples, VAL_RATIO)
    log(f"     訓練集：{len(train_samples)} 張")
    log(f"     驗證集：{len(val_samples)} 張")

    # ── 2. 資料增強 ──
    log("\n[2/5] 設定資料增強 ...")
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(180),               # 任意旋轉（旋轉不變）
        transforms.RandomAffine(0,
                               translate=(0.1, 0.1),  # 隨機平移
                               scale=(0.85, 1.15)),   # 隨機縮放
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    log("     - 隨機旋轉 360°（讓模型學會任意角度）")
    log("     - 隨機平移、縮放、翻轉")

    train_ds = ShapeDataset(train_samples, train_transform)
    val_ds   = ShapeDataset(val_samples,   val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── 3. 模型 ──
    log("\n[3/5] 建立模型 ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"     使用裝置：{device}")
    if device.type == "cuda":
        log(f"     GPU：{torch.cuda.get_device_name(0)}")

    model     = MobileNetV2Gray(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"     參數量：{n_params:,}")

    # ── 4. 訓練 ──
    log(f"\n[4/5] 開始訓練（{EPOCHS} epochs）...")
    log(f"     {'Epoch':>6} {'TrainLoss':>11} {'TrainAcc':>10} {'ValLoss':>10} {'ValAcc':>9}")
    log("     " + "-" * 50)

    best_val_acc = 0.0
    best_state   = None
    t_start      = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, conf = eval_model(model, val_loader, criterion, device)
        scheduler.step()

        log(f"     {epoch:>6d} {tr_loss:>11.4f} {tr_acc:>9.1%} "
            f"{val_loss:>10.4f} {val_acc:>8.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_conf  = conf.copy()

    elapsed = time.time() - t_start
    log(f"\n     訓練完成！耗時 {elapsed:.1f} 秒，最佳驗證準確率：{best_val_acc:.1%}")

    # ── 5. 匯出 ONNX ──
    log("\n[5/5] 匯出 ONNX 模型 ...")
    model.load_state_dict(best_state)
    model.eval()

    dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model.cpu(),
        dummy,
        MODEL_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input":  {0: "batch"},
                      "output": {0: "batch"}},
    )
    size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    log(f"     ✓ {MODEL_PATH}  ({size_mb:.2f} MB)")

    # ── 混淆矩陣 ──
    log("\n  混淆矩陣（驗證集）：")
    log("     " + " " * 12 + "  ".join(f"{c:>9}" for c in CLASSES) + "  (預測)")
    for i, cls in enumerate(CLASSES):
        row = "  ".join(f"{best_conf[i,j]:>9d}" for j in range(len(CLASSES)))
        log(f"     {cls:>10s}  {row}")
    log("     (實際)")

    # ── 各類準確率 ──
    log("\n  各類準確率：")
    for i, cls in enumerate(CLASSES):
        total = best_conf[i].sum()
        correct = best_conf[i, i]
        acc = correct / total if total > 0 else 0
        mark = "✓" if acc >= 0.90 else "⚠"
        log(f"     {mark} {cls:10s} : {acc:.1%}  ({correct}/{total})")

    # 寫入 log
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"\n  訓練記錄已存：{LOG_PATH}")

    log("\n" + "=" * 60)
    log("  🎉 訓練完成！下一步：")
    log(f"     1. 把 {MODEL_PATH} 放在跟 catch_camera.py 同一個資料夾")
    log(f"     2. 執行新版 catch_camera.py（會自動載入模型）")
    log("=" * 60)


if __name__ == "__main__":
    main()
