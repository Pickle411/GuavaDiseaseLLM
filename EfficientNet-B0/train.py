import os
import copy
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


# =========================
# 1. 固定亂數種子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 讓結果更可重現
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2. 畫訓練曲線
# =========================
def plot_training_curves(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
    plt.close()


# =========================
# 3. 單一 epoch 訓練
# =========================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels).item()
        total_samples += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


# =========================
# 4. 單一 epoch 驗證
# =========================
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels).item()
            total_samples += batch_size

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


# =========================
# 5. 主程式
# =========================
def main():
    # -------------------------
    # 基本設定
    # -------------------------
    set_seed(42)

    # 相對路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_root = os.path.abspath(os.path.join(current_dir, "..", "dataset"))
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    test_dir = os.path.join(dataset_root, "test")  # 只檢查結構，不在 train.py 使用

    runs_root = os.path.join(current_dir, "runs")
    train_save_dir = os.path.join(runs_root, "train")
    os.makedirs(train_save_dir, exist_ok=True)

    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    num_workers = 4
    image_size = 224
    early_stopping_patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== Environment ==========")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}")
    print(f"Test dir:  {test_dir}")
    print(f"Save dir:  {train_save_dir}")

    # -------------------------
    # 檢查資料夾是否存在
    # -------------------------
    for path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到資料夾: {path}")

    # -------------------------
    # Data Transform
    # -------------------------
    weights = EfficientNet_B0_Weights.DEFAULT

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std
        )
    ])

    # -------------------------
    # Dataset / DataLoader
    # -------------------------
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("\n========== Dataset Info ==========")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # -------------------------
    # 建立模型
    # -------------------------
    model = models.efficientnet_b0(weights=weights)

    # EfficientNet-B0 最後分類層改成你的類別數
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model = model.to(device)

    # -------------------------
    # Loss / Optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=True
    )

    # -------------------------
    # 訓練紀錄
    # -------------------------
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_no_improve = 0

    print("\n========== Start Training ==========")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # 儲存最新模型
        torch.save(
            model.state_dict(),
            os.path.join(train_save_dir, "last_model.pth")
        )

        # 儲存最佳模型 + Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(
                model.state_dict(),
                os.path.join(train_save_dir, "best_model.pth")
            )
            epochs_no_improve = 0
            print(f"Best model updated. Best Val Acc = {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            print(
                f"\nEarly stopping triggered. "
                f"Validation accuracy did not improve for {early_stopping_patience} consecutive epochs."
            )
            break

    total_time = time.time() - start_time
    print("\n========== Training Finished ==========")
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")

    # 載入最佳權重，確保之後如果你要接續使用，模型是最佳版本
    model.load_state_dict(best_model_wts)

    # -------------------------
    # 存訓練紀錄
    # -------------------------
    with open(os.path.join(train_save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    with open(os.path.join(train_save_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)

    # 畫曲線
    plot_training_curves(history, train_save_dir)

    print(f"\n所有訓練結果已存到: {train_save_dir}")


if __name__ == "__main__":
    main()