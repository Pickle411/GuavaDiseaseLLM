import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2. 畫混淆矩陣
# =========================
def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# 3. 測試函式
# =========================
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing", leave=False)
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

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    test_loss = running_loss / total_samples
    test_acc = running_corrects / total_samples

    return test_loss, test_acc, y_true, y_pred


# =========================
# 4. 主程式
# =========================
def main():
    set_seed(42)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.abspath(os.path.join(current_dir, "..", "dataset"))
    test_dir = os.path.join(dataset_root, "test")

    runs_root = os.path.join(current_dir, "runs")
    train_save_dir = os.path.join(runs_root, "train")
    test_save_dir = os.path.join(runs_root, "test")

    model_path = os.path.join(train_save_dir, "best_model.pth")
    report_path = os.path.join(test_save_dir, "test_classification_report.txt")
    cm_img_path = os.path.join(test_save_dir, "test_confusion_matrix.png")
    result_json_path = os.path.join(test_save_dir, "test_result.json")

    batch_size = 32
    num_workers = 4
    image_size = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== Environment ==========")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Test dir:  {test_dir}")
    print(f"Model:     {model_path}")
    print(f"Save dir:  {test_save_dir}")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"找不到 test 資料夾: {test_dir}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型權重: {model_path}")

    os.makedirs(test_save_dir, exist_ok=True)

    # -------------------------
    # Transform
    # -------------------------
    weights = EfficientNet_B0_Weights.DEFAULT
    test_transform = transforms.Compose([
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
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    class_names = test_dataset.classes
    num_classes = len(class_names)

    print("\n========== Dataset Info ==========")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Test samples: {len(test_dataset)}")

    # -------------------------
    # 建立模型
    # -------------------------
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # -------------------------
    # Loss
    # -------------------------
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Evaluate
    # -------------------------
    test_loss, test_acc, y_true, y_pred = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    print("\n========== Test Result ==========")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=cm_img_path
    )

    result_dict = {
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "class_names": class_names
    }

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4)

    print(f"\n測試結果已儲存到: {test_save_dir}")


if __name__ == "__main__":
    main()