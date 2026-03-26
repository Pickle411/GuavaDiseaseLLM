from pathlib import Path
import torch

# ===== Project Paths =====
PROJECT_ROOT = Path("/mnt/shared/chuanyu_m11317028/GuavaDiseaseLLM")

DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_EXP_DIR = PROJECT_ROOT / "EfficientNet-B0"
WEIGHT_PATH = MODEL_EXP_DIR / "runs" / "train" / "best_model.pth"

SAMPLE_IMAGES_DIR = PROJECT_ROOT / "sample_images"

# ===== Model Settings =====
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 3
IMAGE_SIZE = 224
TOP_K = 3

# 訓練時若有用 ImageNet normalization，這裡要保持一致
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ===== Class Settings =====
CLASS_NAMES = ["Anthracnose", "fruit_fly", "healthy_guava"]

# ===== Device =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Inference Settings =====
LOW_CONFIDENCE_THRESHOLD = 0.60
CONFUSION_GAP_THRESHOLD = 0.15