from PIL import Image
from torchvision import transforms

from common.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def build_inference_transform():
    """
    建立推論時的前處理流程。
    若你訓練時不是這組 transforms，請改成和訓練一致。
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def load_image(image_path: str) -> Image.Image:
    """
    讀取圖片並轉成 RGB。
    """
    image = Image.open(image_path).convert("RGB")
    return image


def preprocess_image(image_path: str):
    """
    讀圖 + 前處理 + 加 batch 維度
    return: tensor with shape [1, 3, H, W]
    """
    image = load_image(image_path)
    transform = build_inference_transform()
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor