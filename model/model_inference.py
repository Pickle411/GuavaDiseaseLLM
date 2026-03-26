import torch
import torch.nn.functional as F
from torchvision import models

from common.config import (
    DEVICE,
    MODEL_NAME,
    NUM_CLASSES,
    TOP_K,
    WEIGHT_PATH,
    LOW_CONFIDENCE_THRESHOLD,
    CONFUSION_GAP_THRESHOLD,
)
from model.preprocess import preprocess_image
from model.class_mapping import get_class_name, get_class_zh_by_idx


class GuavaClassifierInference:
    def __init__(self, weight_path=WEIGHT_PATH, device=DEVICE):
        self.weight_path = str(weight_path)
        self.device = device
        self.model = self._load_model()
        self.model.eval()

    def _build_model(self):
        """
        建立和訓練時相同架構的模型。
        目前預設使用 EfficientNet-B0。
        """
        if MODEL_NAME == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)
            return model

        raise ValueError(f"Unsupported model name: {MODEL_NAME}")

    def _load_model(self):
        """
        兼容兩種常見存法：
        1. state_dict
        2. checkpoint dict (包含 model_state_dict)
        3. 整個 model 物件
        """
        checkpoint = torch.load(self.weight_path, map_location=self.device)

        # case 1: checkpoint 是整個 model 物件
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint.to(self.device)
            return model

        # case 2/3: checkpoint 是 dict
        model = self._build_model()

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                # 有些人直接 torch.save(model.state_dict(), path)
                try:
                    model.load_state_dict(checkpoint)
                except Exception as e:
                    raise RuntimeError(
                        "無法載入權重。請確認 guava_classifier.pth 是 state_dict、"
                        "checkpoint dict，或整個 model 物件。"
                    ) from e
        else:
            raise RuntimeError("不支援的權重格式。")

        model = model.to(self.device)
        return model

    def predict(self, image_path: str):
        """
        輸入圖片路徑，輸出標準化預測結果。
        """
        image_tensor = preprocess_image(image_path).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)[0]

        topk_probs, topk_indices = torch.topk(probs, k=TOP_K)

        predicted_idx = topk_indices[0].item()
        predicted_class = get_class_name(predicted_idx)
        predicted_class_zh = get_class_zh_by_idx(predicted_idx)
        confidence = topk_probs[0].item()

        top_k_results = []
        for score, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
            top_k_results.append({
                "class": get_class_name(idx),
                "class_zh": get_class_zh_by_idx(idx),
                "score": round(float(score), 4)
            })

        is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

        requires_review = False
        if len(top_k_results) >= 2:
            score_gap = top_k_results[0]["score"] - top_k_results[1]["score"]
            if score_gap < CONFUSION_GAP_THRESHOLD:
                requires_review = True

        result = {
            "predicted_class": predicted_class,
            "predicted_class_zh": predicted_class_zh,
            "confidence": round(float(confidence), 4),
            "top_k": top_k_results,
            "is_low_confidence": is_low_confidence,
            "requires_review": requires_review,
        }
        return result


if __name__ == "__main__":
    # 這裡換成你要測試的圖片路徑
    test_image_path = "/mnt/shared/chuanyu_m11317028/GuavaDiseaseLLM/sample_images/Anthracnose.png"

    predictor = GuavaClassifierInference()
    result = predictor.predict(test_image_path)

    print("Prediction Result:")
    print(result)