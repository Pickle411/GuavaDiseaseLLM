import sys
from pathlib import Path

import gradio as gr

# ===== 確保可以找到專案根目錄 =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.model_inference import GuavaClassifierInference
from llm.llm_response import generate_llm_response


# ===== 全域初始化模型，只載入一次 =====
predictor = GuavaClassifierInference()


def format_top_k(top_k_results: list) -> str:
    lines = []
    for i, item in enumerate(top_k_results, start=1):
        lines.append(
            f"{i}. {item['class']} ({item['class_zh']}): {item['score']:.4f}"
        )
    return "\n".join(lines)


def analyze_image(image_path):
    if image_path is None:
        return (
            None,
            "未提供圖片",
            "未提供圖片",
            "N/A",
            "N/A",
            "N/A",
            "請先上傳圖片。"
        )

    try:
        # 1. CV 推論
        prediction_result = predictor.predict(image_path)

        predicted_class = prediction_result["predicted_class"]
        predicted_class_zh = prediction_result["predicted_class_zh"]
        confidence = prediction_result["confidence"]
        top_k_text = format_top_k(prediction_result["top_k"])
        is_low_confidence = "是" if prediction_result["is_low_confidence"] else "否"
        requires_review = "是" if prediction_result["requires_review"] else "否"

        # 2. LLM 解釋
        llm_result = generate_llm_response(prediction_result)
        response_text = llm_result["response_text"]

        return (
            image_path,
            predicted_class,
            predicted_class_zh,
            f"{confidence:.4f}",
            is_low_confidence,
            requires_review,
            top_k_text + "\n\n" + response_text
        )

    except Exception as e:
        return (
            image_path,
            "錯誤",
            "錯誤",
            "錯誤",
            "錯誤",
            "錯誤",
            f"執行過程發生錯誤：\n{str(e)}"
        )


with gr.Blocks(title="Guava Disease LLM Demo") as demo:
    gr.Markdown("# Guava Disease Classification + LLM Demo")
    gr.Markdown("上傳一張芭樂圖片，系統會先進行分類，再用 LLM 產生說明。")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="filepath",
                label="上傳芭樂圖片"
            )
            submit_btn = gr.Button("開始分析")

        with gr.Column():
            output_image = gr.Image(label="圖片預覽")
            predicted_class = gr.Textbox(label="預測類別 (英文)")
            predicted_class_zh = gr.Textbox(label="預測類別 (中文)")
            confidence = gr.Textbox(label="信心值")
            is_low_confidence = gr.Textbox(label="是否低信心")
            requires_review = gr.Textbox(label="是否建議人工複檢")

    output_text = gr.Textbox(
        label="Top-K 與 LLM 說明",
        lines=18
    )

    submit_btn.click(
        fn=analyze_image,
        inputs=input_image,
        outputs=[
            output_image,
            predicted_class,
            predicted_class_zh,
            confidence,
            is_low_confidence,
            requires_review,
            output_text,
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)