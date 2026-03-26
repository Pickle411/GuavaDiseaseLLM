import os
from groq import Groq

from llm.knowledge_base import get_knowledge_by_class
from llm.prompt_builder import build_llm_prompt


def get_llm_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("找不到 GROQ_API_KEY，請先在環境變數中設定。")
    return Groq(api_key=api_key)


def generate_llm_response(prediction_result: dict, model_name: str = "llama-3.1-8b-instant") -> dict:
    """
    根據模型預測結果，生成 LLM 說明。
    回傳:
    {
        "knowledge": ...,
        "prompt": ...,
        "response_text": ...
    }
    """
    predicted_class = prediction_result["predicted_class"]

    knowledge = get_knowledge_by_class(predicted_class)
    prompt = build_llm_prompt(prediction_result, knowledge)

    client = get_llm_client()

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是一個協助說明芭樂影像分類結果的 AI 助手，請用繁體中文回答。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    response_text = completion.choices[0].message.content.strip()

    return {
        "knowledge": knowledge,
        "prompt": prompt,
        "response_text": response_text,
    }


if __name__ == "__main__":
    fake_prediction_result = {
        "predicted_class": "Anthracnose",
        "predicted_class_zh": "炭疽病",
        "confidence": 0.9123,
        "top_k": [
            {"class": "Anthracnose", "class_zh": "炭疽病", "score": 0.9123},
            {"class": "fruit_fly", "class_zh": "果蠅危害", "score": 0.0621},
            {"class": "healthy_guava", "class_zh": "健康芭樂", "score": 0.0256},
        ],
        "is_low_confidence": False,
        "requires_review": False,
    }

    result = generate_llm_response(fake_prediction_result)
    print(result["response_text"])