# llm/prompt_builder.py

def build_llm_prompt(prediction_result: dict, knowledge: dict) -> str:
    predicted_class = prediction_result["predicted_class"]
    predicted_class_zh = prediction_result["predicted_class_zh"]
    confidence = prediction_result["confidence"]
    is_low_confidence = prediction_result["is_low_confidence"]
    requires_review = prediction_result["requires_review"]
    top_k = prediction_result["top_k"]

    top_k_lines = []
    for item in top_k:
        top_k_lines.append(
            f"- {item['class']} ({item['class_zh']}): {item['score']:.4f}"
        )
    top_k_text = "\n".join(top_k_lines)

    features_text = "\n".join([f"- {x}" for x in knowledge["features"]])
    advice_text = "\n".join([f"- {x}" for x in knowledge["advice"]])

    review_note = "是" if requires_review else "否"
    low_conf_note = "是" if is_low_confidence else "否"

    prompt = f"""
你是一個協助說明芭樂影像分類結果的 AI 助手。
請根據下列模型預測結果與參考知識，用繁體中文產生清楚、簡潔、不要過度誇大的說明。

[模型預測結果]
- 預測類別(英文): {predicted_class}
- 預測類別(中文): {predicted_class_zh}
- 信心值: {confidence:.4f}
- 是否低信心: {low_conf_note}
- 是否建議人工複檢: {review_note}

[Top-K 結果]
{top_k_text}

[參考知識]
- 類別中文名稱: {knowledge["class_name_zh"]}
- 類別摘要: {knowledge["summary"]}

[常見特徵]
{features_text}

[建議處理]
{advice_text}

[注意事項]
{knowledge["warning"]}

請遵守以下規則：
1. 只能根據提供的模型結果與參考知識回答，不要自行虛構額外專業診斷。
2. 若模型為低信心或建議人工複檢，回答中要明確提醒使用者再次確認。
3. 語氣自然、清楚，不要太口語，也不要太學術。
4. 請固定輸出為以下四個段落標題：
【判斷結果】
【結果說明】
【建議處理】
【注意事項】
"""
    return prompt.strip()