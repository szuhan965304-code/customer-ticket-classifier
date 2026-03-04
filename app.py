import re
import math
import traceback
from pathlib import Path

import joblib
import jieba
import numpy as np
import pandas as pd
import streamlit as st


# ===============================
# 1️⃣ tokenizer（必須跟訓練時一致）
# ===============================
def jieba_tokenizer(text: str):
    text = "" if text is None else str(text)
    return list(jieba.cut(text))


# ===============================
# 2️⃣ 路徑設定（雲端最穩）
# app.py 在 repo root
# 模型在 models/ 資料夾
# ===============================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "SVM (LinearSVC)": MODELS_DIR / "svm_tfidf.joblib",
    "Logistic Regression": MODELS_DIR / "logistic_tfidf.joblib",
}


# ===============================
# 3️⃣ 輸入清理
# ===============================
def clean_text(x: str) -> str:
    x = "" if x is None else str(x)
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x


def is_gibberish(x: str):
    if not x:
        return True, "沒有輸入文字"

    if len(x) < 2:
        return True, "請輸入至少 2 個字以上"

    letters_digits = sum(ch.isalnum() for ch in x)
    if letters_digits == 0:
        return True, "內容幾乎都是符號"

    eng = sum(("a" <= ch.lower() <= "z") for ch in x)
    if len(x) >= 6 and eng / len(x) > 0.8:
        return True, "疑似英文亂碼，請用中文描述"

    return False, ""


# ===============================
# 4️⃣ SVM 分數轉近似機率
# ===============================
def softmax(z):
    z = np.array(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


# ===============================
# 5️⃣ 快取模型
# Streamlit cache 用 str 路徑當 key 更穩
# ===============================
@st.cache_resource
def load_model(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"找不到模型檔：{path}")
    return joblib.load(path)


# ===============================
# 6️⃣ 預測 + 信心
# ===============================
def predict_with_confidence(model, text: str):
    pred = model.predict([text])[0]

    conf = None
    top_df = None
    labels = None

    # LogisticRegression 通常有 predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        labels = model.classes_
        conf = float(np.max(proba))
        top_df = (
            pd.DataFrame({"label": labels, "score": proba})
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

    # LinearSVC 常用 decision_function
    elif hasattr(model, "decision_function"):
        scores = model.decision_function([text])
        scores = np.array(scores)

        # binary
        if scores.ndim == 1:
            margin = float(scores[0])
            p_pos = 1 / (1 + math.exp(-margin))
            p_neg = 1 - p_pos
            probs = np.array([p_neg, p_pos])
            labels = model.classes_
            conf = float(np.max(probs))
            top_df = (
                pd.DataFrame({"label": labels, "score": probs})
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )
        else:
            # multiclass
            raw = scores[0]
            probs = softmax(raw)
            labels = model.classes_
            conf = float(np.max(probs))
            top_df = (
                pd.DataFrame({"label": labels, "score": probs})
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )

    return pred, conf, top_df


# ===============================
# 7️⃣ Streamlit UI
# ===============================
def main():
    st.set_page_config(page_title="Customer Ticket NLP", layout="centered")
    st.title("客服工單分類（TF-IDF + ML）")

    # ✅ Debug / health check
    with st.expander("環境檢查"):
        st.write("BASE_DIR:", str(BASE_DIR))
        st.write("MODELS_DIR:", str(MODELS_DIR), "exists =", MODELS_DIR.exists())

        if MODELS_DIR.exists():
            try:
                st.write("models files:", [p.name for p in MODELS_DIR.iterdir()])
            except Exception as e:
                st.write("list models error:", e)

        for k, p in MODEL_FILES.items():
            st.write(f"{k} -> {p} exists = {p.exists()}")

    # ✅ Choose model
    model_choice = st.radio("選擇模型", list(MODEL_FILES.keys()), horizontal=True)
    model_path = MODEL_FILES[model_choice]

    # ✅ Threshold
    threshold = st.slider(
        "信心門檻（低於門檻顯示：不確定）",
        min_value=0.20,
        max_value=0.95,
        value=0.45 if "Logistic" in model_choice else 0.55,
        step=0.01,
    )

    # ✅ Input
    text = st.text_area(
        "輸入一句客服內容",
        height=120,
        placeholder="例如：我要退貨 / 商品有瑕疵 / 一直沒收到貨...",
    )

    # ✅ Predict
    if st.button("預測"):
        try:
            text = clean_text(text)
            bad, reason = is_gibberish(text)
            if bad:
                st.warning(f"⚠️ 無法判斷：{reason}")
                return

            model = load_model(str(model_path))
            pred, conf, top_df = predict_with_confidence(model, text)

            st.success("✅ 流程成功")

            if conf is None:
                st.subheader("預測類別")
                st.write(pred)
                st.caption("此模型沒有信心分數")
            else:
                st.subheader("預測結果")
                st.write(f"預測類別：{pred}")
                st.write(f"信心分數：{conf:.3f}")

                if conf < threshold:
                    st.warning("⚠️ 信心偏低，建議補充更多描述")

            if top_df is not None:
                st.subheader("各類別分數（Top 5）")
                st.dataframe(top_df.head(5), use_container_width=True)

        except Exception as e:
            st.error("❌ 發生錯誤（黑畫面原因）")
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))


if __name__ == "__main__":
    main()
