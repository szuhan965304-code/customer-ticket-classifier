import sys
from pathlib import Path

# 解決 Windows 終端中文亂碼
sys.stdout.reconfigure(encoding="utf-8")

import joblib
import jieba

# ⚠️ 必須跟訓練時 pipeline 用的 tokenizer 名稱完全一致
def jieba_tokenizer(text: str):
    text = "" if text is None else str(text)
    return list(jieba.cut(text))


# ====== 路徑設定（穩定寫法）======

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# 選一個模型（只留一行）
model_path = MODEL_DIR / "svm_tfidf.joblib"
# model_path = MODEL_DIR / "logistic_tfidf.joblib"


# ====== 載入模型 ======
model = joblib.load(model_path)


# ====== 測試預測 ======
text = "我要退貨"

pred = model.predict([text])[0]

print("輸入：", text)
print("預測類別：", pred)