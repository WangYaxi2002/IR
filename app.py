# app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import jieba
import re
import os

from SKDCN import SKDCN  # 确保这个文件定义了你的模型类

app = FastAPI(title="垃圾邮件检测系统", version="1.0")

# ===== 全局加载模型（启动时只加载一次）=====
MODEL_PATH = "app.pth"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在，请先训练并保存模型！")

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
vocab = checkpoint["vocab"]
config = checkpoint["config"]

model = SKDCN(
    vocab_size=len(vocab),
    embed_dim=config["embed_dim"],
    num_keywords=config["num_keywords"],
    hidden_dim=config.get("hidden_dim", 512),
    max_len=config.get("max_len", 256),
    num_heads=config.get("num_heads", 4),
    num_layers=config.get("num_layers", 2),
    dropout=config.get("dropout", 0.3),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ 模型加载成功！")


# ===== 预处理函数 =====
def preprocess(text: str, vocab, max_len=512):
    # 清洗
    text = re.sub(r"<[^>]+>", "", text)  # 去 HTML 标签
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = "<PAD>"

    # 分词（中英文混合）
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        tokens = jieba.lcut(text)
    else:
        tokens = text.split()

    # 转 ID
    unk_id = vocab.get("<UNK>", 1)
    pad_id = vocab.get("<PAD>", 0)
    ids = [vocab.get(token, unk_id) for token in tokens]
    ids = ids[:max_len]
    mask = [1] * len(ids)

    # 补 PAD
    while len(ids) < max_len:
        ids.append(pad_id)
        mask.append(0)

    return torch.tensor([ids], dtype=torch.long), torch.tensor([mask], dtype=torch.long)


# ===== 数据模型 =====
class EmailRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    is_spam: bool
    confidence: float
    score: float


# ===== API 接口 =====
@app.post("/predict", response_model=PredictionResponse)
async def predict_spam(request: EmailRequest):
    if not request.text.strip():
        return PredictionResponse(is_spam=False, confidence=0.0, score=0.0)

    input_ids, attention_mask = preprocess(request.text, vocab, config["max_len"])

    with torch.no_grad():
        logits = model(input_ids, attention_mask)  # [1, 1]
        prob = torch.sigmoid(logits).item()  # 转为概率

    return PredictionResponse(
        is_spam=prob > 0.5, confidence=round(prob, 4), score=round(prob, 4)
    )


# ===== 前端页面（可选）=====
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>📧 垃圾邮件检测 (FastAPI)</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 30px auto; padding: 20px; }
            textarea { width: 100%; height: 180px; padding: 10px; box-sizing: border-box; font-size: 14px; }
            button { padding: 12px 24px; font-size: 16px; background: #4CAF50; color: white; border: none; cursor: pointer; margin-top: 10px; }
            button:hover { background: #45a049; }
            #result { margin-top: 20px; padding: 15px; border-radius: 6px; font-weight: bold; }
            .spam { background-color: #ffebee; color: #c62828; border-left: 4px solid #d32f2f; }
            .ham { background-color: #e8f5e9; color: #2e7d32; border-left: 4px solid #388e3c; }
        </style>
    </head>
    <body>
        <h1>📧 垃圾邮件智能检测系统</h1>
        <p>输入邮件内容，AI 将自动判断是否为垃圾邮件。</p>
        <textarea id="emailText" placeholder="例如：恭喜您中奖了！点击领取..."></textarea><br>
        <button onclick="detect()">检测垃圾邮件</button>
        <div id="result"></div>

        <script>
        async function detect() {
            const text = document.getElementById('emailText').value;
            const res = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await res.json();
            
            const resultDiv = document.getElementById('result');
            if (data.is_spam) {
                resultDiv.className = 'spam';
                resultDiv.innerHTML = `⚠️ <strong>垃圾邮件</strong> (置信度: ${data.confidence})`;
            } else {
                resultDiv.className = 'ham';
                resultDiv.innerHTML = `✅ <strong>正常邮件</strong> (置信度: ${(1 - data.confidence).toFixed(4)})`;
            }
        }
        </script>
    </body>
    </html>
    """


# 启动命令：uvicorn app:app --reload --port 8000
