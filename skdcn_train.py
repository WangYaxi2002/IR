import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import jieba
import nltk
import re
import os
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from SKDCN import SKDCN


# 设置随机种子（可复现）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 下载 punkt（仅首次运行需要）
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ----------------------------
# 1. 数据预处理与分词
# ----------------------------


def get_tokenizer(language="chinese"):
    """
    支持 'chinese', 'english', 'mixed'
    """
    if language == "chinese":
        return lambda text: jieba.lcut(text)

    elif language == "english":

        def tokenize(text):
            text = re.sub(r"([.,!?;: $$ \"'])", r" \1 ", text)
            return text.split()

        return tokenize

    else:
        chinese_seg = re.compile(r"[\u4e00-\u9fff]+")

        def tokenize(text):
            tokens = []
            last_end = 0
            for match in chinese_seg.finditer(text):
                start, end = match.span()
                if start > last_end:
                    non_chinese = text[last_end:start]
                    cleaned = re.sub(r"([.,!?;: $$ \"'])", r" \1 ", non_chinese)
                    tokens.extend(cleaned.split())
                tokens.extend(jieba.lcut(match.group()))
                last_end = end
            if last_end < len(text):
                non_chinese = text[last_end:]
                cleaned = re.sub(r"([.,!?;: $$ \"'])", r" \1 ", non_chinese)
                tokens.extend(cleaned.split())
            return [t for t in tokens if t.strip()]

        return tokenize
        raise ValueError("language must be 'chinese', 'english', or 'mixed'")


class TextDataset(Dataset):
    def __init__(self, csv_path, vocab, max_len=256, language="chinese"):
        assert os.path.exists(csv_path), f"File not found: {csv_path}"

        df = pd.read_csv(csv_path)
        # 替换原来的 self.labels = df["label"].astype(int).tolist()
        raw_labels = df["label"].values
        # 自动映射常见格式
        if raw_labels.dtype == object:  # 字符串
            label_map = {"ham": 0, "spam": 1, "normal": 0, "abnormal": 1}
            raw_labels = np.array(
                [label_map.get(str(x).lower().strip(), x) for x in raw_labels]
            )
        elif set(np.unique(raw_labels)) == {-1, 1}:
            raw_labels = (raw_labels + 1) // 2  # -1->0, 1->1

        self.labels = raw_labels.astype(int).tolist()
        assert set(self.labels).issubset(
            {0, 1}
        ), f"Labels must be 0/1, got {set(self.labels)}"
        self.contents = df["content"].fillna("").astype(str).tolist()
        self.vocab = vocab
        self.max_len = max_len
        self.tokenize_fn = get_tokenizer(language)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.tokenize_fn(self.contents[idx])
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens[: self.max_len]]
        # 填充到 max_len
        if len(ids) < self.max_len:
            ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))
        else:
            ids = ids[: self.max_len]

        mask = [1 if i != self.vocab["<PAD>"] else 0 for i in ids]
        if sum(mask) == 0:
            # 强制至少一个有效 token
            ids[0] = self.vocab["<UNK>"]
            mask[0] = 1
        # ⚠️ 关键：BCEWithLogitsLoss 需要 float 标签！
        label = float(self.labels[idx])

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),  # ✅ float!
        }


def build_vocab(csv_paths, language="chinese", min_freq=2, max_vocab=20000):
    counter = Counter()
    tokenize_fn = get_tokenizer(language)
    for path in csv_paths:
        assert os.path.exists(path), f"Vocab file not found: {path}"
        df = pd.read_csv(path)
        for text in tqdm(
            df["content"].fillna("").astype(str),
            desc=f"Building vocab from {os.path.basename(path)}",
        ):
            tokens = tokenize_fn(text)
            counter.update(tokens)

    words = [w for w, c in counter.most_common() if c >= min_freq]
    words = words[: max_vocab - 2]  # 只保留 <PAD>, <UNK>
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, w in enumerate(words, start=2):
        vocab[w] = i
    return vocab


# ----------------------------
# 2. 模型导入（确保 SKDCN 已定义）
# ----------------------------
# 假设 SKDCN 在当前目录或已导入
# from your_module import SKDCN


# ----------------------------
# 3. 评估函数（多指标）
# ----------------------------
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)  # [B]

            logits = model(input_ids, attention_mask)  # [B]
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, prec, rec, f1


# ----------------------------
# 4. 训练主函数
# ----------------------------
def train():
    set_seed(42)
    language = "all"  # ← 改为 "chinese" 或 "english"
    data_dir = "data"

    train_file = os.path.join(data_dir, f"{language}_train.csv")
    test_file = os.path.join(data_dir, f"{language}_test.csv")

    # 构建词汇表
    vocab = build_vocab([train_file], language=language, min_freq=2, max_vocab=20000)
    print(f"✅ Vocab size: {len(vocab)}")

    # 数据集
    train_dataset = TextDataset(train_file, vocab, max_len=256, language=language)
    test_dataset = TextDataset(test_file, vocab, max_len=256, language=language)

    # 注意：jieba 在多进程可能出错，建议 num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SKDCN(
        vocab_size=len(vocab),
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        num_keywords=50,
        max_len=256,
        dropout=0.3,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)

    best_f1 = 0.0
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)  # [B], float

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)  # [B]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print(f"logits:{logits}         labels:{labels} ")

        avg_train_loss = total_loss / len(train_loader)
        val_loss, acc, prec, rec, f1 = evaluate(model, test_loader, device)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(
            f"Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            save_path = f"skdcn_{language}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"🎉 New best F1: {f1:.4f}, model saved to {save_path}")

    print(f"\n✅ Training finished. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
