import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import os
from datetime import datetime
import argparse
import re
from collections import Counter
import string
import jieba  # 仅中文分词需要，如果只做英文可以移除


# 设置随机种子以确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# 简单的文本预处理函数
def preprocess_text(text, language="chinese"):
    """基础文本预处理"""
    if not isinstance(text, str):
        text = str(text)

    # 转小写（英文需要，中文不需要）
    if language == "english":
        text = text.lower()

    # 移除特殊字符，保留基本标点和字母数字
    text = re.sub(r"[^\w\s\u4e00-\u9fff.,!?;:]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 自定义分词器
try:
    import spacy
    from spacy.lang.en import English

    # _EN_NLP = spacy.load(
    #     "en_core_web_sm", disable=["parser", "ner"]
    # )  # 只保留 tokenizer
    # _EN_NLP.max_length = 10000000
    _EN_NLP = None
except OSError:
    # 如果没安装英文模型，回退到简单分词
    _EN_NLP = None
    import warnings

    warnings.warn("spaCy 英文模型 'en_core_web_sm' 未安装，英文分词将使用基础规则。")


def get_tokenizer(language="chinese"):
    """
    返回适合指定语言的分词函数。

    Args:
        language (str): "chinese", "english", 或 "mixed"
    """
    if language == "chinese":

        def tokenize(text):
            return jieba.lcut(text)

    elif language == "english":

        def tokenize(text):
            if _EN_NLP is not None:
                doc = _EN_NLP(text)
                return [token.text for token in doc]
            else:
                # 回退方案
                text = re.sub(r"([.,!?;:\(\)\"'])", r" \1 ", text)
                return text.split()

    elif language == "all":
        # 预编译正则：匹配连续中文、连续非中文（含英文/数字/符号）
        chinese_seg = re.compile(r"[\u4e00-\u9fff]+")  # 中文字符范围
        non_chinese_seg = re.compile(r"[^\u4e00-\u9fff]+")

        def tokenize(text):
            tokens = []
            last_end = 0
            # 找出所有中文片段
            for match in chinese_seg.finditer(text):
                start, end = match.span()
                # 处理前面的非中文部分
                if start > last_end:
                    non_chinese_part = text[last_end:start]
                    if _EN_NLP is not None:
                        en_doc = _EN_NLP(non_chinese_part)
                        tokens.extend([t.text for t in en_doc])
                    else:
                        # 简单英文分词回退
                        cleaned = re.sub(
                            r"([.,!?;:\(\)\"'])", r" \1 ", non_chinese_part
                        )
                        tokens.extend(cleaned.split())
                # 处理中文部分
                chinese_part = match.group()
                tokens.extend(jieba.lcut(chinese_part))
                last_end = end

            # 处理末尾剩余的非中文部分
            if last_end < len(text):
                non_chinese_part = text[last_end:]
                if _EN_NLP is not None:
                    en_doc = _EN_NLP(non_chinese_part)
                    tokens.extend([t.text for t in en_doc])
                else:
                    cleaned = re.sub(r"([.,!?;:\(\)\"'])", r" \1 ", non_chinese_part)
                    tokens.extend(cleaned.split())

            # 过滤空字符串
            return [t for t in tokens if t.strip()]

    else:
        raise ValueError("Unsupported language. Use 'chinese', 'english', or 'mixed'.")

    return tokenize


def build_vocabulary(texts, tokenizer, min_freq=1, max_vocab_size=50000):
    """构建词汇表（按频率排序，保留最高频词）"""
    word_counts = Counter()

    for text in texts:
        tokens = tokenizer(text)
        word_counts.update(tokens)

    # 按频率降序，过滤低频词和空字符串
    vocab_words = [
        word
        for word, count in word_counts.most_common()
        if count >= min_freq and word.strip()
    ]

    # 限制词汇表大小
    if max_vocab_size is not None:
        vocab_words = vocab_words[:max_vocab_size]

    # 构建词汇表：特殊标记 + 高频词
    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    vocab = {token: idx for idx, token in enumerate(special_tokens)}

    for word in vocab_words:
        if word not in vocab:  # 避免覆盖特殊标记
            vocab[word] = len(vocab)

    return vocab


# 自定义数据集类
class SpamDataset(Dataset):
    def __init__(
        self, texts, labels, vocab, tokenizer, max_len=100, language="chinese"
    ):
        self.texts = [preprocess_text(text, language) for text in texts]
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.language = language

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])

        # 分词和数值化
        tokens = self.tokenizer(text)
        tokens = tokens[: self.max_len - 2]  # 为<bos>和<eos>留出空间

        # 添加特殊标记
        tokens = ["<bos>"] + tokens + ["<eos>"]

        # 将词转换为ID，未知词使用<unk>的ID
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

        # 填充
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab["<pad>"]] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[: self.max_len]

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


# 模型1: LSTM
class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        output_dim,
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(input_ids))
        # embedded shape: [batch_size, seq_len, embed_dim]

        # 计算序列的实际长度（忽略padding）
        lengths = (input_ids != 1).sum(dim=1).cpu()  # 假设1是<pad>的索引

        # 使用pack_padded_sequence处理变长序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # 处理双向LSTM
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.dropout(hidden)
        return self.fc(hidden)


# 模型2: CNN
class CNNModel(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim, dropout=0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_dim)
                )
                for fs in filter_sizes
            ]
        )
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids shape: [batch_size, seq_len]
        embedded = self.embedding(input_ids)
        # embedded shape: [batch_size, seq_len, embed_dim]

        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]

        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved shape: [batch_size, n_filters, seq_len - fs + 1]

        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        # pooled shape: [batch_size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat shape: [batch_size, n_filters * len(filter_sizes)]

        return self.fc(cat)


# 模型3: Transformer
class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        output_dim,
        max_len=100,
        dropout=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        self.fc = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # 位置编码
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))

    def forward(self, input_ids):
        # input_ids shape: [batch_size, seq_len]
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        positions = self.position_ids[:, :seq_len]
        embedded = self.embedding(input_ids) + self.pos_embedding(positions)
        embedded = self.dropout(embedded)

        # 创建注意力掩码 (忽略填充部分)
        src_key_padding_mask = input_ids == 1  # 假设1是<pad>的索引

        transformer_out = self.transformer_encoder(
            embedded, src_key_padding_mask=src_key_padding_mask
        )

        # 使用序列的第一个token ([CLS] 类似) 进行分类
        cls_output = transformer_out[:, 0, :]
        return self.fc(cls_output)


# 训练函数
def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(iterator, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        predictions = model(input_ids)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        all_preds.extend(torch.argmax(predictions, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    return epoch_loss / len(iterator), accuracy, precision, recall, f1


# 评估函数
def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            predictions = model(input_ids)
            loss = criterion(predictions, labels)

            epoch_loss += loss.item()
            all_preds.extend(torch.argmax(predictions, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return (
        epoch_loss / len(iterator),
        accuracy,
        precision,
        recall,
        f1,
        conf_matrix,
        all_preds,
        all_labels,
    )


# 训练循环
def run_training(config, train_loader, valid_loader, model, device):
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=0.9,
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    best_valid_f1 = 0
    train_losses = []
    valid_losses = []
    train_f1s = []
    valid_f1s = []

    print("Starting training...")
    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        train_loss, train_acc, train_prec, train_rec, train_f1 = train(
            model, train_loader, optimizer, criterion, device
        )
        valid_loss, valid_acc, valid_prec, valid_rec, valid_f1, _, _, _ = evaluate(
            model, valid_loader, criterion, device
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_f1s.append(train_f1)
        valid_f1s.append(valid_f1)

        # 更新学习率
        scheduler.step(valid_f1)

        # 保存最佳模型
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save(model.state_dict(), config["best_model_path"])

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.3f}"
        )
        print(
            f"\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}% | Valid F1: {valid_f1:.3f}"
        )

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label="Train F1")
    plt.plot(valid_f1s, label="Valid F1")
    plt.title("F1 Score Curves")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.savefig(config["loss_curve_path"])
    plt.close()

    return best_valid_f1


# 测试函数
def test_model(config, test_loader, model, device):
    # 加载最佳模型
    model.load_state_dict(torch.load(config["best_model_path"], map_location=device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_prec, test_rec, test_f1, conf_matrix, preds, labels = (
        evaluate(model, test_loader, criterion, device)
    )

    print(f"Test Loss: {test_loss:.3f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Precision: {test_prec:.3f}")
    print(f"Test Recall: {test_rec:.3f}")
    print(f"Test F1 Score: {test_f1:.3f}")

    # 混淆矩阵
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # 详细分类报告
    print("\nClassification Report:")
    print(
        classification_report(
            labels, preds, target_names=["Not Spam", "Spam"], digits=8
        )
    )

    # 保存结果
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "confusion_matrix": conf_matrix.tolist(),
        "config": config,
    }

    with open(config["results_path"], "w") as f:
        json.dump(results, f, indent=4)

    return results


# 主函数
def main(args):
    # 设置设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Using device: {device}")

    # 设置随机种子
    set_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据
    print(f"Loading {args.mode} data...")
    train_df = pd.read_csv(f"data/{args.mode}_train.csv")
    test_df = pd.read_csv(f"data/{args.mode}_test.csv")
    print("Train samples:", len(train_df))
    print("Test samples:", len(test_df))

    # 验证数据列是否存在
    if "content" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("CSV files must contain 'content' and 'label' columns")

    # 确定语言
    language = args.mode
    print(f"Using {language} language processing")

    # 获取分词器
    tokenizer = get_tokenizer(language)

    # 构建词汇表
    print("Building vocabulary...")
    vocab = build_vocabulary(
        train_df["content"].values,
        tokenizer,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
    )
    print(f"Vocabulary size: {len(vocab)}")

    # 创建数据集
    train_dataset = SpamDataset(
        train_df["content"].values,
        train_df["label"].values,
        vocab,
        tokenizer,
        max_len=args.max_len,
        language=language,
    )

    test_dataset = SpamDataset(
        test_df["content"].values,
        test_df["label"].values,
        vocab,
        tokenizer,
        max_len=args.max_len,
        language=language,
    )

    # 划分验证集
    train_size = int(0.9 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # 配置
    config = {
        "model_name": args.model,
        "vocab_size": len(vocab),
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": 2,  # 二分类
        "n_layers": args.n_layers,
        "bidirectional": args.bidirectional,
        "dropout": args.dropout,
        "n_filters": args.n_filters,
        "filter_sizes": [int(fs) for fs in args.filter_sizes.split(",")],
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "max_len": args.max_len,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_model_path": os.path.join(
            args.save_dir, f"{args.model}_{args.mode}_best_model.pt"
        ),
        "loss_curve_path": os.path.join(
            args.save_dir, f"{args.model}_{args.mode}_loss_curve.png"
        ),
        "results_path": os.path.join(
            args.save_dir, f"{args.model}_{args.mode}_results.json"
        ),
    }

    # 初始化模型
    if args.model == "lstm":
        model = LSTMModel(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            n_layers=config["n_layers"],
            bidirectional=config["bidirectional"],
            dropout=config["dropout"],
        )
    elif args.model == "cnn":
        model = CNNModel(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            n_filters=config["n_filters"],
            filter_sizes=config["filter_sizes"],
            output_dim=config["output_dim"],
            dropout=config["dropout"],
        )
    elif args.model == "transformer":
        model = TransformerModel(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            nhead=config["nhead"],
            num_encoder_layers=config["num_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            output_dim=config["output_dim"],
            max_len=config["max_len"],
            dropout=config["dropout"],
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model = model.to(device)
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # 训练
    best_f1 = run_training(config, train_loader, valid_loader, model, device)
    print(f"Best validation F1 score: {best_f1:.4f}")

    # 测试
    test_results = test_model(config, test_loader, model, device)

    print("Training and evaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spam Detection with Deep Learning")

    # 数据参数
    parser.add_argument(
        "--mode",
        type=str,
        default="chinese",
        choices=["chinese", "english", "all"],
        help="Dataset mode: chinese, english, or all",
    )

    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["lstm", "cnn", "transformer"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=100, help="Embedding dimension"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension for LSTM"
    )
    parser.add_argument("--n_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=True,
        help="Use bidirectional LSTM",
    )
    parser.add_argument(
        "--n_filters", type=int, default=100, help="Number of filters for CNN"
    )
    parser.add_argument(
        "--filter_sizes", type=str, default="2,3,4", help="Filter sizes for CNN"
    )
    parser.add_argument(
        "--nhead", type=int, default=4, help="Number of heads in Transformer"
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=2,
        help="Number of encoder layers in Transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=512,
        help="Feedforward dimension in Transformer",
    )

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer",
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--max_len", type=int, default=100, help="Maximum sequence length"
    )
    parser.add_argument(
        "--min_freq", type=int, default=1, help="Minimum frequency for vocabulary"
    )
    parser.add_argument(
        "--max_vocab_size", type=int, default=20000, help="Maximum vocabulary size"
    )

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="Disable CUDA"
    )
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Directory to save results"
    )

    args = parser.parse_args()

    # 检查是否需要安装jieba（中文分词）
    if args.mode == "chinese":
        try:
            import jieba
        except ImportError:
            print("Installing jieba for Chinese text processing...")
            import sys
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba"])
            import jieba

    main(args)
