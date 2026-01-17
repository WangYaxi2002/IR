from typing import Literal
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(
    name: Literal[
        "trec05p-1", "trec06c", "trec06p", "trec07p", "all", "chinese", "english"
    ],
) -> pd.DataFrame:
    index_path = []
    if name == "trec05p-1" or name == "all" or name == "english":
        index_path.append("data/trec05p-1")
    if name == "trec06c" or name == "all" or name == "chinese":
        index_path.append("data/trec06c")
    if name == "trec06p" or name == "all" or name == "english":
        index_path.append("data/trec06p")
    if name == "trec07p" or name == "all" or name == "english":
        index_path.append("data/trec07p")
    datas = []
    for index in index_path:
        with open(f"../{index}/full/index", "r") as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        for line in lines:
            label = 1 if line[0].lower() == "spam" else 0
            path = f"../{index}{line[1].replace('..', '')}"
            datas.append({"path": path, "label": label})
    df = pd.DataFrame(datas)
    return df


def split_data(
    name: Literal[
        "trec05p-1", "trec06c", "trec06p", "trec07p", "all", "chinese", "english"
    ],
    test_size=0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_data(name)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )
    return train_df, test_df


def get_content(file):
    with open(file, "rb") as mail:
        raw_data = mail.read()
        encoding = chardet.detect(raw_data)["encoding"]
        content = raw_data.decode(encoding or "gbk", errors="ignore")
    return content
