from datasets import split_data
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os


def get_content(file):
    try:
        with open(file, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            print(f"Error reading file1: {file}")
            return ""
    except Exception:
        print(f"Error reading file: {file}")
        return ""


def parallel_get_content(paths):
    max_workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(get_content, paths), total=len(paths), mininterval=0.5)
        )
    return results


def replace_content_parallel(df, save_path, batch_size=5000):
    print(f"Processing {len(df)} files...")
    all_contents = []
    paths = df["path"].tolist()

    for i in tqdm(range(0, len(paths), batch_size), desc="Batches"):
        batch = paths[i : i + batch_size]
        all_contents.extend(parallel_get_content(batch))

    df = df.copy()
    df["content"] = all_contents
    df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":

    # ===== 中文数据 =====
    print("Processing Chinese data...")
    train_data_c, test_data_c = split_data("chinese", test_size=0.2)
    c1 = replace_content_parallel(train_data_c, "data/chinese_train.csv")
    c2 = replace_content_parallel(test_data_c, "data/chinese_test.csv")

    # ===== 英文数据 =====
    print("Processing English data...")
    train_data1, test_data1 = split_data("trec05p-1", test_size=0.2)
    train_data2, test_data2 = split_data("trec06p", test_size=0.2)
    train_data3, test_data3 = split_data("trec07p", test_size=0.2)

    train_data_e = pd.concat([train_data1, train_data2, train_data3], ignore_index=True)
    test_data_e = pd.concat([test_data1, test_data2, test_data3], ignore_index=True)

    e1 = replace_content_parallel(train_data_e, "data/english_train.csv")
    e2 = replace_content_parallel(test_data_e, "data/english_test.csv")

    # ===== 合并 all =====
    print("Merging all data...")
    pd.concat([c1, e1], ignore_index=True).to_csv("data/all_train.csv", index=False)
    pd.concat([c2, e2], ignore_index=True).to_csv("data/all_test.csv", index=False)

    print("Done!")
