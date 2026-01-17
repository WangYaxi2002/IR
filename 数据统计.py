import pandas as pd

filelist = [
    "data/chinese_train.csv",
    "data/chinese_test.csv",
    "data/english_train.csv",
    "data/english_test.csv",
    "data/all_train.csv",
    "data/all_test.csv",
]

for file in filelist:
    df = pd.read_csv(file)
    print(f"File: {file}")
    print(df["label"].value_counts())
    print()
