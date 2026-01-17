import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# 支持切换中文或英文或全部
mode = "english"
print(f"Loading {mode} data...")
train_df = pd.read_csv(f"data/{mode}_train.csv")
test_df = pd.read_csv(f"data/{mode}_test.csv")
print("Train samples:", len(train_df))
print("Test samples:", len(test_df))

X_train = train_df["content"].fillna("").tolist()
y_train = train_df["label"].tolist()

X_test = test_df["content"].fillna("").tolist()
y_test = test_df["label"].tolist()


def eval_model(model, name):
    print(f"\n{'='*50}")
    print(f"Evaluating: {name}")
    print("=" * 50)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=8))

    correct = (y_pred == y_test).sum()
    incorrect = len(y_test) - correct
    print(f"正确: {correct}, 错误: {incorrect}")


# TF-IDF 向量化（统一处理）
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 模型 1: Naive Bayes
eval_model(MultinomialNB(alpha=0.1), "Multinomial Naive Bayes")

# 模型 2: Decision Tree (entropy)
eval_model(
    DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42),
    "Decision Tree (entropy, max_depth=10)",
)

# 模型 3: Decision Tree (gini, deeper)
eval_model(
    DecisionTreeClassifier(
        criterion="gini",
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    ),
    "Decision Tree (gini, tuned)",
)

# 模型 4: Linear SVM
eval_model(LinearSVC(C=1.0, random_state=42, max_iter=2000), "LinearSVC")

print("\n✅ All models evaluated.")
