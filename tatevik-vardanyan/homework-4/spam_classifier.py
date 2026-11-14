import math
import re
import random
from collections import defaultdict

import pandas as pd

EPS = 1.0   # ε (Laplace smoothing)

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def load_data(path="spam.csv"):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    if "label" not in df.columns:
        df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "message"})
    df = df[["label", "message"]].dropna()
    df["label"] = df["label"].str.strip().str.lower()
    return df

def train_test_split(df, test_size=0.2, seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut], df.iloc[cut:]

class MarkovClass:
    def __init__(self):
        self.init_c = defaultdict(int)
        self.trans_c = defaultdict(lambda: defaultdict(int))
        self.state_c = defaultdict(int)       # count(i) անցումներում
        self.n_msgs = 0

    def add(self, words):
        if not words:
            return
        self.n_msgs += 1
        self.init_c[words[0]] += 1
        for a, b in zip(words, words[1:]):
            self.trans_c[a][b] += 1
            self.state_c[a] += 1

    def log_init(self, w, M):
        # π_i = (count(s1=i)+ε) / (N + εM)
        return math.log((self.init_c.get(w, 0) + EPS) / (self.n_msgs + EPS * M))

    def log_trans(self, a, b, M):
        # A_ij = (count(i→j)+ε) / (count(i)+εM)
        c_ij = self.trans_c[a].get(b, 0) if a in self.trans_c else 0
        return math.log((c_ij + EPS) / (self.state_c.get(a, 0) + EPS * M))

    def log_likelihood(self, words, M):
        if not words:
            return 0.0
        ll = self.log_init(words[0], M)
        for a, b in zip(words, words[1:]):
            ll += self.log_trans(a, b, M)
        return ll

class SpamClassifier:
    def fit(self, df):
        self.models = {"spam": MarkovClass(), "ham": MarkovClass()}
        vocab = set()
        for label, msg in zip(df["label"], df["message"]):
            words = preprocess(msg)
            vocab.update(words)
            if label in self.models:
                self.models[label].add(words)
        self.M = len(vocab)
        total = self.models["spam"].n_msgs + self.models["ham"].n_msgs
        self.log_prior = {k: math.log(m.n_msgs / total) for k, m in self.models.items()}

    def predict(self, msg):
        words = preprocess(msg)
        scores = {k: self.models[k].log_likelihood(words, self.M) + self.log_prior[k]
                  for k in self.models}
        return "spam" if scores["spam"] > scores["ham"] else "ham"

if __name__ == "__main__":
    df = load_data("spam.csv")
    train, test = train_test_split(df)

    clf = SpamClassifier()
    clf.fit(train)

    preds = [clf.predict(m) for m in test["message"]]
    correct = sum(p == t for p, t in zip(preds, test["label"]))
    print(f"Train: {len(train)}, Test: {len(test)}, Vocabulary: {clf.M}")
    print(f"Accuracy: {correct / len(test):.4f}")
