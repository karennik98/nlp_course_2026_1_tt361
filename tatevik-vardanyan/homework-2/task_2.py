import re
import numpy as np
import pandas as pd
from collections import Counter

# Data
df = pd.read_csv("bbc-text.csv")
df = df.dropna(subset=["text", "category"]).reset_index(drop=True)

rng = np.random.RandomState(42)
idx = rng.permutation(len(df))
split = int(0.8 * len(df))
train_df = df.iloc[idx[:split]].reset_index(drop=True)
test_df  = df.iloc[idx[split:]].reset_index(drop=True)

# Preprocessing
try:
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP = set(ENGLISH_STOP_WORDS)

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()                     # tokenization
    return [w for w in tokens if w not in STOP and len(w) > 1]

train_tokens = [preprocess(t) for t in train_df["text"]]
test_tokens  = [preprocess(t) for t in test_df["text"]]

# Bag of Words
def build_vocabulary(docs_tokens, min_freq=2):
    counter = Counter(w for doc in docs_tokens for w in doc)
    words = sorted(w for w, c in counter.items() if c >= min_freq)
    return {w: i for i, w in enumerate(words)}

def doc_to_bow(tokens, vocab):
    vec = np.zeros(len(vocab), dtype=np.float32)
    for w in tokens:
        j = vocab.get(w)
        if j is not None:
            vec[j] += 1
    return vec

def build_bow_matrix(docs_tokens, vocab):
    return np.vstack([doc_to_bow(d, vocab) for d in docs_tokens])

vocab = build_vocabulary(train_tokens)
X_train_bow = build_bow_matrix(train_tokens, vocab)
X_test_bow  = build_bow_matrix(test_tokens, vocab)   # transform
print("V =", len(vocab), "| train:", X_train_bow.shape, "| test:", X_test_bow.shape)

# TF-IDF
def compute_tf(bow_matrix, normalize=False):
    tf = bow_matrix.copy()
    if normalize:
        tf = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1)
    return tf

def compute_idf(bow_train):
    N = bow_train.shape[0]
    Nt = (bow_train > 0).sum(axis=0)
    return np.log(N / Nt)

def compute_tfidf(bow_matrix, idf, normalize_tf=False):
    return compute_tf(bow_matrix, normalize_tf) * idf   # broadcasting

idf = compute_idf(X_train_bow)
X_train_tfidf = compute_tfidf(X_train_bow, idf)
X_test_tfidf  = compute_tfidf(X_test_bow, idf)

def l2_normalize(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-12)

# Analysis
inv_vocab = {i: w for w, i in vocab.items()}
categories = sorted(train_df["category"].unique())

def top_words_by_category(category, k=10):
    mask = (train_df["category"] == category).values
    mean_scores = X_train_tfidf[mask].mean(axis=0)
    top = np.argsort(mean_scores)[::-1][:k]
    return [(inv_vocab[i], round(float(mean_scores[i]), 4)) for i in top]

for c in categories:
    print(f"\n[{c}] top 10 ըստ միջին TF-IDF-ի")
    for w, s in top_words_by_category(c):
        print(f"  {w:15s} {s}")

# Բարձր TF + ցածր IDF  /  ցածր TF + բարձր IDF
mean_tf  = X_train_bow.mean(axis=0)
tf_hi    = np.percentile(mean_tf, 95)
tf_lo    = np.percentile(mean_tf, 50)
idf_hi   = np.percentile(idf, 75)
idf_lo   = np.percentile(idf, 25)

high_tf_low_idf = [inv_vocab[i] for i in np.where((mean_tf >= tf_hi) & (idf <= idf_lo))[0]]
low_tf_high_idf = [inv_vocab[i] for i in np.where((mean_tf <= tf_lo) & (idf >= idf_hi))[0]]

print("\nԲարձր TF, ցածր IDF (հաճախ, բայց շատ փաստաթղթերում):", high_tf_low_idf[:20])
print("Ցածր TF, բարձր IDF (հազվադեպ, բայց յուրահատուկ):", low_tf_high_idf[:20])

# Classification (cosine similarity, centroid)
def predict(X, centroids, cats):
    Xn = l2_normalize(X)
    Cn = l2_normalize(centroids)
    return [cats[i] for i in (Xn @ Cn.T).argmax(axis=1)]

def evaluate(Xtr, Xte, name):
    cents = np.vstack([Xtr[(train_df["category"] == c).values].mean(axis=0) for c in categories])
    pred = predict(Xte, cents, categories)
    acc = np.mean(np.array(pred) == test_df["category"].values)
    print(f"{name}: accuracy = {acc:.3f}")

evaluate(X_train_bow,   X_test_bow,   "BoW")
evaluate(X_train_tfidf, X_test_tfidf, "TF-IDF")
