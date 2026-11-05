import re
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "bbc-text (1).csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ============================================================
# STEP 1: load the dataset and take a first look
# ============================================================
def print_raw_dataset(raw_df: pd.DataFrame) -> None:
    print("=== STEP 1: raw dataset ===")
    print(raw_df.shape)
    print(raw_df.columns.tolist())
    print(raw_df["category"].value_counts())
    print(raw_df["text"].iloc[0][:300])


raw_df = pd.read_csv(DATA_PATH)
print_raw_dataset(raw_df)


# ============================================================
# STEP 2: preprocessing - lowercase, remove punctuation,
#         remove stop words, tokenize
# ============================================================
STOP_WORDS = set(ENGLISH_STOP_WORDS)

# anything that is not a lowercase latin letter becomes a space
# (punctuation and digits carry no topic information here)
NON_LETTER_RE = re.compile(r"[^a-z]+")

# the dataset has apostrophes stripped ("don t", "us s"), which leaves
# single-letter fragments - tokens shorter than this are dropped
MIN_TOKEN_LENGTH = 2


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return NON_LETTER_RE.sub(" ", text)


def tokenize(text: str) -> list[str]:
    return text.split()


def remove_stop_words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in STOP_WORDS and len(token) >= MIN_TOKEN_LENGTH]


def preprocess(text: str) -> list[str]:
    text = to_lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    return remove_stop_words(tokens)


def print_preprocessing(raw_df: pd.DataFrame, articles_df: pd.DataFrame) -> None:
    print("\n=== STEP 2: preprocessing ===")
    print("before:", raw_df["text"].iloc[0][:200])
    print("after: ", articles_df["tokens"].iloc[0][:30])
    print("avg tokens per doc before:", round(raw_df["text"].str.split().str.len().mean(), 1))
    print("avg tokens per doc after: ", round(articles_df["tokens"].str.len().mean(), 1))


articles_df = raw_df.copy()
articles_df["tokens"] = articles_df["text"].astype(str).apply(preprocess)
print_preprocessing(raw_df, articles_df)


# ============================================================
# STEP 3: stratified train/test split
#         (vocabulary and IDF will be built from train only)
# ============================================================
def print_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n=== STEP 3: train/test split ===")
    print("train:", train_df.shape, " test:", test_df.shape)
    print(pd.DataFrame({
        "train %": train_df["category"].value_counts(normalize=True).round(3),
        "test %": test_df["category"].value_counts(normalize=True).round(3),
    }))


train_df, test_df = train_test_split(
    articles_df,
    test_size=TEST_SIZE,
    stratify=articles_df["category"],
    random_state=RANDOM_STATE,
)
print_split(train_df, test_df)


# ============================================================
# STEP 4: build the vocabulary from the TRAIN set only
#         every unique train word gets a fixed column index
# ============================================================
def build_vocabulary(docs: Iterable[list[str]]) -> dict[str, int]:
    unique_words = set()
    for tokens in docs:
        unique_words.update(tokens)

    # sorted so the word -> index mapping is the same on every run
    return {word: word_index for word_index, word in enumerate(sorted(unique_words))}


def print_vocabulary(index_to_word: np.ndarray) -> None:
    print("\n=== STEP 4: vocabulary ===")
    print("vocabulary size:", len(index_to_word))
    print("first 10 words:", list(index_to_word[:10]))
    print("last 10 words: ", list(index_to_word[-10:]))


vocab = build_vocabulary(train_df["tokens"])
index_to_word = np.array(sorted(vocab, key=vocab.get))
print_vocabulary(index_to_word)


# ============================================================
# STEP 5: Bag of Words
#         BoW[d][j] = how many times word j appears in document d
#         words that are not in the vocabulary (OOV) are ignored
# ============================================================
def document_to_bow(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocab), dtype=np.float32)
    for token in tokens:
        word_index = vocab.get(token)
        if word_index is not None:
            vector[word_index] += 1
    return vector


def build_bow_matrix(docs: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    # one row per document, one column per vocabulary word
    matrix = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for row, tokens in enumerate(docs):
        matrix[row] = document_to_bow(tokens, vocab)
    return matrix


def get_top_word_indices(vector: np.ndarray, top_n: int) -> np.ndarray:
    return np.argsort(-vector, kind="stable")[:top_n]


def print_bow(train_bow: np.ndarray, test_bow: np.ndarray, train_df: pd.DataFrame, test_df: pd.DataFrame, index_to_word: np.ndarray) -> None:
    print("\n=== STEP 5: Bag of Words ===")
    print("train BoW matrix:", train_bow.shape)
    print("test BoW matrix: ", test_bow.shape)

    # sanity check on one document: its most frequent words
    sample_vector = train_bow[0]
    top_indices = get_top_word_indices(sample_vector, 10)
    print("\nfirst train doc, category:", train_df["category"].iloc[0])
    print("unique words in doc:", np.count_nonzero(sample_vector), " total tokens:", int(sample_vector.sum()))
    print("most frequent words:", {index_to_word[i]: int(sample_vector[i]) for i in top_indices})

    # how sparse the vectors are and how many test tokens fell out of the vocabulary
    print("\navg non-zero cells per train doc:", round((train_bow > 0).sum(axis=1).mean(), 1), "of", len(index_to_word))
    test_total = test_df["tokens"].str.len().sum()
    test_in_vocab = test_bow.sum()
    print("test tokens out of vocabulary (OOV):", f"{1 - test_in_vocab / test_total:.2%}")


train_bow = build_bow_matrix(train_df["tokens"].tolist(), vocab)
test_bow = build_bow_matrix(test_df["tokens"].tolist(), vocab)
print_bow(train_bow, test_bow, train_df, test_df, index_to_word)


# ============================================================
# STEP 6: Term Frequency
#         TF[d][j] = BoW[d][j] / (total vocabulary tokens in document d)
#         so long articles don't get bigger values just for being long
# ============================================================
def compute_tf(bow_matrix: np.ndarray) -> np.ndarray:
    doc_lengths = bow_matrix.sum(axis=1, keepdims=True)
    # a document with no vocabulary words would divide by zero - keep its row all zeros
    doc_lengths[doc_lengths == 0] = 1
    return bow_matrix / doc_lengths


def print_tf(train_tf: np.ndarray) -> None:
    print("\n=== STEP 6: TF ===")
    print("train TF matrix:", train_tf.shape)
    print("every row sums to 1:", np.allclose(train_tf.sum(axis=1), 1))


train_tf = compute_tf(train_bow)
test_tf = compute_tf(test_bow)
print_tf(train_tf)


# ============================================================
# STEP 7: Inverse Document Frequency (from TRAIN only)
#         df[j]  = number of train documents that contain word j
#         IDF[j] = ln(N / df[j])
#         every vocabulary word is in at least one train document,
#         so df is never 0 and no smoothing is needed
# ============================================================
def compute_idf(bow_matrix: np.ndarray) -> np.ndarray:
    n_docs = bow_matrix.shape[0]
    doc_freq = (bow_matrix > 0).sum(axis=0)
    return np.log(n_docs / doc_freq).astype(np.float32)


def print_idf(idf: np.ndarray, index_to_word: np.ndarray) -> None:
    idf_order = np.argsort(idf)
    print("\n=== STEP 7: IDF ===")
    print("IDF vector:", idf.shape, " min:", round(idf.min(), 3), " max:", round(idf.max(), 3))
    print("lowest IDF (most common): ", {index_to_word[i]: round(float(idf[i]), 3) for i in idf_order[:10]})
    print("highest IDF (rarest):     ", {index_to_word[i]: round(float(idf[i]), 3) for i in idf_order[-5:]})


idf = compute_idf(train_bow)
print_idf(idf, index_to_word)


# ============================================================
# STEP 8: TF-IDF = TF * IDF
#         the same train IDF vector multiplies every row (train and test)
# ============================================================
def compute_tfidf(tf_matrix: np.ndarray, idf_vector: np.ndarray) -> np.ndarray:
    return tf_matrix * idf_vector


def print_tfidf(train_bow: np.ndarray, train_tf: np.ndarray, train_tfidf: np.ndarray, test_tfidf: np.ndarray, idf: np.ndarray, vocab: dict[str, int], index_to_word: np.ndarray) -> None:
    print("\n=== STEP 8: TF-IDF ===")
    print("train TF-IDF matrix:", train_tfidf.shape)
    print("test TF-IDF matrix: ", test_tfidf.shape)

    # same document as in STEP 5: compare its top words under BoW and under TF-IDF
    top_bow_indices = get_top_word_indices(train_bow[0], 10)
    top_tfidf_indices = get_top_word_indices(train_tfidf[0], 10)
    print("\nfirst train doc, top words by BoW:   ", [index_to_word[i] for i in top_bow_indices])
    print("first train doc, top words by TF-IDF:", [index_to_word[i] for i in top_tfidf_indices])

    said_index = vocab["said"]
    print(f"'said' -> BoW: {int(train_bow[0, said_index])}, TF: {train_tf[0, said_index]:.4f}, "
          f"IDF: {idf[said_index]:.4f}, TF-IDF: {train_tfidf[0, said_index]:.4f}")


train_tfidf = compute_tfidf(train_tf, idf)
test_tfidf = compute_tfidf(test_tf, idf)
print_tfidf(train_bow, train_tf, train_tfidf, test_tfidf, idf, vocab, index_to_word)


# ============================================================
# STEP 9: Analysis - top 10 words per category by average TF-IDF
#         average the TF-IDF rows of one category, then take the 10 largest columns
# ============================================================
TOP_N_WORDS = 10


def top_words_for_category(tfidf_matrix: np.ndarray, labels: np.ndarray, category: str, top_n: int = TOP_N_WORDS) -> list[tuple[str, float]]:
    category_rows = tfidf_matrix[labels == category]
    average_scores = category_rows.mean(axis=0)
    top_indices = np.argsort(-average_scores)[:top_n]
    return [(index_to_word[i], float(average_scores[i])) for i in top_indices]


def print_top_words(top_words_df: pd.DataFrame) -> None:
    print("\n=== STEP 9: top 10 words per category by average TF-IDF ===")
    print(top_words_df.to_string())


train_labels = train_df["category"].to_numpy()
categories = sorted(train_df["category"].unique())

top_words_df = pd.DataFrame({
    category: [f"{word} ({score:.4f})" for word, score in top_words_for_category(train_tfidf, train_labels, category)]
    for category in categories
})
top_words_df.index = range(1, TOP_N_WORDS + 1)
print_top_words(top_words_df)


# ============================================================
# STEP 10: Analysis - high TF / low IDF words and vice versa
#          a word's TF here = its average TF over the documents that contain it
#          ("when the word appears, how big a share of the document is it")
#          high / low = top / bottom 10% of all vocabulary words
# ============================================================
LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90
TOP_N_LISTED = 15


def average_tf_where_present(tf_matrix: np.ndarray, bow_matrix: np.ndarray) -> np.ndarray:
    doc_freq = (bow_matrix > 0).sum(axis=0)
    return tf_matrix.sum(axis=0) / doc_freq


def describe_words(word_indices: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "word": index_to_word[word_indices],
        "avg TF": word_tf[word_indices].round(4),
        "IDF": idf[word_indices].round(3),
        "docs": (train_bow[:, word_indices] > 0).sum(axis=0),
    }).reset_index(drop=True)


def print_word_groups(high_tf_low_idf: np.ndarray, low_tf_high_idf: np.ndarray, tf_low: float, tf_high: float, idf_low: float, idf_high: float) -> None:
    print("\n=== STEP 10: high TF / low IDF and vice versa ===")
    print(f"TF thresholds:  low <= {tf_low:.4f}, high >= {tf_high:.4f}")
    print(f"IDF thresholds: low <= {idf_low:.3f}, high >= {idf_high:.3f}")

    print(f"\nhigh TF + low IDF: {len(high_tf_low_idf)} words (top {TOP_N_LISTED})")
    print(describe_words(high_tf_low_idf[:TOP_N_LISTED]).to_string())

    print(f"\nlow TF + high IDF: {len(low_tf_high_idf)} words (top {TOP_N_LISTED})")
    print(describe_words(low_tf_high_idf[:TOP_N_LISTED]).to_string())


def plot_tf_vs_idf(idf: np.ndarray, word_tf: np.ndarray, high_tf_low_idf: np.ndarray, low_tf_high_idf: np.ndarray, index_to_word: np.ndarray) -> None:
    # every vocabulary word as one point, the two groups highlighted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(idf, word_tf, s=8, color="#c3c2b7", alpha=0.5, label="other words")
    ax.scatter(idf[high_tf_low_idf], word_tf[high_tf_low_idf], s=12, color="#2a78d6", label="high TF + low IDF")
    ax.scatter(idf[low_tf_high_idf], word_tf[low_tf_high_idf], s=12, color="#eb6834", label="low TF + high IDF")
    for i in high_tf_low_idf[:5]:
        ax.annotate(index_to_word[i], (idf[i], word_tf[i]), fontsize=8, xytext=(4, 2), textcoords="offset points")

    ax.set_yscale("log")
    ax.set_xlabel("IDF  (higher = rarer across documents)")
    ax.set_ylabel("average TF where present  (log scale)")
    ax.set_title("TF vs IDF for every vocabulary word (train set)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("tf_vs_idf.png", dpi=150)
    plt.close(fig)
    print("\nscatter plot saved to tf_vs_idf.png")


word_tf = average_tf_where_present(train_tf, train_bow)

tf_low, tf_high = np.percentile(word_tf, [LOW_PERCENTILE, HIGH_PERCENTILE])
idf_low, idf_high = np.percentile(idf, [LOW_PERCENTILE, HIGH_PERCENTILE])

high_tf_low_idf = np.where((word_tf >= tf_high) & (idf <= idf_low))[0]
low_tf_high_idf = np.where((word_tf <= tf_low) & (idf >= idf_high))[0]

# strongest examples first: lowest IDF (most widespread) in the first group, lowest TF in the second.
# most vocabulary words are rare, so the 10% IDF cut-off still lets in topic words like "yukos" -
# sorting by IDF puts the truly corpus-wide words on top
high_tf_low_idf = high_tf_low_idf[np.argsort(idf[high_tf_low_idf])]
low_tf_high_idf = low_tf_high_idf[np.argsort(word_tf[low_tf_high_idf])]

print_word_groups(high_tf_low_idf, low_tf_high_idf, tf_low, tf_high, idf_low, idf_high)
plot_tf_vs_idf(idf, word_tf, high_tf_low_idf, low_tf_high_idf, index_to_word)


# ============================================================
# STEP 11: Classification - the same Logistic Regression model
#          trained once on BoW vectors and once on TF-IDF vectors
# ============================================================
MAX_ITERATIONS = 1000


def train_classifier(train_matrix: np.ndarray, train_labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=MAX_ITERATIONS, random_state=RANDOM_STATE)
    model.fit(train_matrix, train_labels)
    return model


def print_evaluation(name: str, test_labels: np.ndarray, predictions: np.ndarray, categories: list[str]) -> None:
    print(f"\n--- {name} ---")
    print("accuracy:", f"{accuracy_score(test_labels, predictions):.4f}")
    print(classification_report(test_labels, predictions, labels=categories, digits=4))

    # rows = real category, columns = predicted category
    matrix = confusion_matrix(test_labels, predictions, labels=categories)
    print("confusion matrix (rows = real, columns = predicted):")
    print(pd.DataFrame(matrix, index=categories, columns=categories).to_string())


test_labels = test_df["category"].to_numpy()

bow_model = train_classifier(train_bow, train_labels)
bow_predictions = bow_model.predict(test_bow)

tfidf_model = train_classifier(train_tfidf, train_labels)
tfidf_predictions = tfidf_model.predict(test_tfidf)

print("\n=== STEP 11: classification with Logistic Regression ===")
print_evaluation("Bag of Words", test_labels, bow_predictions, categories)
print_evaluation("TF-IDF", test_labels, tfidf_predictions, categories)
