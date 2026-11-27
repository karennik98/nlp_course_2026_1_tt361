"""
Part 1 - LDA training script.

Pipeline:
    1. Load the first 1000 training posts of 20 Newsgroups (headers/footers/quotes removed)
    2. Preprocess: tokenize, lowercase, remove stopwords, keep words of 3+ characters
    3. Build a dictionary and filter rare (< 5 docs) and common (> 50% of docs) words
    4. Convert documents to Bag-of-Words
    5. Train an LDA model (10 topics, 15 passes, alpha/eta learned automatically)
    6. Save the model and dictionary to models/ and print the discovered topics

The `preprocess` function is imported by the inference script (Part 3),
so new texts are cleaned exactly like the training documents.
"""

import os
import time

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

# Data
NUM_DOCUMENTS = 1000

# Preprocessing
MIN_WORD_LEN = 3
NO_BELOW = 5        # drop words appearing in fewer than 5 documents
NO_ABOVE = 0.5      # drop words appearing in more than 50% of documents

# Model
NUM_TOPICS = 10
PASSES = 15
RANDOM_STATE = 42   # fixed seed -> the same topics on every run
TOP_WORDS = 15

# Paths (relative to this file, so the script works from any working directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "lda_model")
DICTIONARY_PATH = os.path.join(MODELS_DIR, "dictionary.dict")


def print_header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def preprocess(text):
    """
    Turn raw text into a list of clean words.
    simple_preprocess: tokenizes, lowercases, drops punctuation/numbers,
    keeps words of MIN_WORD_LEN..15 characters. Then gensim stopwords are removed.
    """
    tokens = simple_preprocess(text, min_len=MIN_WORD_LEN)
    return [token for token in tokens if token not in STOPWORDS]


def load_data(num_documents):
    """Load the first `num_documents` training posts into a DataFrame."""
    dataset = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))

    df = pd.DataFrame({
        "text": dataset.data[:num_documents],
        "category": [dataset.target_names[t] for t in dataset.target[:num_documents]],
    })
    df["num_words"] = df["text"].str.split().str.len()

    print_header("STEP 1: DATA")
    print(f"Documents loaded: {len(df)}")
    print(f"Empty documents (nothing left after removing headers/footers/quotes): "
          f"{(df['num_words'] == 0).sum()}")
    print("\nWords per document:")
    print(df["num_words"].describe().round(1).to_string())
    print(f"\nDocuments per category ({df['category'].nunique()} categories):")
    print(df["category"].value_counts().to_string())

    return df


def preprocess_documents(df):
    """Add a `tokens` column and drop documents that become empty."""
    df["tokens"] = df["text"].apply(preprocess)
    df["num_tokens"] = df["tokens"].str.len()

    print_header("STEP 2: PREPROCESSING")
    example = df[df["num_words"] > 30].iloc[0]
    print(f"Example (category: {example['category']})")
    print(f"  Before: {' '.join(example['text'].split())[:300]}...")
    print(f"  After:  {example['tokens'][:30]}...")

    empty = df["num_tokens"] == 0
    df = df[~empty].reset_index(drop=True)
    print(f"\nDropped {empty.sum()} empty documents -> {len(df)} remain")
    print(f"Average words per document: before {df['num_words'].mean():.1f}, "
          f"after {df['num_tokens'].mean():.1f}")

    return df


def build_dictionary(tokenized_docs):
    """Create the word <-> id dictionary and filter rare and common words."""
    dictionary = Dictionary(tokenized_docs)

    print_header("STEP 3: DICTIONARY")
    print(f"Vocabulary before filtering: {len(dictionary)} unique words")

    # keep_n=None: apply only the task's two rules, no extra vocabulary size limit
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=None)

    print(f"Vocabulary after filtering:  {len(dictionary)} unique words "
          f"(in >= {NO_BELOW} docs and <= {NO_ABOVE:.0%} of docs)")

    return dictionary


def build_corpus(df, dictionary):
    """Convert each document to Bag-of-Words; drop documents with an empty BoW."""
    df["bow"] = [dictionary.doc2bow(tokens) for tokens in df["tokens"]]

    print_header("STEP 4: BAG-OF-WORDS")
    example = df.iloc[0]
    print(f"Example document (category: {example['category']})")
    print(f"  Tokens:          {example['tokens'][:15]}...")
    print(f"  BoW (id, count): {example['bow'][:10]}...")
    print(f"  BoW as words:    {[(dictionary[i], c) for i, c in example['bow'][:10]]}...")

    empty = df["bow"].str.len() == 0
    df = df[~empty].reset_index(drop=True)
    print(f"\nDropped {empty.sum()} documents with no known words -> {len(df)} remain")

    avg_unique = df["bow"].str.len().mean()
    print(f"Average unique words per document: {avg_unique:.1f} of {len(dictionary)} "
          f"-> matrix is {1 - avg_unique / len(dictionary):.1%} zeros (sparse)")

    return df, df["bow"].tolist()


def train_model(corpus, dictionary):
    """Train the LDA model with the task's settings."""
    print_header("STEP 5: TRAINING")
    print(f"Training LDA: {NUM_TOPICS} topics, {PASSES} passes, alpha='auto', eta='auto' ...")

    start = time.time()
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=PASSES,
        alpha="auto",
        eta="auto",
        random_state=RANDOM_STATE,
    )
    print(f"Done in {time.time() - start:.1f} seconds")

    eta = lda_model.eta
    print(f"\nLearned eta (one value per word): min {eta.min():.4f}, "
          f"mean {eta.mean():.4f}, max {eta.max():.4f}  (started at {1 / NUM_TOPICS})")

    return lda_model


def save_model(lda_model, dictionary):
    """Save the trained model and the dictionary to MODELS_DIR."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    lda_model.save(MODEL_PATH)
    dictionary.save(DICTIONARY_PATH)

    print_header("STEP 6: SAVED")
    print(f"Model:      {MODEL_PATH}")
    print(f"Dictionary: {DICTIONARY_PATH}")


def print_topics(lda_model):
    """Print each topic's top words and its learned alpha."""
    print_header(f"DISCOVERED TOPICS (top {TOP_WORDS} words)")
    print(f"alpha started at {1 / NUM_TOPICS} for every topic; "
          f"larger learned alpha = topic is common across documents\n")

    topics = lda_model.show_topics(num_topics=NUM_TOPICS, num_words=TOP_WORDS, formatted=False)
    for topic_id, words in topics:
        print(f"Topic {topic_id} (alpha={lda_model.alpha[topic_id]:.3f}):")
        print("   " + ", ".join(f"{word} ({prob:.3f})" for word, prob in words))


def main():
    df = load_data(NUM_DOCUMENTS)
    df = preprocess_documents(df)
    dictionary = build_dictionary(df["tokens"])
    df, corpus = build_corpus(df, dictionary)
    lda_model = train_model(corpus, dictionary)
    save_model(lda_model, dictionary)
    print_topics(lda_model)


if __name__ == "__main__":
    main()
