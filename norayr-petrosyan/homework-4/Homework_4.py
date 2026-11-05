import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def tokenize(text):
    words = re.findall(r"[a-z]+", text.lower())
    return [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 1]


def build_vocabulary(documents):
    words = sorted({word for doc in documents for word in doc})
    return {word: i for i, word in enumerate(words)}


def bow_vector(doc, vocabulary):
    return Counter(vocabulary[word] for word in doc if word in vocabulary)


def term_frequency(doc, vocabulary):
    return bow_vector(doc, vocabulary)


def inverse_document_frequency(documents, vocabulary):
    frequency = Counter()
    for doc in documents:
        frequency.update(set(doc))
    return [math.log(len(documents) / frequency[word]) for word in vocabulary]


def tfidf_vector(doc, vocabulary, idf):
    counts = term_frequency(doc, vocabulary)
    return {i: count * idf[i] for i, count in counts.items()}


def matrix(vectors, width):
    values, columns, starts = [], [], [0]
    for vector in vectors:
        columns.extend(vector.keys())
        values.extend(vector.values())
        starts.append(len(values))
    return csr_matrix((values, columns, starts), shape=(len(vectors), width))


def main(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    articles = {row["text"].strip(): row["category"] for row in rows}
    docs = [tokenize(text) for text in articles]
    train, test, y_train, y_test = train_test_split(
        docs, list(articles.values()), test_size=0.2, random_state=42,
        stratify=list(articles.values())
    )
    vocab = build_vocabulary(train)
    idf = inverse_document_frequency(train, vocab)
    print(f"{len(rows)} rows, {len(articles)} distinct articles")
    print(f"Training: {len(train)}, test: {len(test)}, vocabulary: {len(vocab)} words")

    bow_train = matrix([bow_vector(doc, vocab) for doc in train], len(vocab))
    bow_test = matrix([bow_vector(doc, vocab) for doc in test], len(vocab))
    tfidf_train = matrix([tfidf_vector(doc, vocab, idf) for doc in train], len(vocab))
    tfidf_test = matrix([tfidf_vector(doc, vocab, idf) for doc in test], len(vocab))

    for name, x_train, x_test in [("BoW", bow_train, bow_test),
                                  ("TF-IDF", tfidf_train, tfidf_test)]:
        model = MultinomialNB()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        print(f"\n{name} accuracy: {accuracy_score(y_test, predictions):.2%}")
        print(classification_report(y_test, predictions, digits=3))

    words = list(vocab)
    sport_rows = np.array(y_train) == "sport"
    scores = np.asarray(tfidf_train[sport_rows].mean(axis=0)).ravel()
    print("Top 10 words by average TF-IDF in sport training articles")
    for i in np.argsort(-scores)[:10]:
        print(f"{words[i]:16} {scores[i]:.3f}")

    doc = next(doc for doc, label in zip(train, y_train) if label == "sport")
    counts = term_frequency(doc, vocab)
    print("\nExample sport article begins:", " ".join(doc[:12]))
    common = [i for i in counts if counts[i] >= 2]
    rare = [i for i in counts if counts[i] == 1]
    for title, indices in [("Repeated words with the lowest IDF", sorted(common, key=idf.__getitem__)),
                           ("Single-use words with the highest IDF", sorted(rare, key=idf.__getitem__, reverse=True))]:
        print(title)
        for i in indices[:5]:
            print(f"{words[i]:16} TF={counts[i]}  IDF={idf[i]:.3f}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else next(Path(__file__).parent.glob("bbc-text*.csv"))
    main(path)
