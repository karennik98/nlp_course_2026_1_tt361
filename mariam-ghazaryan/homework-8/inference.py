"""Part 3: classify new documents into the trained LDA topics."""

import json
import os

from gensim import corpora
from gensim.models import LdaModel

from preprocessing import preprocess

MODEL_DIR = "models"
LABELS_FILE = os.path.join(MODEL_DIR, "topic_labels.json")

SAMPLE_DOCUMENTS = [
    "The new graphics card delivers amazing performance for gaming. The GPU can handle 4K "
    "resolution easily with ray tracing enabled. Gamers will love the improved frame rates.",
    "Scientists discovered a new exoplanet orbiting a distant star in the habitable zone. "
    "The research team published their findings in Nature journal. This discovery could "
    "provide insights into planetary formation.",
    "The basketball team won the championship after an incredible final game. The players "
    "celebrated with fans in the stadium. It was the team's first title in twenty years.",
    "Congress passed a new bill regarding healthcare reform. The president is expected to "
    "sign the legislation next week. The policy will affect millions of citizens across the country.",
    "I love cooking Italian food at home. Pasta carbonara and margherita pizza are my favorite "
    "dishes to make. Fresh ingredients make all the difference in authentic recipes.",
]


def load_model():
    lda = LdaModel.load(os.path.join(MODEL_DIR, "lda_model"))
    dictionary = corpora.Dictionary.load(os.path.join(MODEL_DIR, "dictionary.dict"))
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, encoding="utf-8") as f:
            labels = json.load(f)
    else:
        print(f"Warning: {LABELS_FILE} not found, using default topic names.")
        labels = {}
    labels = {i: labels.get(str(i), f"Topic {i}") for i in range(lda.num_topics)}
    return lda, dictionary, labels


def classify(text, lda, dictionary, top_n=3):
    """Return the top_n (topic_id, probability) pairs for a document."""
    bow = dictionary.doc2bow(preprocess(text))
    distribution = lda.get_document_topics(bow, minimum_probability=0.0)
    return sorted(distribution, key=lambda x: x[1], reverse=True)[:top_n]


def show_topics_summary(lda, labels):
    print("=" * 70)
    print("LOADED TOPICS")
    print("=" * 70)
    for topic_id in range(lda.num_topics):
        top = ", ".join(w for w, _ in lda.show_topic(topic_id, topn=8))
        print(f"{topic_id}. {labels[topic_id]:<25} {top}")


def show_classification(text, lda, dictionary, labels):
    preview = text if len(text) <= 100 else text[:100] + "..."
    print(f"\nDocument: {preview}")
    known = [t for t in preprocess(text) if t in dictionary.token2id]
    print(f"Known words in vocabulary: {len(known)} -> {known}")
    if not known:
        print("  (no known words, so the result is just the prior topic mix)")
    for rank, (topic_id, prob) in enumerate(classify(text, lda, dictionary), 1):
        words = ", ".join(w for w, _ in lda.show_topic(topic_id, topn=5))
        print(f"  {rank}. {labels[topic_id]:<25} {prob:6.2%}   [{words}]")


def main():
    lda, dictionary, labels = load_model()
    show_topics_summary(lda, labels)

    print("\n" + "=" * 70)
    print("SAMPLE DOCUMENT CLASSIFICATION")
    print("=" * 70)
    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        print(f"\n--- Sample {i} ---", end="")
        show_classification(doc, lda, dictionary, labels)


if __name__ == "__main__":
    main()
