"""
Part 3 - Inference script.

Classifies new documents with the trained LDA model:
text -> same preprocessing as training -> Bag-of-Words -> topic distribution -> top 3 topics.
Topic numbers are shown with the names assigned in label_topics.py.
"""

import os
import sys

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from label_topics import LABELS_PATH, load_labels
from train_lda import DICTIONARY_PATH, MODEL_PATH, preprocess

TOP_N_TOPICS = 3
TOP_WORDS = 5
PREVIEW_CHARS = 100

SAMPLE_DOCUMENTS = [
    "The new graphics card delivers amazing performance for gaming. The GPU can handle "
    "4K resolution easily with ray tracing enabled. Gamers will love the improved frame rates.",

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


def load_resources():
    """Load the trained model, dictionary and topic labels."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DICTIONARY_PATH):
        sys.exit("Model not found. Run train_lda.py first.")

    lda_model = LdaModel.load(MODEL_PATH)
    dictionary = Dictionary.load(DICTIONARY_PATH)

    if not os.path.exists(LABELS_PATH):
        print("Warning: no topic labels found (run label_topics.py) - using 'Topic N' names")
    labels = load_labels(lda_model.num_topics)

    return lda_model, dictionary, labels


def top_words(lda_model, topic_id, count=TOP_WORDS):
    return ", ".join(word for word, _ in lda_model.show_topic(topic_id, topn=count))


def print_topic_summary(lda_model, labels):
    """Print all loaded topics with their names and top words."""
    print(f"{'=' * 70}\nLOADED TOPICS\n{'=' * 70}")
    for topic_id, name in labels.items():
        print(f"Topic {topic_id}: {name:<25} | {top_words(lda_model, topic_id)}")


def classify(text, lda_model, dictionary, top_n=TOP_N_TOPICS):
    """
    Classify a document into topics.
    Returns the top `top_n` topics as [(topic_id, probability), ...],
    or an empty list if none of the document's words are in the dictionary.
    """
    tokens = preprocess(text)
    bow = dictionary.doc2bow(tokens)
    if not bow:
        return []

    distribution = lda_model.get_document_topics(bow, minimum_probability=0)
    return sorted(distribution, key=lambda item: item[1], reverse=True)[:top_n]


def split_known_unknown(text, dictionary):
    """Split the preprocessed words into those the model knows and those it ignores."""
    tokens = list(dict.fromkeys(preprocess(text)))  # unique, original order
    known = [token for token in tokens if token in dictionary.token2id]
    unknown = [token for token in tokens if token not in dictionary.token2id]
    return known, unknown


def display_result(number, text, results, lda_model, dictionary, labels):
    """Print a document preview, the words used, and its top topics."""
    preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "..."
    known, unknown = split_known_unknown(text, dictionary)

    print(f"\n{'=' * 70}")
    print(f"Sample {number}: \"{preview}\"")
    print(f"Known words   ({len(known)}): {', '.join(known) or '-'}")
    print(f"Unknown words ({len(unknown)}): {', '.join(unknown) or '-'}")
    print("-" * 70)

    if not results:
        print("No known words - the model cannot classify this document.")
        return

    for rank, (topic_id, prob) in enumerate(results, start=1):
        print(f"{rank}. {labels[topic_id]:<25} {prob:6.1%}  | {top_words(lda_model, topic_id)}")


def main():
    lda_model, dictionary, labels = load_resources()
    print_topic_summary(lda_model, labels)

    print(f"\n\nCLASSIFYING {len(SAMPLE_DOCUMENTS)} SAMPLE DOCUMENTS")
    for number, text in enumerate(SAMPLE_DOCUMENTS, start=1):
        results = classify(text, lda_model, dictionary)
        display_result(number, text, results, lda_model, dictionary, labels)


if __name__ == "__main__":
    main()
