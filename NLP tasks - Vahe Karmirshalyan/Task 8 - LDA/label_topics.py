"""
Part 2 - Topic labeling script.

Shows each topic's top words and asks the user for a meaningful name.
Press Enter to keep the current name ("Topic N" on the first run,
the previously saved name on later runs). Labels are saved to JSON.
"""

import json
import os
import sys

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from train_lda import DICTIONARY_PATH, MODEL_PATH, MODELS_DIR

TOP_WORDS = 20
SUMMARY_WORDS = 5
LABELS_PATH = os.path.join(MODELS_DIR, "topic_labels.json")


def load_model():
    """Load the trained LDA model and dictionary saved by train_lda.py."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DICTIONARY_PATH):
        sys.exit("Model not found. Run train_lda.py first.")

    lda_model = LdaModel.load(MODEL_PATH)
    dictionary = Dictionary.load(DICTIONARY_PATH)
    print(f"Loaded model with {lda_model.num_topics} topics "
          f"and a dictionary of {len(dictionary)} words")
    return lda_model, dictionary


def load_labels(num_topics):
    """Return saved labels if they exist, otherwise default names 'Topic N'."""
    labels = {topic_id: f"Topic {topic_id}" for topic_id in range(num_topics)}

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, encoding="utf-8") as file:
            saved = json.load(file)
        # JSON keys are strings ("0"), convert back to int (0)
        labels.update({int(topic_id): name for topic_id, name in saved.items()})

    return labels


def display_topic(lda_model, topic_id, current_name):
    """Print a topic's top words with their probabilities."""
    print(f"\n{'-' * 70}")
    print(f"Topic {topic_id}   (current name: {current_name})")
    print(f"{'-' * 70}")
    for rank, (word, prob) in enumerate(lda_model.show_topic(topic_id, topn=TOP_WORDS), start=1):
        print(f"  {rank:2d}. {word:<15} {prob:.4f}")


def ask_label(topic_id, current_name):
    """Ask for a topic name; an empty answer keeps the current name."""
    name = input(f"\nName for Topic {topic_id} (Enter to keep '{current_name}'): ").strip()
    return name if name else current_name


def save_labels(labels):
    """Save labels to JSON (keys become strings, non-English characters kept readable)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump({str(topic_id): name for topic_id, name in labels.items()},
                  file, indent=4, ensure_ascii=False)
    print(f"\nLabels saved to {LABELS_PATH}")


def print_summary(lda_model, labels):
    """Print every topic with its assigned name and top words."""
    print(f"\n{'=' * 70}\nTOPIC SUMMARY\n{'=' * 70}")
    for topic_id, name in labels.items():
        words = ", ".join(word for word, _ in lda_model.show_topic(topic_id, topn=SUMMARY_WORDS))
        print(f"Topic {topic_id}: {name:<30} | {words}")


def main():
    lda_model, _ = load_model()
    labels = load_labels(lda_model.num_topics)
    if os.path.exists(LABELS_PATH):
        print(f"Found existing labels in {LABELS_PATH} - Enter keeps the saved name")

    print(f"\nFor each topic, read its top {TOP_WORDS} words and type a meaningful name.")
    for topic_id in range(lda_model.num_topics):
        display_topic(lda_model, topic_id, labels[topic_id])
        labels[topic_id] = ask_label(topic_id, labels[topic_id])

    save_labels(labels)
    print_summary(lda_model, labels)


if __name__ == "__main__":
    main()
