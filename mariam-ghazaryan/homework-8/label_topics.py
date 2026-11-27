"""Part 2: interactively give human-readable names to the LDA topics."""

import json
import os

from gensim import corpora
from gensim.models import LdaModel

MODEL_DIR = "models"
LABELS_FILE = os.path.join(MODEL_DIR, "topic_labels.json")


def main():
    lda = LdaModel.load(os.path.join(MODEL_DIR, "lda_model"))
    corpora.Dictionary.load(os.path.join(MODEL_DIR, "dictionary.dict"))

    labels = {}
    for topic_id in range(lda.num_topics):
        default_name = f"Topic {topic_id}"
        print("\n" + "=" * 70)
        print(f"{default_name} - top 20 words")
        print("=" * 70)
        for word, prob in lda.show_topic(topic_id, topn=20):
            print(f"  {word:<20} {prob:.4f}")

        name = input(f"\nName for {default_name} (Enter to keep default): ").strip()
        labels[str(topic_id)] = name or default_name

    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"\nLabels saved to {LABELS_FILE}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for topic_id in range(lda.num_topics):
        top = ", ".join(w for w, _ in lda.show_topic(topic_id, topn=5))
        print(f"Topic {topic_id}: {labels[str(topic_id)]:<25} [{top}]")


if __name__ == "__main__":
    main()
