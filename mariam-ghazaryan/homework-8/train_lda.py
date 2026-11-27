"""Part 1: train an LDA model on the first 1000 training docs of 20 Newsgroups."""

import os

from gensim import corpora
from gensim.models import LdaModel
from sklearn.datasets import fetch_20newsgroups

from preprocessing import preprocess

MODEL_DIR = "models"
NUM_DOCS = 1000
NUM_TOPICS = 10
PASSES = 15


def main():
    # 1. Load data
    print("Loading 20 Newsgroups dataset...")
    data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    documents = data.data[:NUM_DOCS]
    print(f"Using {len(documents)} documents")

    # 2. Preprocess
    print("Preprocessing...")
    texts = [preprocess(doc) for doc in documents]

    # 3. Dictionary + filter rare (<5 docs) and common (>50% of docs) words
    dictionary = corpora.Dictionary(texts)
    print(f"Vocabulary before filtering: {len(dictionary)}")
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    print(f"Vocabulary after filtering:  {len(dictionary)}")

    # 4. Bag-of-Words corpus
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 5. Train LDA
    print(f"Training LDA ({NUM_TOPICS} topics, {PASSES} passes)...")
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=PASSES,
        alpha="auto",
        eta="auto",
        random_state=42,
    )

    # 6. Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    lda.save(os.path.join(MODEL_DIR, "lda_model"))
    dictionary.save(os.path.join(MODEL_DIR, "dictionary.dict"))
    print(f"Model and dictionary saved to '{MODEL_DIR}/'")

    # 7. Print topics with top 15 words
    print("\n" + "=" * 70)
    print("DISCOVERED TOPICS (top 15 words)")
    print("=" * 70)
    for topic_id in range(NUM_TOPICS):
        words = [w for w, _ in lda.show_topic(topic_id, topn=15)]
        print(f"Topic {topic_id}: {', '.join(words)}")


if __name__ == "__main__":
    main()
