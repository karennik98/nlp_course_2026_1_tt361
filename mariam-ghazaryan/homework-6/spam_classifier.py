"""SMS Spam Classification using Markov Chains and Bayes' Theorem.

Each class (spam / ham) gets its own first-order Markov model of word
sequences. A message M = [w1, ..., wn] is scored in log space:

    log P(c | M)  ∝  log P(c) + log P(w1 | c) + Σ log P(wt | wt-1, c)

and assigned to the class with the higher score.

Add-one (Laplace) smoothing, V = vocabulary size:
    initial:     P(w1 | c)        = (count_c(w1 first) + 1) / (messages_c + V)
    transition:  P(wt | wt-1, c)  = (count_c(wt-1, wt) + 1) / (count_c(wt-1) + V)
"""

import csv
import math
import random
import re
import sys

FILE_NAME = "spam.csv"
TRAIN_RATIO = 0.8
SEED = 42
CLASSES = ("spam", "ham")


# ---------------------------------------------------------------- data

def load_data(file_name):
    """Load (label, message) pairs from the CSV.

    Supports headers 'label'/'message' and the Kaggle variant 'v1'/'v2'.
    """
    try:
        with open(file_name, "r", encoding="latin-1", newline="") as f:
            reader = csv.reader(f)
            header = [h.strip().lower() for h in next(reader)]
            label_col = header.index("label") if "label" in header else 0
            text_col = header.index("message") if "message" in header else 1
            data = []
            for row in reader:
                if len(row) <= max(label_col, text_col):
                    continue
                label = row[label_col].strip().lower()
                if label in CLASSES:
                    data.append((label, row[text_col]))
            return data
    except FileNotFoundError:
        sys.exit(f"Error: file '{file_name}' not found.")
    except (OSError, StopIteration) as e:
        sys.exit(f"Error: could not read '{file_name}': {e!r}")


def preprocess(message):
    """Lowercase, keep only letters/digits, split into words."""
    return re.sub(r"[^a-z0-9\s]", " ", message.lower()).split()


def train_test_split(data, train_ratio=TRAIN_RATIO, seed=SEED):
    """Shuffle and split into train (80%) and test (20%)."""
    data = data[:]
    random.Random(seed).shuffle(data)
    cut = int(len(data) * train_ratio)
    return data[:cut], data[cut:]


# ---------------------------------------------------------------- model

class MarkovBayesClassifier:
    def fit(self, train):
        self.log_prior = {}
        self.n_messages = {}                           # messages with >= 1 word
        self.initial = {c: {} for c in CLASSES}        # count(w as first word)
        self.transitions = {c: {} for c in CLASSES}    # count(w_prev, w)
        self.prev_totals = {c: {} for c in CLASSES}    # count(w_prev)
        self.vocab = {c: set() for c in CLASSES}

        for c in CLASSES:
            messages = [preprocess(text) for label, text in train if label == c]
            # log P(c) = log(messages of class c / total messages)
            self.log_prior[c] = math.log(len(messages) / len(train))

            messages = [m for m in messages if m]
            self.n_messages[c] = len(messages)
            for words in messages:
                self.vocab[c].update(words)
                self.initial[c][words[0]] = self.initial[c].get(words[0], 0) + 1
                for prev, word in zip(words, words[1:]):
                    row = self.transitions[c].setdefault(prev, {})
                    row[word] = row.get(word, 0) + 1
                    self.prev_totals[c][prev] = self.prev_totals[c].get(prev, 0) + 1

        # Shared vocabulary size (+1 slot for unseen words) so that both
        # class models are smoothed over the same event space.
        self.V = len(self.vocab["spam"] | self.vocab["ham"]) + 1
        return self

    def log_initial(self, word, c):
        return math.log((self.initial[c].get(word, 0) + 1) / (self.n_messages[c] + self.V))

    def log_transition(self, prev, word, c):
        count = self.transitions[c].get(prev, {}).get(word, 0)
        return math.log((count + 1) / (self.prev_totals[c].get(prev, 0) + self.V))

    def log_posterior(self, words, c):
        """log P(M | c) + log P(c)"""
        score = self.log_prior[c]
        if not words:
            return score
        score += self.log_initial(words[0], c)
        for prev, word in zip(words, words[1:]):
            score += self.log_transition(prev, word, c)
        return score

    def predict(self, message):
        words = preprocess(message)
        spam = self.log_posterior(words, "spam")
        ham = self.log_posterior(words, "ham")
        return "spam" if spam > ham else "ham"


# ---------------------------------------------------------------- evaluation

def evaluate(model, test):
    tp = tn = fp = fn = 0
    for label, text in test:
        pred = model.predict(text)
        if pred == "spam":
            if label == "spam":
                tp += 1
            else:
                fp += 1
        else:
            if label == "ham":
                tn += 1
            else:
                fn += 1
    accuracy = (tp + tn) / len(test)
    return accuracy, tp, tn, fp, fn


def main():
    data = load_data(FILE_NAME)
    if not data:
        sys.exit(f"Error: no labelled messages found in '{FILE_NAME}'.")
    train, test = train_test_split(data)
    model = MarkovBayesClassifier().fit(train)

    print(f"Messages: {len(data)} (train {len(train)}, test {len(test)})")
    print(f"log P(spam) = {model.log_prior['spam']:.4f}")
    print(f"log P(ham)  = {model.log_prior['ham']:.4f}")
    print(f"Vocabulary: spam {len(model.vocab['spam'])}, ham {len(model.vocab['ham'])}")

    accuracy, tp, tn, fp, fn = evaluate(model, test)
    print(f"\nTest accuracy: {accuracy:.4f} ({tp + tn}/{len(test)})")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print("            spam    ham")
    print(f"  spam    {tp:6d} {fn:6d}")
    print(f"  ham     {fp:6d} {tn:6d}")

    print("\nExamples:")
    for text in ("Congratulations! You have won a free prize, call now to claim",
                 "Hey, are we still meeting for lunch tomorrow?"):
        print(f"  [{model.predict(text)}] {text}")


if __name__ == "__main__":
    main()
