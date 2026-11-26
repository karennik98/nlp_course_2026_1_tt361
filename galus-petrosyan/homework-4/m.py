import csv
import math
import random
import re
from collections import Counter, defaultdict


def preprocess_text(text):
    """Lowercases text, removes punctuation/special characters, and tokenizes into words."""
    text = text.lower()
    # Keep alphanumeric characters and spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return tokens


def load_data(filepath):
    """Loads spam.csv and returns a list of (label, tokenized_message) tuples."""
    dataset = []
    # utf-8-sig handles BOM and standard characters cleanly; fallback to latin-1 if needed
    with open(filepath, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized_row = {
                key.strip().lower(): value
                for key, value in row.items()
                if key is not None
            }
            label = (
                normalized_row.get("label")
                or normalized_row.get("category")
                or normalized_row.get("v1")
            )
            message = (
                normalized_row.get("message")
                or normalized_row.get("text")
                or normalized_row.get("v2")
            )
            if label and message:
                tokens = preprocess_text(message)
                if len(tokens) > 0:  # Ignore empty messages after cleaning
                    dataset.append((label.strip().lower(), tokens))
    return dataset


def train_test_split(dataset, train_ratio=0.8, seed=42):
    """Shuffles and splits the dataset into training and testing sets."""
    random.seed(seed)
    data_copy = list(dataset)
    random.shuffle(data_copy)
    split_idx = int(len(data_copy) * train_ratio)
    return data_copy[:split_idx], data_copy[split_idx:]


class MarkovBayesSpamClassifier:

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.log_prior_spam = 0.0
        self.log_prior_ham = 0.0

        # Vocabulary across the entire training set
        self.vocab = set()
        self.M = 0  # Vocabulary size

        # Counts for Spam
        self.spam_init_counts = Counter()
        self.spam_trans_counts = defaultdict(Counter)
        self.spam_word_counts = Counter()
        self.N_spam = 0

        # Counts for Ham
        self.ham_init_counts = Counter()
        self.ham_trans_counts = defaultdict(Counter)
        self.ham_word_counts = Counter()
        self.N_ham = 0

    def fit(self, train_data):
        """Trains the Markov Chain models for both spam and ham classes."""
        total_messages = len(train_data)
        if total_messages == 0:
            raise ValueError(
                "Training dataset is empty. Verify spam.csv path and column headers."
            )

        for label, tokens in train_data:
            for token in tokens:
                self.vocab.add(token)

            if label == "spam":
                self.N_spam += 1
                self.spam_init_counts[tokens[0]] += 1
                for i in range(len(tokens) - 1):
                    curr_word = tokens[i]
                    next_word = tokens[i + 1]
                    self.spam_trans_counts[curr_word][next_word] += 1
                    self.spam_word_counts[curr_word] += 1
            elif label == "ham":
                self.N_ham += 1
                self.ham_init_counts[tokens[0]] += 1
                for i in range(len(tokens) - 1):
                    curr_word = tokens[i]
                    next_word = tokens[i + 1]
                    self.ham_trans_counts[curr_word][next_word] += 1
                    self.ham_word_counts[curr_word] += 1

        if self.N_spam == 0 or self.N_ham == 0:
            raise ValueError(
                f"Both 'spam' and 'ham' samples are required (got spam={self.N_spam}, ham={self.N_ham})."
            )

        # Add 1 to M to account for unseen words (<UNK>) during inference
        self.M = len(self.vocab) + 1

        # Calculate log priors
        self.log_prior_spam = math.log(self.N_spam / total_messages)
        self.log_prior_ham = math.log(self.N_ham / total_messages)

    def _get_log_initial_prob(self, word, is_spam=True):
        r"""Computes log(\hat{\pi}_i) = log((count(s_1 = i) + \epsilon) / (N + \epsilon * M))"""
        if is_spam:
            count_s1 = self.spam_init_counts.get(word, 0)
            n_docs = self.N_spam
        else:
            count_s1 = self.ham_init_counts.get(word, 0)
            n_docs = self.N_ham

        prob = (count_s1 + self.epsilon) / (n_docs + self.epsilon * self.M)
        return math.log(prob)

    def _get_log_transition_prob(self, prev_word, curr_word, is_spam=True):
        r"""Computes log(\hat{A}_{ij}) = log((count(i -> j) + \epsilon) / (count(i) + \epsilon * M))"""
        if is_spam:
            trans_count = self.spam_trans_counts[prev_word].get(curr_word, 0)
            word_count = self.spam_word_counts.get(prev_word, 0)
        else:
            trans_count = self.ham_trans_counts[prev_word].get(curr_word, 0)
            word_count = self.ham_word_counts.get(prev_word, 0)

        prob = (trans_count + self.epsilon) / (word_count + self.epsilon * self.M)
        return math.log(prob)

    def _compute_log_likelihood(self, tokens, is_spam=True):
        r"""Computes log P(M | class) = log \pi_{s_1} + \sum_{t=2}^T log A_{s_{t-1}, s_t}"""
        if not tokens:
            return 0.0

        log_prob = self._get_log_initial_prob(tokens[0], is_spam=is_spam)
        for i in range(len(tokens) - 1):
            log_prob += self._get_log_transition_prob(
                tokens[i], tokens[i + 1], is_spam=is_spam
            )
        return log_prob

    def predict_one(self, tokens):
        """Predicts whether a single tokenized message is 'spam' or 'ham'."""
        log_p_spam_given_m = (
            self._compute_log_likelihood(tokens, is_spam=True)
            + self.log_prior_spam
        )
        log_p_ham_given_m = (
            self._compute_log_likelihood(tokens, is_spam=False)
            + self.log_prior_ham
        )

        return "spam" if log_p_spam_given_m > log_p_ham_given_m else "ham"

    def evaluate(self, test_data):
        """Evaluates accuracy, precision, recall, and F1-score on the test set."""
        correct = 0
        tp = fp = fn = tn = 0

        for label, tokens in test_data:
            pred = self.predict_one(tokens)
            if pred == label:
                correct += 1

            if label == "spam" and pred == "spam":
                tp += 1
            elif label == "ham" and pred == "spam":
                fp += 1
            elif label == "spam" and pred == "ham":
                fn += 1
            elif label == "ham" and pred == "ham":
                tn += 1

        accuracy = correct / len(test_data)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        }


if __name__ == "__main__":
    filepath = "spam.csv"
    print(f"Loading data from {filepath}...")
    dataset = load_data(filepath)
    print(f"Total valid messages loaded: {len(dataset)}")

    train_data, test_data = train_test_split(dataset, train_ratio=0.8, seed=42)
    print(f"Training set size: {len(train_data)} | Test set size: {len(test_data)}")

    classifier = MarkovBayesSpamClassifier(epsilon=0.1)
    classifier.fit(train_data)
    print(f"Vocabulary size (including <UNK>): {classifier.M}")

    metrics = classifier.evaluate(test_data)
    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:  {metrics['f1_score'] * 100:.2f}%")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")

    sample_messages = [
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461.",
        "Hey, are we still meeting for lunch at 1pm today?",
        "URGENT! Your mobile number has won $2000. Reply YES to claim now!",
    ]
    print("\n--- Sample Predictions ---")
    for msg in sample_messages:
        tokens = preprocess_text(msg)
        prediction = classifier.predict_one(tokens)
        print(f"[{prediction.upper()}] {msg}")