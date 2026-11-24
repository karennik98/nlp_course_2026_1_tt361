
import pandas as pd
import re
import math

from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path


EPSILON = 0.001


def preprocess(message):
    message = str(message).lower()
    message = re.sub(r"[^\w\s]", "", message)
    return message.split()


class MarkovModel:

    def __init__(self):
        self.initial_counts = Counter()
        self.transition_counts = defaultdict(Counter)
        self.vocabulary = set()
        self.total_initial = 0

    def train(self, messages):

        for message in messages:
            words = preprocess(message)

            if not words:
                continue

            self.vocabulary.update(words)

            self.initial_counts[words[0]] += 1
            self.total_initial += 1

            for i in range(len(words) - 1):

                current_word = words[i]
                next_word = words[i + 1]

                self.transition_counts[current_word][next_word] += 1

    def initial_probability(self, word):

        vocabulary_size = len(self.vocabulary) + 1

        numerator = self.initial_counts.get(word, 0) + EPSILON

        denominator = (
            self.total_initial
            + EPSILON * vocabulary_size
        )

        return numerator / denominator

    def transition_probability(self, current_word, next_word):

        vocabulary_size = len(self.vocabulary) + 1

        counts = self.transition_counts.get(current_word, {})

        numerator = counts.get(next_word, 0) + EPSILON

        denominator = (
            sum(counts.values())
            + EPSILON * vocabulary_size
        )

        return numerator / denominator

    def log_likelihood(self, message):

        words = preprocess(message)

        if not words:
            return 0

        probability = math.log(
            self.initial_probability(words[0])
        )

        for i in range(len(words) - 1):

            transition = self.transition_probability(
                words[i],
                words[i + 1]
            )

            probability += math.log(transition)

        return probability


class SpamClassifier:

    def __init__(self):

        self.spam_model = MarkovModel()
        self.ham_model = MarkovModel()

        self.log_prior_spam = 0
        self.log_prior_ham = 0

    def train(self, data):

        spam_messages = data[
            data["label"] == "spam"
        ]["message"]

        ham_messages = data[
            data["label"] == "ham"
        ]["message"]

        total_messages = len(data)

        self.log_prior_spam = math.log(
            len(spam_messages) / total_messages
        )

        self.log_prior_ham = math.log(
            len(ham_messages) / total_messages
        )

        self.spam_model.train(spam_messages)
        self.ham_model.train(ham_messages)

        vocabulary = (
            self.spam_model.vocabulary
            | self.ham_model.vocabulary
        )

        self.spam_model.vocabulary = vocabulary
        self.ham_model.vocabulary = vocabulary

    def predict(self, message):

        spam_score = (
            self.spam_model.log_likelihood(message)
            + self.log_prior_spam
        )

        ham_score = (
            self.ham_model.log_likelihood(message)
            + self.log_prior_ham
        )

        if spam_score > ham_score:
            return "spam"

        return "ham"


def save_results(data, train_data, test_data,
                 predictions, accuracy, classifier):

    results_file = Path(__file__).parent / "results.txt"

    with open(results_file, "w", encoding="utf-8") as file:

        file.write("SMS Spam Classification Results\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Dataset size: {len(data)}\n")
        file.write(f"Training size: {len(train_data)}\n")
        file.write(f"Testing size: {len(test_data)}\n")

        file.write(f"Accuracy: {accuracy * 100:.2f}%\n")

        file.write(
            f"Correct predictions: "
            f"{sum(a == p for a, p in zip(test_data['label'], predictions))}\n"
        )

        file.write(
            f"Incorrect predictions: "
            f"{sum(a != p for a, p in zip(test_data['label'], predictions))}\n"
        )

        file.write("\nPrior Probabilities:\n")

        file.write(
            f"P(spam): {math.exp(classifier.log_prior_spam):.4f}\n"
        )

        file.write(
            f"P(ham): {math.exp(classifier.log_prior_ham):.4f}\n"
        )

        file.write("\nClassification Results:\n")
        file.write("=" * 60 + "\n")

        for i, (message, actual, predicted) in enumerate(
            zip(
                test_data["message"],
                test_data["label"],
                predictions
            ),
            start=1
        ):

            file.write(f"\nMessage #{i}\n")
            file.write(f"Message: {message}\n")
            file.write(f"Actual label: {actual}\n")
            file.write(f"Predicted label: {predicted}\n")
            file.write(f"Correct: {actual == predicted}\n")
            file.write("-" * 60 + "\n")

    print("\nResults saved to results.txt")


def main():

    filename = Path(__file__).parent / "spam.csv"

    try:

        data = pd.read_csv(
            filename,
            encoding="utf-8-sig"
        )

    except UnicodeDecodeError:

        data = pd.read_csv(
            filename,
            encoding="latin-1"
        )

    except (OSError, pd.errors.ParserError) as error:

        print("File error:", error)
        return

    data = data.rename(columns={
        "Category": "label",
        "Message": "message",
        "v1": "label",
        "v2": "message"
    })

    if not {"label", "message"}.issubset(data.columns):

        print("Invalid dataset columns:", data.columns.tolist())
        return

    data = data[["label", "message"]].dropna()

    data["label"] = data["label"].str.lower().str.strip()

    data = data[data["label"].isin(["spam", "ham"])]

    if data["label"].value_counts().reindex(
        ["spam", "ham"], fill_value=0
    ).min() < 5:

        print("Not enough messages in each class.")
        return

    print("=" * 50)
    print("SMS SPAM CLASSIFICATION")
    print("=" * 50)

    print("Dataset size:", len(data))

    print("\nClass distribution:")
    print(data["label"].value_counts())

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["label"]
    )

    print("\nTraining size:", len(train_data))
    print("Testing size:", len(test_data))

    classifier = SpamClassifier()
    classifier.train(train_data)

    print("\nTraining completed successfully!")

    predictions = [
        classifier.predict(message)
        for message in test_data["message"]
    ]

    accuracy = accuracy_score(
        test_data["label"],
        predictions
    )

    print(f"Accuracy: {accuracy * 100:.2f}%")

    save_results(
        data,
        train_data,
        test_data,
        predictions,
        accuracy,
        classifier
    )

    messages = [
        "Congratulations! You won a free prize!",
        "Hello, how are you?",
        "Call now to claim your reward",
        "See you tomorrow at university"
    ]

    print("\nExample predictions:")
    print("-" * 50)

    for message in messages:

        prediction = classifier.predict(message)

        print(f"{message} --> {prediction}")

    while True:

        message = input("\nEnter SMS (or exit): ")

        if message.strip().lower() == "exit":
            break

        if not message.strip():
            continue

        prediction = classifier.predict(message)

        print("Prediction:", prediction)

        results_file = Path(__file__).parent / "results.txt"

        with open(results_file, "a", encoding="utf-8") as file:

            file.write("\nManual prediction:\n")
            file.write(f"Message: {message}\n")
            file.write(f"Predicted label: {prediction}\n")
            file.write("-" * 60 + "\n")


if __name__ == "__main__":
    main()
