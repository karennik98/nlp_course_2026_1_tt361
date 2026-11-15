"""
SMS Spam Classification using Markov Chains and Bayes' Theorem.

Steps:
    1. Load data            (load_data)
    2. Preprocess           (preprocess)
    3. Train / test split   (split_data)
    4. Train Markov chains  (MarkovChain, MarkovBayesClassifier.fit)
    5. Predict              (MarkovChain.log_prob, MarkovBayesClassifier.predict)
    6. Evaluate             (evaluate, compare_epsilons)
"""

import math
import re
from collections import Counter

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_PATH = "spam.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPSILON = 1.0
EPSILON_CANDIDATES = [1.0, 0.1, 0.01, 0.001]
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
def load_data(path):
    """
    Load spam.csv and return a DataFrame with columns ['label', 'message'].

    The raw file has columns v1 (label), v2 (message) and 3 unnamed columns.
    For ~50 rows the CSV parser split the message at its commas and put the
    pieces into the unnamed columns, so we join them back with ','.
    """
    # The file is not UTF-8, latin-1 can decode any byte.
    raw = pd.read_csv(path, encoding="latin-1")

    # Columns 1..end hold the message (v2 + the overflow columns).
    message_parts = raw.iloc[:, 1:].fillna("")
    messages = message_parts.apply(
        lambda row: ",".join(part for part in row if part != ""),
        axis=1,
    )

    df = pd.DataFrame({"label": raw.iloc[:, 0], "message": messages})
    return df


def describe_data(df):
    """Print basic facts about the dataset."""
    print("Shape:", df.shape)
    print()

    counts = df["label"].value_counts()
    print("Label counts:")
    for label, count in counts.items():
        print(f"  {label:<5} {count:>5}  ({count / len(df):.1%})")
    print(f"Baseline accuracy (always 'ham'): {counts['ham'] / len(df):.1%}")
    print()

    print("Duplicate messages:", df["message"].duplicated().sum())
    print()

    print("Examples:")
    for label in ("ham", "spam"):
        for message in df[df["label"] == label]["message"].head(2):
            print(f"  [{label}] {message}")


# ---------------------------------------------------------------------------
# Step 2: Preprocess
# ---------------------------------------------------------------------------
def preprocess(message):
    """
    Turn a raw SMS into a list of tokens (the states of the Markov chain).

        lowercase -> remove apostrophes -> non [a-z0-9] chars become spaces -> split

    Numbers are kept as they are (e.g. '87121', '21st').
    """
    text = message.lower()
    text = text.replace("'", "")             # don't -> dont
    text = re.sub(r"[^a-z0-9]+", " ", text)  # punctuation, symbols -> space
    return text.split()


def describe_tokens(df):
    """Print before/after examples and token statistics."""
    print("Before -> after:")
    for label in ("ham", "spam"):
        for message in df[df["label"] == label]["message"].head(2):
            print(f"  [{label}] {message}")
            print(f"         {preprocess(message)}")
    print()

    lengths = df["tokens"].apply(len)
    for label in ("ham", "spam"):
        mask = df["label"] == label
        print(f"{label:<5} avg tokens: {lengths[mask].mean():5.1f}   "
              f"empty: {(lengths[mask] == 0).sum()}   "
              f"one-token: {(lengths[mask] == 1).sum()}")

    vocabulary = {token for tokens in df["tokens"] for token in tokens}
    print("Vocabulary size (whole dataset):", len(vocabulary))

    print("Messages that became empty:")
    for message in df[lengths == 0]["message"]:
        print(f"  {message!r}")


def remove_duplicates(df):
    """Keep one copy of each message (no duplicate has conflicting labels)."""
    before = len(df)
    deduped = df.drop_duplicates(subset="message").reset_index(drop=True)

    counts = deduped["label"].value_counts()
    print(f"Removed duplicates: {before - len(deduped)}   remaining: {len(deduped)}")
    print(f"  ham: {counts['ham']}   spam: {counts['spam']} ({counts['spam'] / len(deduped):.1%})")
    print(f"New baseline accuracy (always 'ham'): {counts['ham'] / len(deduped):.1%}")
    return deduped


# ---------------------------------------------------------------------------
# Step 3: Train / test split
# ---------------------------------------------------------------------------
def split_data(df):
    """
    Split into 80% train / 20% test.

    stratify keeps the ham/spam ratio the same in both parts,
    random_state makes the split reproducible.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    return train_df, test_df


def describe_split(train_df, test_df):
    """Print sizes, class ratios and overlap of the two parts."""
    for name, part in (("train", train_df), ("test", test_df)):
        n_spam = (part["label"] == "spam").sum()
        print(f"{name:<5} {len(part):>5} messages   "
              f"ham: {len(part) - n_spam:>5}   spam: {n_spam:>4} ({n_spam / len(part):.1%})")

    train_messages = set(train_df["message"])
    overlap = test_df["message"].isin(train_messages).sum()
    print(f"Test messages that also appear in train (duplicates): {overlap}")

    train_vocabulary = {t for tokens in train_df["tokens"] for t in tokens}
    test_tokens = [t for tokens in test_df["tokens"] for t in tokens]
    unseen = sum(t not in train_vocabulary for t in test_tokens)
    print(f"Train vocabulary size: {len(train_vocabulary)}")
    print(f"Test tokens never seen in train: {unseen} / {len(test_tokens)} "
          f"({unseen / len(test_tokens):.1%})")


# ---------------------------------------------------------------------------
# Step 4: Train Markov chains
# ---------------------------------------------------------------------------
class MarkovChain:
    """
    First-order Markov chain over words, trained on the messages of ONE class.

    Only counts are stored (no dense M x M matrix). Probabilities are computed
    from the counts with add-epsilon smoothing:

        pi_i  = (count(s1 = i) + eps) / (N + eps * M)
        A_ij  = (count(i -> j) + eps) / (count(i) + eps * M)

    N        = number of messages (sequences) in this class
    M        = vocabulary size (shared by both classes)
    count(i) = number of times i is followed by another word
    """

    def __init__(self, vocabulary_size, epsilon):
        self.M = vocabulary_size
        self.epsilon = epsilon
        self.N = 0
        self.initial_counts = Counter()     # count(s1 = i)
        self.transition_counts = Counter()  # count(i -> j), key = (i, j)
        self.from_counts = Counter()        # count(i)

    def fit(self, token_lists):
        for tokens in token_lists:
            if not tokens:                  # empty message: nothing to count
                continue
            self.N += 1
            self.initial_counts[tokens[0]] += 1
            for current_word, next_word in zip(tokens, tokens[1:]):
                self.transition_counts[(current_word, next_word)] += 1
                self.from_counts[current_word] += 1
        return self

    def log_initial(self, word):
        """log pi_word"""
        numerator = self.initial_counts[word] + self.epsilon
        denominator = self.N + self.epsilon * self.M
        return math.log(numerator / denominator)

    def log_transition(self, current_word, next_word):
        """log A_{current_word, next_word}"""
        numerator = self.transition_counts[(current_word, next_word)] + self.epsilon
        denominator = self.from_counts[current_word] + self.epsilon * self.M
        return math.log(numerator / denominator)

    # Step 5
    def log_prob(self, tokens):
        """
        log P(tokens | this class) = log pi_{w1} + sum_t log A_{w(t-1), w(t)}

        An empty message has no words, so it adds nothing (log 1 = 0).
        """
        if not tokens:
            return 0.0
        total = self.log_initial(tokens[0])
        for current_word, next_word in zip(tokens, tokens[1:]):
            total += self.log_transition(current_word, next_word)
        return total


class MarkovBayesClassifier:
    """One Markov chain per class + log prior probabilities (Bayes' theorem)."""

    def __init__(self, epsilon=EPSILON):
        self.epsilon = epsilon
        self.vocabulary = set()
        self.log_priors = {}
        self.chains = {}

    def fit(self, train_df):
        # Shared vocabulary -> same M for both classes (training data only!)
        self.vocabulary = {t for tokens in train_df["tokens"] for t in tokens}
        M = len(self.vocabulary)

        for label in ("ham", "spam"):
            class_tokens = train_df[train_df["label"] == label]["tokens"]
            self.log_priors[label] = math.log(len(class_tokens) / len(train_df))
            self.chains[label] = MarkovChain(M, self.epsilon).fit(class_tokens)
        return self

    # Step 5
    def scores(self, tokens):
        """
        log P(c | M) up to the constant log P(M), for each class c:
            log P(c) + log P(M | c)
        P(M) is the same for both classes, so it is never needed.
        """
        return {label: self.log_priors[label] + chain.log_prob(tokens)
                for label, chain in self.chains.items()}

    def predict(self, tokens):
        """'spam' if log P(spam|M) > log P(ham|M), otherwise 'ham'."""
        s = self.scores(tokens)
        return "spam" if s["spam"] > s["ham"] else "ham"

    def predict_many(self, token_lists):
        return [self.predict(tokens) for tokens in token_lists]


def explain_prediction(model, tokens):
    """Print every term of the score for both classes, side by side."""
    spam, ham = model.chains["spam"], model.chains["ham"]
    print(f"  {'term':<28}{'spam':>9}{'ham':>9}{'spam-ham':>10}")

    rows = [("log prior", model.log_priors["spam"], model.log_priors["ham"])]
    if tokens:
        rows.append((f"log pi({tokens[0]})",
                     spam.log_initial(tokens[0]), ham.log_initial(tokens[0])))
        for i, j in zip(tokens, tokens[1:]):
            rows.append((f"log A({i} -> {j})",
                         spam.log_transition(i, j), ham.log_transition(i, j)))

    for name, s, h in rows:
        print(f"  {name:<28}{s:9.2f}{h:9.2f}{s - h:10.2f}")
    total = model.scores(tokens)
    print(f"  {'TOTAL':<28}{total['spam']:9.2f}{total['ham']:9.2f}"
          f"{total['spam'] - total['ham']:10.2f}  -> {model.predict(tokens)}")


def describe_predictions(model, test_df):
    """Show how predictions are made (accuracy comes in Step 6)."""
    # One short spam and one short ham message from the test set, term by term
    for label in ("spam", "ham"):
        short = test_df[(test_df["label"] == label) & (test_df["tokens"].apply(len).between(5, 8))]
        row = short.iloc[0]
        print(f"True label: {label}   message: {row['message']!r}")
        explain_prediction(model, row["tokens"])
        print()

    # Edge cases
    for tokens in ([], ["ok"]):
        s = model.scores(tokens)
        print(f"Edge case {tokens}: spam {s['spam']:.2f}  ham {s['ham']:.2f}  "
              f"-> {model.predict(tokens)}")
    print()

    # Predictions on the whole test set (just counts here)
    predictions = model.predict_many(test_df["tokens"])
    predicted = Counter(predictions)
    actual = Counter(test_df["label"])
    print(f"Test set: predicted ham {predicted['ham']}, spam {predicted['spam']}   "
          f"(actual ham {actual['ham']}, spam {actual['spam']})")
    return predictions


def describe_model(model):
    """Print what the trained model contains, plus sanity checks."""
    M = len(model.vocabulary)
    print(f"Vocabulary size M = {M}   epsilon = {model.epsilon}")
    print()

    for label in ("ham", "spam"):
        chain = model.chains[label]
        print(f"[{label}]  log P({label}) = {model.log_priors[label]:.4f}  "
              f"(P = {math.exp(model.log_priors[label]):.3f})")
        print(f"  N (non-empty messages): {chain.N}")
        print(f"  distinct first words:   {len(chain.initial_counts)}")
        print(f"  distinct transitions:   {len(chain.transition_counts)}  "
              f"(of M*M = {M * M:,} possible -> {len(chain.transition_counts) / (M * M):.4%} seen)")
        print(f"  total transitions:      {sum(chain.transition_counts.values())}")

        top_first = ", ".join(f"{w}({c})" for w, c in chain.initial_counts.most_common(8))
        print(f"  top first words:  {top_first}")
        top_pairs = ", ".join(f"{i}->{j}({c})" for (i, j), c in chain.transition_counts.most_common(8))
        print(f"  top transitions:  {top_pairs}")
        print()

    # Same words under both chains: where do the classes disagree?
    print("Example probabilities (spam vs ham):")
    examples = [("initial", "free"), ("initial", "ok"),
                ("transition", ("call", "now")), ("transition", ("i", "am")),
                ("transition", ("to", "claim")), ("transition", ("zzz", "qqq"))]
    for kind, item in examples:
        if kind == "initial":
            s, h = model.chains["spam"].log_initial(item), model.chains["ham"].log_initial(item)
            name = f"pi({item})"
        else:
            s, h = model.chains["spam"].log_transition(*item), model.chains["ham"].log_transition(*item)
            name = f"A({item[0]} -> {item[1]})"
        print(f"  {name:<22} spam: {math.exp(s):.6f} (log {s:7.3f})   "
              f"ham: {math.exp(h):.6f} (log {h:7.3f})")
    print()

    # Sanity check: pi and every row of A must sum to 1 over the vocabulary.
    print("Sanity check (should all be 1.0):")
    for label in ("ham", "spam"):
        chain = model.chains[label]
        pi_sum = sum(math.exp(chain.log_initial(w)) for w in model.vocabulary)
        row_sum = sum(math.exp(chain.log_transition("to", w)) for w in model.vocabulary)
        print(f"  [{label}] sum of pi = {pi_sum:.6f}   sum of A('to' -> *) = {row_sum:.6f}")


# ---------------------------------------------------------------------------
# Step 6: Evaluate
# ---------------------------------------------------------------------------
def evaluate(true_labels, predicted_labels):
    """
    Confusion matrix and metrics, with 'spam' as the positive class.

        TP = spam predicted as spam     FN = spam predicted as ham (missed)
        FP = ham predicted as spam      TN = ham predicted as ham
    """
    pairs = list(zip(true_labels, predicted_labels))
    tp = sum(t == "spam" and p == "spam" for t, p in pairs)
    fn = sum(t == "spam" and p == "ham" for t, p in pairs)
    fp = sum(t == "ham" and p == "spam" for t, p in pairs)
    tn = sum(t == "ham" and p == "ham" for t, p in pairs)

    accuracy = (tp + tn) / len(pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0   # of predicted spam, how many are spam
    recall = tp / (tp + fn) if tp + fn else 0.0      # of real spam, how many we caught
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def print_evaluation(metrics, baseline):
    m = metrics
    print("Confusion matrix (rows = true, columns = predicted):")
    print(f"              {'ham':>6}{'spam':>6}")
    print(f"  true ham    {m['tn']:>6}{m['fp']:>6}")
    print(f"  true spam   {m['fn']:>6}{m['tp']:>6}")
    print()
    print(f"Accuracy:  {m['accuracy']:.2%}   (baseline 'always ham': {baseline:.2%})")
    print(f"Precision: {m['precision']:.2%}   (of messages flagged as spam, really spam)")
    print(f"Recall:    {m['recall']:.2%}   (of real spam, caught)")
    print(f"F1:        {m['f1']:.2%}")


def cross_validate(train_df, epsilon):
    """Mean accuracy of 5-fold stratified cross-validation on the TRAINING set."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []
    for fit_idx, val_idx in folds.split(train_df, train_df["label"]):
        fit_part, val_part = train_df.iloc[fit_idx], train_df.iloc[val_idx]
        model = MarkovBayesClassifier(epsilon).fit(fit_part)
        predictions = model.predict_many(val_part["tokens"])
        accuracies.append(evaluate(val_part["label"], predictions)["accuracy"])
    return sum(accuracies) / len(accuracies)


def compare_epsilons(train_df, test_df):
    """
    Choose epsilon with cross-validation on the training set only.
    Test results are printed for every epsilon for comparison, but the
    choice is made by CV accuracy so the test set stays unseen.
    """
    print(f"{'epsilon':>8}{'CV acc':>9}{'test acc':>10}{'precision':>11}"
          f"{'recall':>8}{'F1':>8}{'FP':>5}{'FN':>5}")
    results = {}
    for epsilon in EPSILON_CANDIDATES:
        cv_accuracy = cross_validate(train_df, epsilon)
        model = MarkovBayesClassifier(epsilon).fit(train_df)
        m = evaluate(test_df["label"], model.predict_many(test_df["tokens"]))
        results[epsilon] = cv_accuracy
        print(f"{epsilon:>8}{cv_accuracy:>9.2%}{m['accuracy']:>10.2%}{m['precision']:>11.2%}"
              f"{m['recall']:>8.2%}{m['f1']:>8.2%}{m['fp']:>5}{m['fn']:>5}")

    best = max(results, key=results.get)
    print(f"\nBest epsilon by cross-validation: {best}")
    return best


def show_errors(test_df, predictions, limit=5):
    """Print a few misclassified test messages."""
    test_df = test_df.assign(predicted=predictions)
    for kind, true_label in (("Missed spam (FN)", "spam"), ("False alarms (FP)", "ham")):
        errors = test_df[(test_df["label"] == true_label) & (test_df["predicted"] != true_label)]
        print(f"{kind}: {len(errors)}")
        for message in errors["message"].head(limit):
            print(f"  {message!r}")


def main():
    # Step 1: load
    df = load_data(DATA_PATH)
    describe_data(df)
    print("\n" + "=" * 70 + "\n")

    # Step 2: preprocess
    df["tokens"] = df["message"].apply(preprocess)
    describe_tokens(df)
    print("\n" + "=" * 70 + "\n")

    # Remove duplicate messages BEFORE splitting, so the same text
    # can never end up in both train and test (data leakage).
    df = remove_duplicates(df)
    print("\n" + "=" * 70 + "\n")

    # Step 3: split
    train_df, test_df = split_data(df)
    describe_split(train_df, test_df)
    print("\n" + "=" * 70 + "\n")

    # Step 4: train
    model = MarkovBayesClassifier(epsilon=EPSILON).fit(train_df)
    describe_model(model)
    print("\n" + "=" * 70 + "\n")

    # Step 5: predict
    predictions = describe_predictions(model, test_df)
    print("\n" + "=" * 70 + "\n")

    # Step 6: evaluate
    baseline = (test_df["label"] == "ham").mean()
    print(f"--- Model with epsilon = {EPSILON} ---")
    print_evaluation(evaluate(test_df["label"], predictions), baseline)
    print()
    show_errors(test_df, predictions)
    print()

    print("--- Epsilon comparison ---")
    best_epsilon = compare_epsilons(train_df, test_df)
    print()

    print(f"--- Final model with epsilon = {best_epsilon} ---")
    best_model = MarkovBayesClassifier(best_epsilon).fit(train_df)
    best_predictions = best_model.predict_many(test_df["tokens"])
    print_evaluation(evaluate(test_df["label"], best_predictions), baseline)
    print()
    show_errors(test_df, best_predictions)


if __name__ == "__main__":
    main()
