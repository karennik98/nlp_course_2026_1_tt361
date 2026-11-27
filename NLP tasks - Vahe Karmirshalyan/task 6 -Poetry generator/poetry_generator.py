"""Markov chain poetry generator trained on Robert Frost's poems."""

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Constants
CORPUS_PATH = Path(__file__).resolve().parent / "robert_frost.txt"
CORPUS_ENCODING = "utf-8"
END_TOKEN = "END"
LINES_TO_GENERATE = 4
# Guards against endless lines, e.g. a deterministic cycle that never reaches END.
MAX_WORDS_PER_LINE = 25
# Set to an integer to get the same random poem on every run.
RANDOM_SEED = None
TOP_WORDS_TO_SHOW = 5
SEPARATOR_WIDTH = 60

# Words are runs of letters; an apostrophe is kept only inside a word (sound's, can't).
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")

Distribution = dict[str, float]


@dataclass
class MarkovModel:
    """The three probability tables learned from the corpus."""

    initial: Distribution
    first_order: dict[str, Distribution]
    second_order: dict[tuple[str, str], Distribution]


# File reading
def read_corpus_lines(path: Path) -> list[str]:
    """Return the raw lines of the corpus file."""
    try:
        with path.open(encoding=CORPUS_ENCODING) as corpus_file:
            return corpus_file.readlines()
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit(f"Cannot read corpus file {path}: {error}")


# Text processing
def tokenize(line: str) -> list[str]:
    """Lowercase a line, drop punctuation and return its words."""
    return WORD_PATTERN.findall(line.lower())


def tokenize_lines(raw_lines: list[str]) -> list[list[str]]:
    """Tokenize every line and skip the ones without words (stanza breaks)."""
    tokenized_lines = [tokenize(line) for line in raw_lines]
    return [words for words in tokenized_lines if words]


# Training
def count_transitions(lines: list[list[str]]) -> tuple[dict, dict, dict]:
    """Count first words, (w1 -> w2) pairs and (w[t-2], w[t-1] -> w[t]) triples."""
    initial_counts = defaultdict(int)
    first_order_counts = defaultdict(lambda: defaultdict(int))
    second_order_counts = defaultdict(lambda: defaultdict(int))

    for words in lines:
        # END closes every line, so the model learns where lines stop.
        sequence = words + [END_TOKEN]

        initial_counts[sequence[0]] += 1
        # First-order is used only for the second word of a line, so it is trained on that position.
        first_order_counts[sequence[0]][sequence[1]] += 1
        for index in range(2, len(sequence)):
            previous_pair = (sequence[index - 2], sequence[index - 1])
            second_order_counts[previous_pair][sequence[index]] += 1

    return initial_counts, first_order_counts, second_order_counts


def normalize(counts: dict[str, int]) -> Distribution:
    """Turn counts into probabilities: P(word) = count(word) / total_count."""
    total_count = sum(counts.values())
    return {word: count / total_count for word, count in counts.items()}


def train_model(lines: list[list[str]]) -> MarkovModel:
    """Build the three probability tables from tokenized lines."""
    if not lines:
        raise SystemExit("The corpus has no lines with words")

    initial_counts, first_order_counts, second_order_counts = count_transitions(lines)

    return MarkovModel(
        initial=normalize(initial_counts),
        first_order={word: normalize(next_counts) for word, next_counts in first_order_counts.items()},
        second_order={pair: normalize(next_counts) for pair, next_counts in second_order_counts.items()},
    )


# Word selection
def choose(distribution: Distribution, deterministic: bool, rng: random.Random) -> str:
    """Pick the most probable word, or a random one by cumulative probability."""
    if deterministic:
        return max(distribution, key=distribution.get)

    random_value = rng.random()
    cumulative_probability = 0.0
    for word, probability in distribution.items():
        cumulative_probability += probability
        if random_value < cumulative_probability:
            return word

    # Floating-point sums can end slightly below 1.0, so the last word covers the rest.
    return word


# Generation
def generate_line(model: MarkovModel, deterministic: bool, rng: random.Random) -> str:
    """Generate one line: w1 from initial, w2 from first-order, the rest from second-order."""
    first_word = choose(model.initial, deterministic, rng)
    second_word = choose(model.first_order[first_word], deterministic, rng)
    words = [first_word]

    previous_word, current_word = first_word, second_word
    while current_word != END_TOKEN and len(words) < MAX_WORDS_PER_LINE:
        words.append(current_word)
        next_word = choose(model.second_order[(previous_word, current_word)], deterministic, rng)
        previous_word, current_word = current_word, next_word

    return " ".join(words)


def generate_poem(model: MarkovModel, deterministic: bool, rng: random.Random, line_count: int) -> list[str]:
    """Generate a poem of line_count lines."""
    return [generate_line(model, deterministic, rng) for _ in range(line_count)]


# Output
def format_top(distribution: Distribution) -> str:
    """Format the most probable entries of a distribution as 'word 0.123'."""
    top_entries = sorted(distribution.items(), key=lambda entry: entry[1], reverse=True)[:TOP_WORDS_TO_SHOW]
    return ", ".join(f"{word} {probability:.3f}" for word, probability in top_entries)


def print_model_summary(model: MarkovModel, line_count: int) -> None:
    """Print table sizes and a few example probabilities."""
    print(f"Training lines:            {line_count}")
    print(f"Initial words:             {len(model.initial)}")
    print(f"First-order states (w1):   {len(model.first_order)}")
    print(f"Second-order states (w,w): {len(model.second_order)}")
    print()
    print(f"Top P(w1):             {format_top(model.initial)}")
    if "and" in model.first_order:
        print(f"Top P(w2 | 'and'):     {format_top(model.first_order['and'])}")
    if ("in", "a") in model.second_order:
        print(f"Top P(w | 'in', 'a'):  {format_top(model.second_order[('in', 'a')])}")


def print_poem(title: str, poem_lines: list[str]) -> None:
    """Print a titled poem."""
    print("=" * SEPARATOR_WIDTH)
    print(title)
    print("=" * SEPARATOR_WIDTH)
    for line in poem_lines:
        print(line)
    print()


def main() -> None:
    """Train the model on the corpus and print generated poems."""
    lines = tokenize_lines(read_corpus_lines(CORPUS_PATH))
    model = train_model(lines)

    rng = random.Random(RANDOM_SEED)
    print_model_summary(model, len(lines))
    print()
    print_poem("Cumulative probability method", generate_poem(model, False, rng, LINES_TO_GENERATE))
    print_poem("Deterministic method", generate_poem(model, True, rng, LINES_TO_GENERATE))


if __name__ == "__main__":
    main()
