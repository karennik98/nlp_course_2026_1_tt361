"""Markov Chain Poetry Generator.

Learns word patterns from Robert Frost's poems (robert_frost.txt) and
generates new lines with a second-order Markov chain:

    w1 ~ P(w1)                  initial word probabilities
    w2 ~ P(w2 | w1)             first-order transitions
    wt ~ P(wt | wt-2, wt-1)     second-order transitions, until 'END'
"""

import random
import string
import sys

FILE_NAME = "robert_frost.txt"
END = "END"
NUM_LINES = 4

# Dashes/hyphens become spaces (so "soul-and-body" -> "soul and body"),
# every other punctuation mark is simply deleted.
_DASHES = "-–—"
_PUNCT = string.punctuation + "‘’“”…"
_TABLE = str.maketrans(_DASHES, " " * len(_DASHES), "".join(c for c in _PUNCT if c not in _DASHES))


def read_lines(file_name):
    """Read the poem line by line; exit with a clear message on file errors."""
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            return f.readlines()
    except FileNotFoundError:
        sys.exit(f"Error: file '{file_name}' not found.")
    except OSError as e:
        sys.exit(f"Error: could not read '{file_name}': {e}")


def tokenize(line):
    """Remove punctuation, convert to lowercase, split into words."""
    return line.translate(_TABLE).lower().split()


def add_count(table, key, word):
    """table[key][word] += 1"""
    table.setdefault(key, {})
    table[key][word] = table[key].get(word, 0) + 1


def normalize(counts):
    """Turn {word: count} into {word: probability}: P(word) = count / total."""
    total = sum(counts.values())
    return {word: count / total for word, count in counts.items()}


def train(lines):
    """Build the three probability dictionaries.

    initial      : {w1: P(w1)}                      count(w1 first) / number of lines
    first_order  : {w_prev: {w: P(w | w_prev)}}     count(w_prev, w) / count(w_prev)
    second_order : {(w_a, w_b): {w: P(w | w_a, w_b)}}
                                                    count(w_a, w_b, w) / count(w_a, w_b)
    Every line is terminated by the 'END' token.
    """
    initial = {}
    first_order = {}
    second_order = {}

    for line in lines:
        words = tokenize(line)
        if not words:  # blank line between stanzas
            continue
        words.append(END)

        initial[words[0]] = initial.get(words[0], 0) + 1

        for i in range(1, len(words)):
            add_count(first_order, words[i - 1], words[i])
            if i >= 2:
                add_count(second_order, (words[i - 2], words[i - 1]), words[i])

    initial = normalize(initial)
    first_order = {k: normalize(v) for k, v in first_order.items()}
    second_order = {k: normalize(v) for k, v in second_order.items()}
    return initial, first_order, second_order


def sample_word(probabilities):
    """Cumulative probability method.

    Lay the words on the number line [0, 1), draw r ~ U(0, 1) and return
    the first word whose cumulative probability exceeds r.
    """
    r = random.random()
    cumulative = 0.0
    word = None
    for word, p in probabilities.items():
        cumulative += p
        if r < cumulative:
            return word
    return word  # guards against floating point round-off (sum = 0.9999...)


def generate_line(initial, first_order, second_order):
    """Generate one line of poetry."""
    w1 = sample_word(initial)
    line = [w1]

    w2 = sample_word(first_order[w1])
    if w2 == END:
        return w1
    line.append(w2)

    while True:
        w = sample_word(second_order[(line[-2], line[-1])])
        if w == END:
            break
        line.append(w)
    return " ".join(line)


def main():
    lines = read_lines(FILE_NAME)
    initial, first_order, second_order = train(lines)
    if not initial:
        sys.exit(f"Error: '{FILE_NAME}' contains no text.")

    for _ in range(NUM_LINES):
        print(generate_line(initial, first_order, second_order))


if __name__ == "__main__":
    main()
