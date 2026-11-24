import random
import re
from collections import defaultdict, Counter
from pathlib import Path


def read_poem(filename):
    lines = []

    try:
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                line = line.lower()
                line = re.sub(r"[^\w\s]", "", line)
                words = line.split()

                if words:
                    lines.append(words)

    except OSError as error:
        print("File error:", error)

    return lines


def calculate_probabilities(counts):
    total = sum(counts.values())

    return {
        word: count / total
        for word, count in counts.items()
    }


def train(lines):
    initial_counts = Counter()
    first_order_counts = defaultdict(Counter)
    second_order_counts = defaultdict(Counter)

    for line in lines:

        initial_counts[line[0]] += 1

        words = line + ["END"]

        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]

            first_order_counts[current_word][next_word] += 1

        for i in range(len(words) - 2):
            pair = (words[i], words[i + 1])
            next_word = words[i + 2]

            second_order_counts[pair][next_word] += 1

    initial_probs = calculate_probabilities(initial_counts)

    first_probs = {
        word: calculate_probabilities(counts)
        for word, counts in first_order_counts.items()
    }

    second_probs = {
        pair: calculate_probabilities(counts)
        for pair, counts in second_order_counts.items()
    }

    return initial_probs, first_probs, second_probs


def select_word(probabilities):
    r = random.random()
    cumulative = 0

    for word, probability in probabilities.items():
        cumulative += probability

        if r < cumulative:
            return word

    return list(probabilities.keys())[-1]


def generate_line(initial_probs, first_probs, second_probs):

    first_word = select_word(initial_probs)
    line = [first_word]
    second_word = select_word(first_probs[first_word])

    if second_word == "END":
        return " ".join(line).capitalize()

    line.append(second_word)

    while len(line) < 30:

        pair = (line[-2], line[-1])

        if pair not in second_probs:
            break

        next_word = select_word(second_probs[pair])

        if next_word == "END":
            break

        line.append(next_word)

    return " ".join(line).capitalize()


def main():
    filename = Path(__file__).parent / "robert_frost.txt"

    lines = read_poem(filename)

    if not lines:
        print("No training data found.")
        return

    initial_probs, first_probs, second_probs = train(lines)

    print("Training completed successfully!")
    print("Number of training lines:", len(lines))

    print("\nGenerated Poetry:\n")

    for _ in range(4):
        print(generate_line(
            initial_probs,
            first_probs,
            second_probs
        ))


if __name__ == "__main__":
    main()
