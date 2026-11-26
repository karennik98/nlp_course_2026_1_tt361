
import csv
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

BASE_DIR = Path(__file__).parent

states = ["H", "S"]
colors = ["R", "G", "B"]
observations = ["R", "B", "G"]

students = defaultdict(list)

with open(BASE_DIR / "student_data.csv",
          "r", encoding="utf-8-sig") as file:

    reader = csv.DictReader(file)

    for row in reader:

        student_id = row["StudentID"]
        day = int(row["Day"])
        mood = row["Mood"].strip().upper()
        color = row["ShirtColor"].strip().upper()

        students[student_id].append(
            (day, mood, color)
        )

for records in students.values():
    records.sort(key=lambda x: x[0])

initial_counts = Counter()
transition_counts = defaultdict(Counter)
emission_counts = defaultdict(Counter)

for records in students.values():

    initial_counts[records[0][1]] += 1
    for day, mood, color in records:
        emission_counts[mood][color] += 1
    for i in range(len(records) - 1):

        current_mood = records[i][1]
        next_mood = records[i + 1][1]

        transition_counts[current_mood][next_mood] += 1

initial = {}

for state in states:

    initial[state] = (
        initial_counts[state] / len(students)
    )

transition = {}

for state in states:

    total = sum(transition_counts[state].values())

    transition[state] = {}

    for next_state in states:

        transition[state][next_state] = (
            transition_counts[state][next_state] / total
        )

emission = {}

for state in states:

    total = sum(emission_counts[state].values())

    emission[state] = {}

    for color in colors:

        emission[state][color] = (
            emission_counts[state][color] / total
        )

results = []

results.append("HIDDEN MARKOV MODEL")
results.append("Student Mood Analysis")
results.append("=" * 50)

results.append(
    f"Number of students: {len(students)}"
)

results.append(
    f"Total observations: "
    f"{sum(len(r) for r in students.values())}"
)

results.append("\n1. Initial Probability Distribution")

for state in states:

    results.append(
        f"P({state}) = {initial[state]:.6f}"
    )

results.append("\n2. Transition Matrix")
results.append("From/To       H          S")

for state in states:

    results.append(
        f"{state:<10}"
        f"{transition[state]['H']:.6f}   "
        f"{transition[state]['S']:.6f}"
    )

results.append("\n3. Emission Matrix")
results.append("Mood          R          G          B")

for state in states:

    results.append(
        f"{state:<10}"
        f"{emission[state]['R']:.6f}   "
        f"{emission[state]['G']:.6f}   "
        f"{emission[state]['B']:.6f}"
    )

def calculate_probability(sequence):

    m1, m2, m3 = sequence

    factors = [
        initial[m1],
        emission[m1]["R"],
        transition[m1][m2],
        emission[m2]["B"],
        transition[m2][m3],
        emission[m3]["G"]
    ]

    probability = 1

    for value in factors:
        probability *= value

    return probability, factors

results.append("\n4. Observations: R -> B -> G")
results.append("\n5. All Possible Sequences")

max_probability = -1
best_sequence = None

for sequence in product(states, repeat=3):

    probability, factors = calculate_probability(
        sequence
    )

    sequence_text = " -> ".join(sequence)

    calculation = " * ".join(
        f"{value:.6f}" for value in factors
    )

    results.append(f"\nSequence: {sequence_text}")
    results.append(f"Calculation: {calculation}")
    results.append(f"Probability: {probability:.10f}")

    if probability > max_probability:

        max_probability = probability
        best_sequence = sequence

results.append("\n6. Most Likely Mood Sequence")
results.append("=" * 50)

results.append(" -> ".join(best_sequence))

results.append(
    f"Maximum probability: {max_probability:.10f}"
)

result_text = "\n".join(results)

print(result_text)

with open(BASE_DIR / "results.txt",
          "w", encoding="utf-8") as file:

    file.write(result_text)

print("\nResults saved to results.txt")
