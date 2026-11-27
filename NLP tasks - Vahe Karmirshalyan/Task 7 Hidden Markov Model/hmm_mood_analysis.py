"""Hidden Markov Model for student mood analysis.

Learns the initial, transition and emission probabilities from labeled data
(mood + shirt color per day), then finds the most likely mood sequence for a
new shirt color sequence by brute force and by the Viterbi algorithm.
"""

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

# Constants
DATA_PATH = Path(__file__).resolve().parent / "student_data.csv"
DATA_ENCODING = "utf-8"
STATES = ("H", "S")
OBSERVATIONS = ("R", "G", "B")
STATE_NAMES = {"H": "Happy", "S": "Sad"}
TEST_SEQUENCE = ("R", "B", "G")
ROW_SUM_TOLERANCE = 1e-9
PROBABILITY_DIGITS = 4
SCORE_DIGITS = 6
CELL_WIDTH = 16
SEPARATOR_WIDTH = 72

Distribution = dict[str, float]
Counts = dict[str, int]


@dataclass
class EventCounts:
    """Raw counts collected from the training sequences."""

    initial: Counts
    transition: dict[str, Counts]
    emission: dict[str, Counts]


@dataclass
class HiddenMarkovModel:
    """The three probability tables of an HMM: pi, A and B."""

    initial: Distribution
    transition: dict[str, Distribution]
    emission: dict[str, Distribution]


@dataclass
class ScoredSequence:
    """One candidate hidden sequence with the factors of its joint probability."""

    states: tuple[str, ...]
    factors: list[float]
    score: float


@dataclass
class ViterbiResult:
    """Best path plus the day-by-day best scores and backpointers."""

    path: tuple[str, ...]
    score: float
    best_scores: list[Distribution]
    backpointers: list[dict[str, str]]


# Lecture examples, used to verify the decoding code before it runs on our data.
LECTURE_MOOD_MODEL = HiddenMarkovModel(
    initial={"H": 0.4, "S": 0.6},
    transition={"H": {"H": 0.7, "S": 0.3}, "S": {"H": 0.5, "S": 0.5}},
    emission={"H": {"R": 0.8, "G": 0.1, "B": 0.1}, "S": {"R": 0.2, "G": 0.3, "B": 0.5}},
)
LECTURE_MOOD_COLORS = ("G", "B", "R")
LECTURE_MOOD_BEST = (("S", "S", "H"), 0.018)

LECTURE_DOCTOR_MODEL = HiddenMarkovModel(
    initial={"Healthy": 0.6, "Fever": 0.4},
    transition={"Healthy": {"Healthy": 0.7, "Fever": 0.3}, "Fever": {"Healthy": 0.4, "Fever": 0.6}},
    emission={
        "Healthy": {"Normal": 0.5, "Cold": 0.4, "Dizzy": 0.1},
        "Fever": {"Normal": 0.1, "Cold": 0.3, "Dizzy": 0.6},
    },
)
LECTURE_DOCTOR_ANSWERS = ("Normal", "Cold", "Dizzy")
LECTURE_DOCTOR_BEST = (("Healthy", "Healthy", "Fever"), 0.01512)


# Step 1: Loading
def read_rows(path: Path) -> list[dict[str, str]]:
    """Return the CSV rows as dictionaries keyed by column name."""
    try:
        with path.open(encoding=DATA_ENCODING, newline="") as data_file:
            return list(csv.DictReader(data_file))
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit(f"Cannot read data file {path}: {error}")


def group_by_student(rows: list[dict[str, str]]) -> list[list[tuple[str, str]]]:
    """Return one (mood, color) sequence per student, ordered by day."""
    days_by_student = defaultdict(list)
    # Line 1 of the file is the header, so data starts at line 2.
    for line_number, row in enumerate(rows, start=2):
        try:
            student = row["StudentID"].strip()
            day = int(row["Day"])
            mood = row["Mood"].strip()
            color = row["ShirtColor"].strip()
        except (KeyError, ValueError, AttributeError):
            raise SystemExit(f"Malformed row at line {line_number}: {row}")

        if mood not in STATES or color not in OBSERVATIONS:
            raise SystemExit(f"Unknown mood or color at line {line_number}: {row}")
        days_by_student[student].append((day, mood, color))

    if not days_by_student:
        raise SystemExit("The data file has no rows.")

    # Each student is a separate chain: no transition links two different students.
    return [
        [(mood, color) for _, mood, color in sorted(days)]
        for days in days_by_student.values()
    ]


# Step 2: Counting
def count_events(sequences: list[list[tuple[str, str]]]) -> EventCounts:
    """Count first-day moods, (mood -> next mood) pairs and (mood, color) pairs."""
    initial_counts = defaultdict(int)
    transition_counts = {state: defaultdict(int) for state in STATES}
    emission_counts = {state: defaultdict(int) for state in STATES}

    for sequence in sequences:
        moods = [mood for mood, _ in sequence]

        initial_counts[moods[0]] += 1
        # zip pairs each day with the next one; the last day has no next day.
        for previous_mood, current_mood in zip(moods, moods[1:]):
            transition_counts[previous_mood][current_mood] += 1
        for mood, color in sequence:
            emission_counts[mood][color] += 1

    return EventCounts(initial_counts, transition_counts, emission_counts)


# Step 3: Normalizing
def normalize(counts: Counts, outcomes: tuple[str, ...], label: str) -> Distribution:
    """Turn counts into probabilities: P(outcome) = count(outcome) / row_total."""
    total = sum(counts.values())
    if total == 0:
        raise SystemExit(f"No data to estimate {label}.")
    return {outcome: counts.get(outcome, 0) / total for outcome in outcomes}


def build_model(counts: EventCounts) -> HiddenMarkovModel:
    """Estimate pi, A and B from the counts (maximum likelihood, no smoothing)."""
    initial = normalize(counts.initial, STATES, "the initial distribution")
    transition = {
        state: normalize(counts.transition[state], STATES, f"transitions from {state}")
        for state in STATES
    }
    emission = {
        state: normalize(counts.emission[state], OBSERVATIONS, f"emissions of {state}")
        for state in STATES
    }
    return HiddenMarkovModel(initial, transition, emission)


def check_rows_sum_to_one(model: HiddenMarkovModel) -> None:
    """Every row of pi, A and B is a probability distribution, so it must sum to 1."""
    rows = {"pi": model.initial}
    rows.update({f"A[{state}]": row for state, row in model.transition.items()})
    rows.update({f"B[{state}]": row for state, row in model.emission.items()})

    for name, row in rows.items():
        if abs(sum(row.values()) - 1) > ROW_SUM_TOLERANCE:
            raise ValueError(f"Row {name} sums to {sum(row.values())}, expected 1.")


# Step 4: Brute-force decoding
def score_sequence(
    model: HiddenMarkovModel, states: tuple[str, ...], observations: tuple[str, ...]
) -> ScoredSequence:
    """Joint probability P(states, observations) = P(M1)P(C1|M1) * prod P(Mt|Mt-1)P(Ct|Mt)."""
    factors = [model.initial[states[0]], model.emission[states[0]][observations[0]]]
    for day in range(1, len(observations)):
        factors.append(model.transition[states[day - 1]][states[day]])
        factors.append(model.emission[states[day]][observations[day]])
    return ScoredSequence(states, factors, math.prod(factors))


def brute_force_decode(
    model: HiddenMarkovModel, observations: tuple[str, ...]
) -> list[ScoredSequence]:
    """Score every possible hidden sequence (|S|^T of them)."""
    states = tuple(model.initial)
    return [
        score_sequence(model, candidate, observations)
        for candidate in product(states, repeat=len(observations))
    ]


# Step 5: Viterbi decoding
def viterbi_decode(model: HiddenMarkovModel, observations: tuple[str, ...]) -> ViterbiResult:
    """Keep only the best route into each state per day, then backtrack."""
    states = tuple(model.initial)

    # Day 1 has no previous day, so the start probability replaces the transition.
    best_scores = [
        {state: model.initial[state] * model.emission[state][observations[0]] for state in states}
    ]
    backpointers = [{}]

    for day in range(1, len(observations)):
        previous_scores = best_scores[-1]
        scores_today = {}
        pointers_today = {}
        for state in states:
            best_previous = max(
                states, key=lambda previous: previous_scores[previous] * model.transition[previous][state]
            )
            scores_today[state] = (
                previous_scores[best_previous]
                * model.transition[best_previous][state]
                * model.emission[state][observations[day]]
            )
            pointers_today[state] = best_previous
        best_scores.append(scores_today)
        backpointers.append(pointers_today)

    # Backtracking: start from the best final state and follow the pointers backwards.
    last_state = max(states, key=lambda state: best_scores[-1][state])
    path = [last_state]
    for day in range(len(observations) - 1, 0, -1):
        path.append(backpointers[day][path[-1]])
    path.reverse()

    return ViterbiResult(tuple(path), best_scores[-1][last_state], best_scores, backpointers)


# Verification
def run_lecture_checks() -> None:
    """Reproduce the lecture's answers; stop if the decoding code disagrees."""
    expected_path, expected_score = LECTURE_MOOD_BEST
    scored = brute_force_decode(LECTURE_MOOD_MODEL, LECTURE_MOOD_COLORS)
    best = max(scored, key=lambda candidate: candidate.score)
    viterbi = viterbi_decode(LECTURE_MOOD_MODEL, LECTURE_MOOD_COLORS)
    for method, path, score in (("brute force", best.states, best.score), ("Viterbi", viterbi.path, viterbi.score)):
        if path != expected_path or not math.isclose(score, expected_score):
            raise ValueError(f"Lecture mood check failed ({method}): got {path} = {score}")

    expected_path, expected_score = LECTURE_DOCTOR_BEST
    viterbi = viterbi_decode(LECTURE_DOCTOR_MODEL, LECTURE_DOCTOR_ANSWERS)
    if viterbi.path != expected_path or not math.isclose(viterbi.score, expected_score):
        raise ValueError(f"Lecture doctor check failed: got {viterbi.path} = {viterbi.score}")

    print("Lecture checks passed: mood example (S,S,H = 0.018) and doctor example (H,H,F = 0.01512).")


# Report
def print_section(title: str) -> None:
    print("\n" + "=" * SEPARATOR_WIDTH)
    print(title)
    print("=" * SEPARATOR_WIDTH)


def format_fraction(count: int, total: int) -> str:
    return f"{count}/{total} = {count / total:.{PROBABILITY_DIGITS}f}"


def print_table(row_label: str, columns: tuple[str, ...], rows: dict[str, Counts]) -> None:
    """Print each row's counts as 'count/row_total = probability'."""
    print(f"{row_label:<8}" + "".join(f"{column:>{CELL_WIDTH}}" for column in columns))
    for name, counts in rows.items():
        total = sum(counts.values())
        cells = "".join(f"{format_fraction(counts.get(column, 0), total):>{CELL_WIDTH}}" for column in columns)
        print(f"{name:<8}{cells}")


def print_parameters(counts: EventCounts) -> None:
    print_section("PART 1: PARAMETER LEARNING")

    print("\nInitial distribution  P(M1) = Count(M1 on day 1) / number of students")
    print_table("", STATES, {"start": counts.initial})

    print("\nTransition matrix  P(M2|M1) = Count(M1 -> M2) / Count(M1 with a next day)")
    print_table("from\\to", STATES, counts.transition)

    print("\nEmission matrix  P(C|M) = Count(C, M) / Count(M)")
    print_table("mood", OBSERVATIONS, counts.emission)


def format_states(states: tuple[str, ...]) -> str:
    return ",".join(states)


def print_brute_force(scored: list[ScoredSequence], best: ScoredSequence, observations: tuple[str, ...]) -> None:
    print_section(f"PART 2: BRUTE FORCE FOR OBSERVATIONS {format_states(observations)}")
    print("P(M1)*P(C1|M1) * P(M2|M1)*P(C2|M2) * P(M3|M2)*P(C3|M3)\n")
    for candidate in scored:
        product_text = " x ".join(f"{factor:.{PROBABILITY_DIGITS}f}" for factor in candidate.factors)
        marker = "  <- best" if candidate is best else ""
        print(f"{format_states(candidate.states)}:  {product_text} = {candidate.score:.{SCORE_DIGITS}f}{marker}")

    mood_names = " -> ".join(STATE_NAMES[state] for state in best.states)
    print(f"\nargmax P(M1,M2,M3,{format_states(observations)}) = {format_states(best.states)} ({mood_names})")
    print(f"Probability = {best.score:.{SCORE_DIGITS}f}")


def print_viterbi(result: ViterbiResult, observations: tuple[str, ...]) -> None:
    print_section("BONUS: VITERBI ALGORITHM")
    for day, (color, scores) in enumerate(zip(observations, result.best_scores), start=1):
        cells = []
        for state, score in scores.items():
            source = "start" if day == 1 else f"from {result.backpointers[day - 1][state]}"
            cells.append(f"{state} = {score:.{SCORE_DIGITS}f} ({source})")
        print(f"Day {day} ({color}):  " + "   ".join(cells))
    print(f"\nViterbi path = {format_states(result.path)}, probability = {result.score:.{SCORE_DIGITS}f}")


def main() -> None:
    run_lecture_checks()

    sequences = group_by_student(read_rows(DATA_PATH))
    counts = count_events(sequences)
    model = build_model(counts)
    check_rows_sum_to_one(model)
    print_parameters(counts)

    scored = brute_force_decode(model, TEST_SEQUENCE)
    best = max(scored, key=lambda candidate: candidate.score)
    print_brute_force(scored, best, TEST_SEQUENCE)

    viterbi = viterbi_decode(model, TEST_SEQUENCE)
    print_viterbi(viterbi, TEST_SEQUENCE)
    agreement = "agrees" if viterbi.path == best.states else "DISAGREES"
    print(f"Viterbi {agreement} with brute force.")


if __name__ == "__main__":
    main()
