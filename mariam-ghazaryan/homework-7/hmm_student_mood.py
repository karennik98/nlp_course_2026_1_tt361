"""
Homework 7 - Hidden Markov Model: Student Mood Analysis

Hidden states:  H (Happy), S (Sad)
Observations:   R (Red), G (Green), B (Blue) shirt colour

Part 1: learn initial, transition and emission probabilities by counting.
Part 2: score every mood sequence for the observations R, B, G and pick the
        most likely one (brute force), then confirm with the Viterbi algorithm.
"""

import csv
import itertools
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

STATES = ["H", "S"]
COLORS = ["R", "G", "B"]
DATA_FILE = Path(__file__).with_name("student_data.csv")


def load_sequences(path):
    """Return {student_id: [(mood, color), ...]} ordered by day."""
    rows = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[int(r["StudentID"])].append((int(r["Day"]), r["Mood"], r["ShirtColor"]))
    return {sid: [(m, c) for _, m, c in sorted(days)] for sid, days in rows.items()}


def learn_parameters(sequences):
    init_count = defaultdict(int)
    trans_count = defaultdict(int)       # (m1, m2) -> count
    trans_from = defaultdict(int)        # m1 -> number of transitions out of m1
    emit_count = defaultdict(int)        # (m, c) -> count
    state_count = defaultdict(int)       # m -> total days in m

    for seq in sequences.values():
        init_count[seq[0][0]] += 1
        for mood, color in seq:
            emit_count[(mood, color)] += 1
            state_count[mood] += 1
        for (m1, _), (m2, _) in zip(seq, seq[1:]):
            trans_count[(m1, m2)] += 1
            trans_from[m1] += 1

    n_students = len(sequences)
    pi = {m: Fraction(init_count[m], n_students) for m in STATES}
    # Count(M1) for the transition matrix = number of times M1 is followed by a
    # next day (the last day of each student has no successor), so rows sum to 1.
    A = {m1: {m2: Fraction(trans_count[(m1, m2)], trans_from[m1]) for m2 in STATES}
         for m1 in STATES}
    B = {m: {c: Fraction(emit_count[(m, c)], state_count[m]) for c in COLORS}
         for m in STATES}
    counts = dict(init=init_count, trans=trans_count, trans_from=trans_from,
                  emit=emit_count, state=state_count, n=n_students)
    return pi, A, B, counts


def joint_probability(moods, obs, pi, A, B):
    p = pi[moods[0]] * B[moods[0]][obs[0]]
    for t in range(1, len(obs)):
        p *= A[moods[t - 1]][moods[t]] * B[moods[t]][obs[t]]
    return p


def viterbi(obs, pi, A, B):
    delta = [{m: pi[m] * B[m][obs[0]] for m in STATES}]
    back = [{}]
    for t in range(1, len(obs)):
        delta.append({})
        back.append({})
        for m in STATES:
            prev = max(STATES, key=lambda k: delta[t - 1][k] * A[k][m])
            delta[t][m] = delta[t - 1][prev] * A[prev][m] * B[m][obs[t]]
            back[t][m] = prev
    last = max(STATES, key=lambda m: delta[-1][m])
    path = [last]
    for t in range(len(obs) - 1, 0, -1):
        path.insert(0, back[t][path[0]])
    return path, delta[-1][last]


def fmt(num, den):
    return f"{num}/{den} = {num / den:.4f}"


def main():
    sequences = load_sequences(DATA_FILE)
    pi, A, B, c = learn_parameters(sequences)

    print("=" * 64)
    print("PART 1: PARAMETER LEARNING")
    print("=" * 64)
    print(f"Students: {c['n']}, days per student: {len(next(iter(sequences.values())))}")

    print("\n1) Initial probability distribution P(M1)")
    for m in STATES:
        print(f"   P({m}) = {c['init'][m]}/{c['n']} = {float(pi[m]):.4f}")

    print("\n2) Transition matrix P(M2 | M1)   (rows = M1, cols = M2)")
    print("        " + "".join(f"{m:>18}" for m in STATES))
    for m1 in STATES:
        print(f"   {m1}    " + "".join(f"{fmt(c['trans'][(m1, m2)], c['trans_from'][m1]):>18}" for m2 in STATES))

    print("\n3) Emission matrix P(C | M)   (rows = M, cols = C)")
    print("        " + "".join(f"{col:>18}" for col in COLORS))
    for m in STATES:
        print(f"   {m}    " + "".join(f"{fmt(c['emit'][(m, col)], c['state'][m]):>18}" for col in COLORS))

    obs = ["R", "B", "G"]
    print("\n" + "=" * 64)
    print(f"PART 2: MOST LIKELY MOOD SEQUENCE FOR OBSERVATIONS {', '.join(obs)}")
    print("=" * 64)
    results = []
    for moods in itertools.product(STATES, repeat=len(obs)):
        p = joint_probability(moods, obs, pi, A, B)
        results.append((moods, p))
        terms = [f"P({moods[0]})", f"P({obs[0]}|{moods[0]})"]
        for t in range(1, len(obs)):
            terms += [f"P({moods[t]}|{moods[t-1]})", f"P({obs[t]}|{moods[t]})"]
        values = [pi[moods[0]], B[moods[0]][obs[0]]]
        for t in range(1, len(obs)):
            values += [A[moods[t - 1]][moods[t]], B[moods[t]][obs[t]]]
        print(f"\n{{{','.join(moods)}}}: " + " x ".join(terms))
        print("   = " + " x ".join(f"{float(v):.4f}" for v in values))
        print(f"   = {float(p):.6f}")

    best, best_p = max(results, key=lambda r: r[1])
    print("\nRanking:")
    for moods, p in sorted(results, key=lambda r: r[1], reverse=True):
        mark = "  <-- argmax" if moods == best else ""
        print(f"   {{{','.join(moods)}}}  {float(p):.6f}{mark}")

    print(f"\nMost likely sequence: {{{','.join(best)}}} with P = {float(best_p):.6f}")
    vpath, vp = viterbi(obs, pi, A, B)
    print(f"Viterbi check:        {{{','.join(vpath)}}} with P = {float(vp):.6f}")


if __name__ == "__main__":
    main()
