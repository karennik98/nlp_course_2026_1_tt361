# Homework 7: Hidden Markov Model, Student Mood Analysis

- **Hidden states (mood):** Happy (H), Sad (S)
- **Observations (shirt colour):** Red (R), Green (G), Blue (B)
- **Data:** `student_data.csv`, which has 5 students × 20 days = 100 observations

Run it (standard library only):

```bash
python3 hmm_student_mood.py
```

The full printed output is in `output.txt`.

## Part 1: Parameter learning

### 1. Initial probability distribution P(M₁)

Moods on day 1: student 1 = H, 2 = H, 3 = S, 4 = H, 5 = S

| M₁ | Count | P(M₁) |
|----|-------|-------|
| H  | 3     | 3/5 = **0.600** |
| S  | 2     | 2/5 = **0.400** |

### 2. Transition matrix P(M₂ | M₁)

There are 19 day-to-day transitions per student, so 95 in total. Count(M₁) is the number of times M₁ is followed by another day, which makes each row sum to 1.

| M₁ \ M₂ | H | S | Total |
|---------|---|---|-------|
| **H** | 36/55 = **0.6545** | 19/55 = **0.3455** | 55 |
| **S** | 18/40 = **0.4500** | 22/40 = **0.5500** | 40 |

### 3. Emission matrix P(C | M)

Over all 100 days, the student was Happy on 57 and Sad on 43.

| M \ C | R | G | B |
|-------|---|---|---|
| **H** | 41/57 = **0.7193** | 16/57 = **0.2807** | 0/57 = **0.0000** |
| **S** | 0/43 = **0.0000** | 6/43 = **0.1395** | 37/43 = **0.8605** |

In this data, red is only ever worn when Happy and blue only when Sad. Green appears with both moods.

## Part 2: Testing the model on the observations R, B, G

P(M₁,M₂,M₃,R,B,G) = P(M₁)·P(R|M₁)·P(M₂|M₁)·P(B|M₂)·P(M₃|M₂)·P(G|M₃)

| Sequence | Calculation | Probability |
|----------|-------------|-------------|
| {H,H,H} | 0.6 × 0.7193 × 0.6545 × **0** × 0.6545 × 0.2807 | 0 |
| {H,H,S} | 0.6 × 0.7193 × 0.6545 × **0** × 0.3455 × 0.1395 | 0 |
| **{H,S,H}** | 0.6 × 0.7193 × 0.3455 × 0.8605 × 0.4500 × 0.2807 | **0.016205** |
| {H,S,S} | 0.6 × 0.7193 × 0.3455 × 0.8605 × 0.5500 × 0.1395 | 0.009845 |
| {S,H,H} | 0.4 × **0** × 0.4500 × 0 × 0.6545 × 0.2807 | 0 |
| {S,H,S} | 0.4 × **0** × 0.4500 × 0 × 0.3455 × 0.1395 | 0 |
| {S,S,H} | 0.4 × **0** × 0.5500 × 0.8605 × 0.4500 × 0.2807 | 0 |
| {S,S,S} | 0.4 × **0** × 0.5500 × 0.8605 × 0.5500 × 0.1395 | 0 |

Six of the eight sequences have probability 0 for one of two reasons:
- they start with S, but P(R|S) = 0
- they have H on day 2, but P(B|H) = 0

## Result

**argmax P(M₁,M₂,M₃,R,B,G) = {H, S, H}**, with P ≈ **0.0162**

The most likely moods for the shirts Red, Blue, Green are **Happy, Sad, Happy**. The Viterbi algorithm in the script gives the same path and probability.

{H,S,H} beats the runner-up {H,S,S} on the last step only. Going from S to H and then emitting green gives 0.45 × 0.2807 = 0.126. Staying in S and emitting green gives 0.55 × 0.1395 = 0.077.

> **Note on the denominator:** the transition counts could also be divided by the *total* number of days in each mood (57 for H, 43 for S). In that case the rows no longer sum to 1, because the last day of each student has no next day. Even so, {H,S,H} is still the best sequence, with P ≈ 0.0145.
