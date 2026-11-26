import csv
from itertools import product

moods = ["H", "S"]
colors = ["R", "G", "B"]

#Step 1:Reads the data and groups it by student
students = {}
try:
    with open("student_data.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["StudentID"]
            day = int(row["Day"])
            if sid not in students:
                students[sid] = []
            students[sid].append((day, row["Mood"], row["ShirtColor"]))
except FileNotFoundError:
    print("Error: file student_data.csv not found!")
    exit()


for sid in students:#Sorts each student's records by day
    students[sid].sort()

#Step 2:Count
initial_count = {m: 0 for m in moods}#Count(M1 on first day)
trans_count = {m1: {m2: 0 for m2 in moods} for m1 in moods}#Count(M1 -> M2)
emis_count = {m: {c: 0 for c in colors} for m in moods}#Count(C, M)
mood_count = {m: 0 for m in moods}#Count(M) for emission

for sid in students:
    records = students[sid]
    initial_count[records[0][1]] += 1#mood on the first day
    for i in range(len(records)):
        mood = records[i][1]
        color = records[i][2]
        emis_count[mood][color] += 1
        mood_count[mood] += 1
        if i < len(records) - 1:#transition to the next day
            next_mood = records[i + 1][1]
            trans_count[mood][next_mood] += 1

#Step 3:Parameter learning
total_students = len(students)

initial = {}#P(M1) = Count(M1 on first day)/Total students
for m in moods:
    initial[m] = initial_count[m] / total_students

transition = {}#P(M2|M1) = Count(M1 -> M2)/Count(M1)
for m1 in moods:
    total = sum(trans_count[m1].values()) 
    transition[m1] = {}
    for m2 in moods:
        transition[m1][m2] = trans_count[m1][m2] / total

emission = {}#P(C|M) = Count(C, M)/Count(M)
for m in moods:
    emission[m] = {}
    for c in colors:
        emission[m][c] = emis_count[m][c] / mood_count[m]

print("Initial Probability Distribution:")
for m in moods:
    print(f"  P({m}) = {initial[m]:.4f}")

print("\nTransition Matrix P(M2|M1):")
print("        H       S")
for m1 in moods:
    print(f"  {m1}  {transition[m1]['H']:.4f}  {transition[m1]['S']:.4f}")

print("\nEmission Matrix P(C|M):")
print("        R       G       B")
for m in moods:
    print(f"  {m}  {emission[m]['R']:.4f}  {emission[m]['G']:.4f}  {emission[m]['B']:.4f}")

#Step 4:Testing the model on R, B, G
observed = ["R", "B", "G"]
print("\nObserved shirt colors:", observed)
print("\nProbabilities of all mood sequences:")

best_seq = None
best_prob = -1

for seq in product(moods, repeat=3):#HHH, HHS, HSH, ... SSS
    m1, m2, m3 = seq
    prob = (initial[m1] * emission[m1][observed[0]] *
            transition[m1][m2] * emission[m2][observed[1]] *
            transition[m2][m3] * emission[m3][observed[2]])
    print(f"  P({m1},{m2},{m3},R,B,G) = {prob:.6f}")
    if prob > best_prob:
        best_prob = prob
        best_seq = seq

#Step 5:argmax
print("\nMost likely mood sequence:", ", ".join(best_seq))
print(f"Probability/argmax: {best_prob:.6f}")