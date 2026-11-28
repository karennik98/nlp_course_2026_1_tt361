import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "student_data.csv")

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: File not found!")
    exit()
except OSError as e:
    print(f"Error reading file: {e}")
    exit()

MOODS = {'H','S'}
COLORS = {'R','G','B'}
STUDENTS = 5
DAYS = 20

init = dict()
for m in MOODS:
    init[m] = len((data[(data['Day']==1) & (data['Mood']==m)]))/len((data[data['Day']==1]))

print("Initial Probability Matrix\n",init,"\n----------------")

transitions = dict()
for i in MOODS:
    transitions[i] = dict()
    for m in MOODS:
        transitions[i][m] = 0

for i in range(len(data)):
    if data.iloc[i]['Day'] == DAYS:
        continue
    transitions[data.iloc[i]['Mood']][data.iloc[i+1]['Mood']]+=1

for i in MOODS:
    total = sum(transitions[i].values())
    transitions[i] = {k: v / total for k, v in transitions[i].items()}

print("Transition Matrix\n",transitions,"\n----------------")

emission = dict()

for mood in MOODS:
    emission[mood] = dict()
    for color in COLORS:
        emission[mood][color] = len((data[(data['Mood']==mood) & (data['ShirtColor']==color)]))
    total = sum(emission[mood].values())
    emission[mood] = {k: v / total for k, v in emission[mood].items()}

print("Emission  Matrix\n",emission,"\n----------------")    

def prob(moods,colors):
    probability = init[moods[0]]*emission[moods[0]][colors[0]]
    for i in range(1,len(colors)):
        probability *= transitions[moods[i-1]][moods[i]]*emission[moods[i]][colors[i]]
    return probability

moodsequence = [['H','H','H'], ['H','H','S'], ['H','S','H'], ['H','S','S'], ['S','H','H'], ['S','H','S'], ['S','S','H'], ['S','S','S']]
colorsequence = ['R','B','G']

probabilities = []
for i in moodsequence:
    probabilities.append(prob(i,colorsequence))

output = {
    'MOODS': moodsequence,
    'PROBABILITY' : probabilities
}

out = pd.DataFrame(output)
best_sequence = out.loc[out['PROBABILITY'].idxmax()]
print(out)
print(f"Maximal probability states for {colorsequence}\n",best_sequence)