import os
import random
import string
from nltk.util import ngrams

# Read the text from the file

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "robert_frost (1).txt")

try:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    print("Error: File not found!")
    exit()
except OSError as e:
    print(f"Error reading file: {e}")
    exit()

# Text preprocessing
text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))

# Create a dictionary to store the word pairs and their corresponding next words
initdic_c = dict()
firstdic_c = dict()
seconddic_c = dict()
initdic_p = dict()
firstdic_p = dict()
seconddic_p = dict()
first = dict()
second = dict()
word_p = dict()

train = text.split('\n')
lines = len(train)
linescount = 0

for i in range(lines):
    words = train[i].split()
    if not words:
        continue
    linescount += 1
    words.append('END')
    train[i] = words
    for word in words:
        if word in word_p:
            word_p[word] = word_p[word] + 1
        else:
            word_p[word] = 1
    if words[0] in initdic_c:
        initdic_c[words[0]] = initdic_c[words[0]] + 1
    else:
        initdic_c[words[0]] = 1
    bigrams = ngrams(words, 2)
    for gram in bigrams:
        if gram in firstdic_c:
            firstdic_c[gram] = firstdic_c[gram] + 1
        else:
            firstdic_c[gram] = 1
        if gram[0] in first:
            first[gram[0]] = first[gram[0]] + 1
        else:
            first[gram[0]] = 1
    trigrams = ngrams(words, 3)
    for gram in trigrams:
        if gram in seconddic_c:
            seconddic_c[gram] = seconddic_c[gram] + 1
        else:
            seconddic_c[gram] = 1
        if gram[0:2] in second:
            second[gram[0:2]] = second[gram[0:2]] + 1
        else:
            second[gram[0:2]] = 1

total_count = sum(word_p.values())
word_p = {k: v / total_count for k, v in word_p.items()}

initdic_p = {k: v / linescount for k, v in initdic_c.items()}

for i in firstdic_c:
    firstdic_p[i] = firstdic_c[i] / first[i[0]]

for i in seconddic_c:
    seconddic_p[i] = seconddic_c[i] / second[i[0:2]]

firstdic_p = dict(sorted(firstdic_p.items(), key=lambda item: item[1], reverse=True))
seconddic_p = dict(sorted(seconddic_p.items(), key=lambda item: item[1], reverse=True))

# How many sentences to generate and maximal length of each sentence
n = 4
maxlen = 20

# Generate a sentence using Deterministic Method

dettext = []

for i in range(n):
    detsentences = [max(initdic_p, key=initdic_p.get)]

    for secondword in firstdic_p:
        if secondword[0] == detsentences[-1]:
            detsentences.append(secondword[1])
            break

    while len(detsentences) < maxlen:
        last_two = tuple(detsentences[-2:])
        for thirdword in seconddic_p:
            if thirdword[0:2] == last_two:
                detsentences.append(thirdword[2])
                break
        if detsentences[-1] == 'END':
            detsentences.pop()
            break

    dettext.append(' '.join(detsentences))
    dettext.append('\n')

print("Generated sentence (Deterministic)\n---\n" + ''.join(dettext))

# Generate a sentence using Cumulative Probability Method
cumulativeprobtext = []

for i in range(n):
    alpha = random.random()
    for firstword in initdic_p:
        if alpha < initdic_p[firstword]:
            cumulativesentences = [firstword]
            break
        else:
            alpha = alpha - initdic_p[firstword]

    alpha = random.random()
    for secondword in firstdic_p:
        if secondword[0] == cumulativesentences[-1]:
            if alpha < firstdic_p[secondword]:
                cumulativesentences.append(secondword[1])
                break
            else:
                alpha = alpha - firstdic_p[secondword]

    while len(cumulativesentences) < maxlen:
        alpha = random.random()
        last_two = tuple(cumulativesentences[-2:])
        for thirdword in seconddic_p:
            if thirdword[0:2] == last_two:
                if alpha < seconddic_p[thirdword]:
                    cumulativesentences.append(thirdword[2])
                    break
                else:
                    alpha = alpha - seconddic_p[thirdword]
        if cumulativesentences[-1] == 'END':
            cumulativesentences.pop()
            break

    cumulativeprobtext.append(' '.join(cumulativesentences))
    cumulativeprobtext.append('\n')

print("Generated sentence (Cumulative Probability)\n---\n" + ''.join(cumulativeprobtext))