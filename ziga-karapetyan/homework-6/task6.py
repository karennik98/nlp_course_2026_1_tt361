import random
import string

#Step 1:Read the file line by line
def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.readlines()
    except FileNotFoundError:
        print("Error: file", filename, "not found!")
        return []
    except OSError as e:
        print("Error reading file:", e)
        return []

#Step 2:Text processing
def clean(line):
    line = line.replace("—", " ").replace("-", " ")#dashes and hyphens -> space
    line = line.translate(str.maketrans("", "", string.punctuation))#remove punctuation
    return line.lower().split()

def add_count(dictionary, key, word):
    if key not in dictionary:
        dictionary[key] = {}
    dictionary[key][word] = dictionary[key].get(word, 0) + 1

#Turns counts into probabilities: P(word) = count(word) / total_count
def to_probabilities(counts):
    total = sum(counts.values())
    probs = {}
    for word in counts:
        probs[word] = counts[word] / total
    return probs

#Step 3:Training: building the three dictionaries 
def train(lines):
    initial = {}        #A.first word of a line
    first_order = {}    #B.w(t-1) -> w(t)
    second_order = {}   #C.(w(t-2), w(t-1)) -> w(t)
    total_lines = 0

    for line in lines:
        words = clean(line)
        if len(words) == 0:       
            continue
        total_lines += 1
        words.append("END")    

        #A.Initial word
        initial[words[0]] = initial.get(words[0], 0) + 1

        #B.First-order transitions (every word -> next word)
        for i in range(1, len(words)):
            add_count(first_order, words[i - 1], words[i])

        #C.Second-order transitions (two words -> next word)
        for i in range(2, len(words)):
            add_count(second_order, (words[i - 2], words[i - 1]), words[i])

    #A.P(w1) = count(w1 as first word) / total number of lines
    for word in initial:
        initial[word] = initial[word] / total_lines

    #B.P(wt | wt-1) = count(wt-1, wt) / count(wt-1)
    for word in first_order:
        first_order[word] = to_probabilities(first_order[word])

    #C.P(wt | wt-2, wt-1) = count(wt-2, wt-1, wt) / count(wt-2, wt-1)
    for pair in second_order:
        second_order[pair] = to_probabilities(second_order[pair])

    return initial, first_order, second_order

#Step 4:Word selection
#Method A:cumulative probability
def pick_word_cumulative(probs):
    r = random.random()
    cumulative = 0
    for word in probs:
        cumulative += probs[word]
        if r < cumulative:
            return word
    return word                    

#Method B:deterministic (highest probability)
def pick_word_deterministic(probs):
    return max(probs, key=probs.get)

#Step 5: Generation
def generate_line(initial, first_order, second_order, pick_word):
    w1 = pick_word(initial)#first word: P(w1)
    line = [w1]

    w2 = pick_word(first_order[w1])#second word: P(w2 | w1)
    if w2 == "END":
        return " ".join(line)
    line.append(w2)

    while True:#next words: P(wt | wt-2, wt-1)
        w3 = pick_word(second_order[(w1, w2)])
        if w3 == "END":               
            break
        line.append(w3)
        w1, w2 = w2, w3

    return " ".join(line)

#Main
lines = read_file("robert_frost (1).txt")
if lines:
    initial, first_order, second_order = train(lines)

    print("Generated poem (Method A - cumulative probability):\n")
    for i in range(4):
        print(generate_line(initial, first_order, second_order, pick_word_cumulative))

    print("\nGenerated poem (Method B - deterministic):\n")
    for i in range(4):
        print(generate_line(initial, first_order, second_order, pick_word_deterministic))