import csv
import math
import random
import re

#Step 1:Loading and preprocessing the data
def clean(text):
    text = text.lower()                         
    text = re.sub(r"[^a-z0-9\s]", " ", text)#removing punctuation
    return text.split()                      

data = []
with open("spam.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        words = clean(row["Message"])
        if words: 
            data.append((row["Category"], words))

#Step 2:Splitting the data into train (80%) and test (20%)
random.seed(42)
random.shuffle(data)
split = int(0.8 * len(data))
train = data[:split]
test = data[split:]

#Step 3:Training the model
spam_msgs = [words for label, words in train if label == "spam"]
ham_msgs = [words for label, words in train if label == "ham"]

#Log prior probabilities
log_p_spam = math.log(len(spam_msgs) / len(train))
log_p_ham = math.log(len(ham_msgs) / len(train))

vocab = set()
for label, words in train:
    vocab.update(words)
M = len(vocab)
eps = 1

def count_model(messages):
    initial = {}#count(s1 = i): how often a word starts a message
    transition = {}#count(i -> j): how often word j comes after word i
    from_count = {}#count(i): how often word i is followed by any word
    for words in messages:
        initial[words[0]] = initial.get(words[0], 0) + 1
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            transition[(a, b)] = transition.get((a, b), 0) + 1
            from_count[a] = from_count.get(a, 0) + 1
    return initial, transition, from_count, len(messages)

spam_model = count_model(spam_msgs)
ham_model = count_model(ham_msgs)

#Step 4:Prediction
def log_likelihood(words, model):
    initial, transition, from_count, N = model
    #Initial probability: (count(s1=i) + eps) / (N + eps*M)
    log_p = math.log((initial.get(words[0], 0) + eps) / (N + eps * M))
    #Transition probabilities: (count(i->j) + eps) / (count(i) + eps*M)
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        log_p += math.log((transition.get((a, b), 0) + eps) / (from_count.get(a, 0) + eps * M))
    return log_p

def predict(words):
    spam_score = log_likelihood(words, spam_model) + log_p_spam
    ham_score = log_likelihood(words, ham_model) + log_p_ham
    if spam_score > ham_score:
        return "spam"
    return "ham"

#Step 5:Evaluation
correct = 0
for label, words in test:
    if predict(words) == label:
        correct += 1

print("Train size:", len(train), " Test size:", len(test))
print("log P(spam) =", round(log_p_spam, 4))
print("log P(ham)  =", round(log_p_ham, 4))
print("Accuracy:", round(correct / len(test) * 100, 2), "%")

msg = "Congratulations! You won a free prize, call now"
print(msg, "->", predict(clean(msg)))