import pandas as pd
import os
import string
from sklearn.model_selection import train_test_split
import numpy as np
from nltk.util import ngrams
from sklearn.metrics import (accuracy_score,f1_score,precision_score,recall_score)

#Function to calculate initial and transition probabilities for the Naive Bayes classifier
def initialandtransition_probabilities(DATA,eps=1):
    DATAVOC = {}
    DATAINIT = {}
    DATATRANS = {}

    for message in DATA:
        if not message:
            continue
        if message[0] not in DATAINIT:
            DATAINIT[message[0]] = 1
        else:
            DATAINIT[message[0]] += 1
        for word in message:
            if word not in DATAVOC:
                DATAVOC[word] = 1
            else:
                DATAVOC[word] += 1
            if word not in DATAINIT:
                DATAINIT[word] = 0
    
        bigrams = ngrams(message, 2)
        for gram in bigrams:
            if gram not in DATATRANS:
                DATATRANS[gram] = 1
            else:
                DATATRANS[gram] += 1

    N = len(DATA)
    M = len(DATAVOC)
    nepsm = N + eps * M
    for word in DATAINIT:
        DATAINIT[word] = np.log((DATAINIT[word] + eps) / nepsm)
    for gram in DATATRANS:
        DATATRANS[gram] = np.log((DATATRANS[gram] + eps) / (DATAVOC[gram[0]] + eps * M))

    return DATAINIT, DATAVOC, DATATRANS, nepsm

#Function to predict the probability of a message being spam or ham using the Naive Bayes classifier
def probpredict(message, INIT, VOC, TRANS, prior, nepsm, eps=1):
    M = len(VOC)
    prob = prior
    if not message:
        return prob
    prob += INIT.get(message[0], np.log(eps/nepsm))
    bigrams = ngrams(message, 2)
    for gram in bigrams:
        if gram in TRANS:
            prob += TRANS[gram]
        elif gram[0] in VOC:
            prob += np.log(eps / (VOC[gram[0]] + eps * M))
        else:
            prob += np.log(1 / M)
    return prob

if __name__ == "__main__":
    # Load the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "spam.csv")

    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found!")
        exit()
    except OSError as e:
        print(f"Error reading file: {e}")
        exit()

    # Preprocess the dataset
    data = data.dropna()
    data = data.drop_duplicates()
    data['Message'] = data['Message'].str.lower()
    data['Message'] = data['Message'].str.translate(str.maketrans('', '', string.punctuation))
    data['Category'] = data['Category'].map({'spam': 1, 'ham': 0})
    data['Message'] = data['Message'].str.replace(r'\d+', '', regex=True)
    data['Message'] = data['Message'].str.replace(r'\s+', ' ', regex=True).str.strip()
    data['Message'] = data['Message'].str.split()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Category'], test_size=0.2, random_state=42, stratify=data['Category'])
    SPAMD = X_train[y_train == 1]
    HAMD = X_train[y_train == 0]
    # Calculate prior probabilities for spam and ham
    priorspam = np.log(y_train.value_counts(normalize=True).get(1, 0))
    priorham = np.log(y_train.value_counts(normalize=True).get(0, 0))

    # Calculate initial and transition probabilities for spam and ham
    SPAMINIT, SPAMDVOC, SPAMTRANS, SPAMnepsm = initialandtransition_probabilities(SPAMD)
    HAMINIT, HAMDVOC, HAMTRANS, HAMnepsm = initialandtransition_probabilities(HAMD)

    predict = pd.DataFrame()
    predict['Spam Probability'] = X_test.apply(lambda x: probpredict(x, SPAMINIT, SPAMDVOC, SPAMTRANS, priorspam, SPAMnepsm))
    predict['Ham Probability'] = X_test.apply(lambda x: probpredict(x, HAMINIT, HAMDVOC, HAMTRANS, priorham, HAMnepsm))

    # Evaluate the model's performance
    y_pred = (predict['Spam Probability'] > predict['Ham Probability']).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")