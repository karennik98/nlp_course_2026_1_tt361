import string

text = """Natural Language Processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and human
language. It's used to analyze text, allowing machines to understand, interpret, and
manipulate human language. NLP has many real-world applications, including machine
translation, sentiment analysis, and chatbots."""


def tokenize(text):
    return text.split()

tokens_step1 = tokenize(text)
print("Step 1:")
print(tokens_step1)
print(f"Count: {len(tokens_step1)}\n")


def lowercase_and_remove_punctuation(tokens):
    cleaned_tokens = []
    for token in tokens:
        token = token.lower()
        token = token.translate(str.maketrans('', '', string.punctuation))
        if token:
            cleaned_tokens.append(token)
    return cleaned_tokens

tokens_step2_3 = lowercase_and_remove_punctuation(tokens_step1)
print("Step 2-3:")
print(tokens_step2_3)
print(f"Count: {len(tokens_step2_3)}\n")


stop_words = ["the", "a", "an", "in", "on", "at", "for", "to", "of",
              "and", "is", "are", "with", "between", "it", "its", "has", "many"]

def remove_stop_words(tokens, stop_words):
    return [token for token in tokens if token not in stop_words]

tokens_step4 = remove_stop_words(tokens_step2_3, stop_words)
print("Step 4:")
print(tokens_step4)
print(f"Count: {len(tokens_step4)}\n")


def simple_stem(word):
    suffixes = ["ications", "ication", "ing", "tion", "sion",
                "ies", "ied", "es", "ed", "ly", "s"]

    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]

    return word

def stem_tokens(tokens):
    return [simple_stem(token) for token in tokens]

tokens_step5 = stem_tokens(tokens_step4)
print("Step 5:")
print(tokens_step5)
print(f"Count: {len(tokens_step5)}\n")


lemma_dict = {
    "is": "be",
    "are": "be",
    "was": "be",
    "were": "be",
    "has": "have",
    "have": "have",
    "had": "have",
    "computers": "computer",
    "interactions": "interaction",
    "machines": "machine",
    "applications": "application",
    "concerned": "concern",
    "used": "use",
    "understand": "understand",
    "interpret": "interpret",
    "manipulate": "manipulate",
    "including": "include",
}

def lemmatize(tokens, lemma_dict):
    return [lemma_dict.get(token, token) for token in tokens]

tokens_lemmatized = lemmatize(tokens_step4, lemma_dict)
print("Bonus (lemmatization):")
print(tokens_lemmatized)
print()

print("=" * 60)
print("Stemming vs Lemmatization comparison")
print("=" * 60)
for original, stemmed, lemmatized in zip(tokens_step4, tokens_step5, tokens_lemmatized):
    if stemmed != lemmatized:
        print(f"{original:20} | stem: {stemmed:15} | lemma: {lemmatized}")