import string

text = """Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial
intelligence concerned with the interactions between computers and human language. It's used to
analyze text, allowing machines to understand, interpret, and manipulate human language. NLP has
many real-world applications, including machine translation, sentiment analysis, and chatbots."""

tokens_step1 = text.split()

translator = str.maketrans('', '', string.punctuation)
tokens_step2 = [tok.lower().translate(translator) for tok in tokens_step1]
tokens_step2 = [tok for tok in tokens_step2 if tok]

stop_words = ["the", "a", "an", "in", "on", "at", "for", "to",
              "of", "and", "is", "are", "with", "has", "it",
              "its", "between"]

tokens_step4 = [tok for tok in tokens_step2 if tok not in stop_words]

def simple_stem(word):
    suffixes = ["ational", "ization", "ing", "edly", "ed",
                "ies", "ied", "ly", "es", "s"]
    for suf in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word

tokens_step5 = [simple_stem(tok) for tok in tokens_step4]

lemma_dict = {
    "is": "be", "are": "be", "was": "be", "were": "be",
    "concerned": "concern", "interactions": "interaction",
    "computers": "computer", "allowing": "allow",
    "machines": "machine", "applications": "application",
    "interpret": "interpret", "including": "include",
    "translation": "translate", "analysis": "analyze",
    "chatbots": "chatbot",
}

def simple_lemmatize(word):
    return lemma_dict.get(word, word)

tokens_lemmatized = [simple_lemmatize(tok) for tok in tokens_step4]

print(tokens_step1)
print(tokens_step2)
print(tokens_step4)
print(tokens_step5)
print(tokens_lemmatized)
