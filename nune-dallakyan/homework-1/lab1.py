
text = """ Natural Language Processing (NLP) is a subfield of linguistics,
computer science, and artificial intelligence concerned with the
interactions between computers and human language. It's used to
analyze text, allowing machines to understand, interpret, and
manipulate human language. NLP has many real-world applications,
including machine translation, sentiment analysis, and chatbots.
"""

# բաժանել բոլոր բառերը
tokens = text.split()

print("1-tokenization")
print(tokens)
print()


#դարձնել փոքրատառ
lowercase_tokens = []

for token in tokens:
    lowercase_tokens.append(token.lower())


# հեռացնել կետադրական նշանները
punctuation = ".,;:()'\"-"

clean_tokens = []

for token in lowercase_tokens:
    clean_token = ""
    for character in token:
        if character not in punctuation:
            clean_token += character
    clean_tokens.append(clean_token)


print("2 & 3 - lowercase and punctuation removal")
print(clean_tokens)
print()

# հեռացնել ավելորդ բառերը 
stop_words = [
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "for",
    "to",
    "of",
    "and",
    "is",
    "are"
]

tokens_without_stop_words = []

for token in clean_tokens:
    if token not in stop_words:
        tokens_without_stop_words.append(token)

print("4 - stop word removal")
print(tokens_without_stop_words)
print()

#հեռացնել վերջավորությունները

suffixes = [
    "ing",
    "ed",
    "ly",
    "es",
    "s"
]

def simple_stem(word):
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


stemmed_tokens = []

for token in tokens_without_stop_words:
    stemmed_tokens.append(simple_stem(token))


print("5 - stemming")
print(stemmed_tokens)
print()


# գտնեկ արմատը
lemma_dict = {
    "is": "be",
    "used": "use",
    "has": "have",
    "machines": "machine",
    "applications": "application",
    "including": "include",
    "chatbots": "chatbot",
    "allowing": "allow"
}


def simple_lemmatize(word):
    if word in lemma_dict:
        return lemma_dict[word]
    return word


lemmatized_tokens = []

for token in tokens_without_stop_words:
    lemmatized_tokens.append(simple_lemmatize(token))


print("6 - lemmatization")
print(lemmatized_tokens)
print()

