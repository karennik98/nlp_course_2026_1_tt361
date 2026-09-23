import string

TEXT = (  # input text from the assignment
    "Natural Language Processing (NLP) is a subfield of linguistics, computer science, "
    "and artificial intelligence concerned with the interactions between computers and "
    "human language. It's used to analyze text, allowing machines to understand, "
    "interpret, and manipulate human language. NLP has many real-world applications, "
    "including machine translation, sentiment analysis, and chatbots."
)

STOP_WORDS = {  # hint list + a few more common function words
    "the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "is", "are",
    "it", "its", "with", "has", "have", "be", "was", "were", "this", "that", "by",
}


def tokenize(text):  # step 1: split on whitespace, punctuation stays attached for now
    return text.split()


def to_lowercase(tokens):  # step 2: 'NLP' and 'nlp' become the same token
    return [token.lower() for token in tokens]


def remove_punctuation(tokens):  # step 3: drop punctuation chars, drop tokens that become empty
    punctuation = set(string.punctuation)
    cleaned = []
    for token in tokens:
        new_token = "".join(ch for ch in token if ch not in punctuation)
        if new_token:
            cleaned.append(new_token)
    return cleaned


def remove_stop_words(tokens, stop_words=STOP_WORDS):  # step 4: remove common words with little meaning
    return [token for token in tokens if token not in stop_words]


SUFFIX_RULES = [  # (suffix, replacement), longer suffixes first
    ("ational", "ate"),
    ("ization", "ize"),
    ("fulness", "ful"),
    ("ations", ""),
    ("ation", ""),
    ("ness", ""),
    ("ions", ""),
    ("ion", ""),
    ("ing", ""),
    ("ies", "y"),
    ("sses", "ss"),
    ("ed", ""),
    ("ly", ""),
    ("s", ""),
]

MIN_STEM_LENGTH = 3  # don't strip if the stem would get too short (e.g. 'used')


def simple_stem(word):  # step 5: strip the first matching suffix ('ss'/'is'/'us' endings are not plurals)
    for suffix, replacement in SUFFIX_RULES:
        if word.endswith(suffix):
            if suffix == "s" and word.endswith(("ss", "is", "us")):
                continue
            stem = word[: -len(suffix)]
            if len(stem) >= MIN_STEM_LENGTH:
                return stem + replacement
    return word


def stem_tokens(tokens):  # apply stemming to every token
    return [simple_stem(token) for token in tokens]


LEMMA_DICT = {  # bonus: inflected form -> base form
    "is": "be", "are": "be", "was": "be", "were": "be", "been": "be",
    "has": "have", "had": "have",
    "used": "use", "concerned": "concern",
    "allowing": "allow", "including": "include", "processing": "process",
    "linguistics": "linguistics",
    "interactions": "interaction", "computers": "computer",
    "machines": "machine", "applications": "application",
    "chatbots": "chatbot", "analysis": "analysis",
    "many": "many", "its": "it",
}


def lemmatize(word):  # bonus: dictionary lookup, fallback removes plural 's'
    if word in LEMMA_DICT:
        return LEMMA_DICT[word]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def lemmatize_tokens(tokens):  # apply lemmatization to every token
    return [lemmatize(token) for token in tokens]


def main():  # run all steps, print the result of each and compare stems vs lemmas
    print("Original text:")
    print(TEXT)
    print()

    tokens = tokenize(TEXT)
    print(f"1. Tokens after tokenization ({len(tokens)}):")
    print(tokens)
    print()

    tokens = remove_punctuation(to_lowercase(tokens))
    print(f"2. Tokens after lowercasing and punctuation removal ({len(tokens)}):")
    print(tokens)
    print()

    tokens = remove_stop_words(tokens)
    print(f"3. Tokens after stop word removal ({len(tokens)}):")
    print(tokens)
    print()

    stems = stem_tokens(tokens)
    print(f"4. Tokens after stemming ({len(stems)}):")
    print(stems)
    print()

    lemmas = lemmatize_tokens(tokens)
    print(f"Bonus. Tokens after lemmatization ({len(lemmas)}):")
    print(lemmas)
    print()

    print("Bonus. Stemming vs. lemmatization:")
    print(f"{'token':<15}{'stem':<15}{'lemma':<15}")
    print("-" * 45)
    for token, stem, lemma in zip(tokens, stems, lemmas):
        print(f"{token:<15}{stem:<15}{lemma:<15}")
    print()
    print(
        "Observations: the stemmer only chops suffixes, so it can produce\n"
        "non-words ('applications' -> 'applic', 'including' -> 'includ') and it\n"
        "cannot handle irregular forms ('used' stays 'used'). The dictionary\n"
        "lemmatizer always returns real words ('applications' -> 'application',\n"
        "'used' -> 'use'), but it only knows the words that are in its dictionary."
    )


if __name__ == "__main__":
    main()
