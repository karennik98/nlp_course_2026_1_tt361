"""
Text Preprocessing Exercise (Using Basic Python)
Steps: tokenize -> lowercase -> remove punctuation -> remove stop words -> stem
"""

RAW_TEXT = (
    "Natural Language Processing (NLP) is a subfield of linguistics, computer "
    "science, and artificial intelligence concerned with the interactions "
    "between computers and human language. It's used to analyze text, allowing "
    "machines to understand, interpret, and manipulate human language. NLP has "
    "many real-world applications, including machine translation, sentiment "
    "analysis, and chatbots."
)

STOP_WORDS = ["the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "is", "are"]


def tokenize(text):
    """Step 1: split the text into individual word tokens.

    A plain whitespace split is enough here - punctuation is still
    attached to tokens (e.g. "(NLP)", "linguistics,") on purpose,
    since removing it is step 3's job, not step 1's.
    """
    return text.split()


def lowercase(tokens):
    """Step 2: convert all tokens to lowercase, in place.

    Lowercasing is always one token in, one token out, so each
    position can just be overwritten directly - no new list needed.
    """
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()


def remove_punctuation(tokens):
    """Step 3: strip punctuation marks from each token, in place.

    A hyphen joins two separate words (e.g. "real-world"), so each token
    is split on "-" first, turning it into "real" and "world" instead of
    fusing them into "realworld". Every other non-letter character
    (parentheses, commas, periods, apostrophes, ...) is just dropped.

    One token can become zero, one, or two tokens here, so the count can
    change - that rules out overwriting positions directly. The new
    contents are collected first, then spliced into the same list object
    with tokens[:] = ..., which keeps its identity instead of returning
    a separate new list.
    """
    cleaned = []
    for token in tokens:
        for piece in token.split("-"):
            word = "".join(ch for ch in piece if ch.isalpha())
            if word:
                cleaned.append(word)
    tokens[:] = cleaned


def remove_stop_words(tokens):
    """Step 4: drop common stop words, in place.

    Removing words changes the count, so - like step 3 - the kept
    words are collected first, then spliced back with tokens[:] = ...
    """
    kept = [token for token in tokens if token not in STOP_WORDS]
    tokens[:] = kept


def stem(tokens):
    """Step 5: reduce words to a root form using simple suffix stripping, in place.

    Suffixes are general patterns, not tuned to this text specifically:
      -ing, -ed, -ly, -s are dropped, but only if at least 3 letters
      remain, so short words aren't mangled (e.g. "used" -> "us" is
      blocked). Words ending in "sis", "ss", or "us" are left alone,
      since there the trailing "s" is part of the root rather than a
      suffix (e.g. "analysis", "class", "virus" should stay as is).

    Each word maps to exactly one stemmed word, so - like step 2 -
    positions are overwritten directly.
    """
    for i in range(len(tokens)):
        word = tokens[i]
        if word.endswith("ing") and len(word[:-3]) >= 3:
            tokens[i] = word[:-3]
        elif word.endswith("ed") and len(word[:-2]) >= 3:
            tokens[i] = word[:-2]
        elif word.endswith("ly") and len(word[:-2]) >= 3:
            tokens[i] = word[:-2]
        elif word.endswith("s") and not word.endswith(("sis", "ss", "us")) and len(word[:-1]) >= 3:
            tokens[i] = word[:-1]


def main():
    tokens = tokenize(RAW_TEXT)
    print("Raw text:", RAW_TEXT)
    print()
    print("1. Tokens:", tokens)
    print()

    print("step 2 lowercase")
    lowercase(tokens)
    print("2. Lowercased:", tokens)
    print()

    print("step 3 remove punctuation")
    remove_punctuation(tokens)
    print("3. Punctuation removed:", tokens)
    print()

    remove_stop_words(tokens)
    print("4. Stop words removed:", tokens)
    print()

    stem(tokens)
    print("5. Stemmed:", tokens)
    print()


if __name__ == "__main__":
    main()
