"""
Homework 1 - Text Preprocessing Exercise (Using Basic Python)

Pipeline:
    1. Tokenization
    2. Lowercasing
    3. Punctuation removal
    4. Stop word removal
    5. Stemming (simple suffix stripping)
    Bonus: dictionary-based lemmatization, compared with stemming

Only built-in Python functions are used (no imports, no NLP libraries).
"""

TEXT = (
    "Natural Language Processing (NLP) is a subfield of linguistics, computer science, "
    "and artificial intelligence concerned with the interactions between computers and "
    "human language. It's used to analyze text, allowing machines to understand, "
    "interpret, and manipulate human language. NLP has many real-world applications, "
    "including machine translation, sentiment analysis, and chatbots."
)


PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# Stop words:
STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "is", "are",
    "it", "its", "with", "between", "has", "have", "be", "by", "as", "or",
    "this", "that", "from", "was", "were",
}


# ---------------------------------------------------------------------------
# Step 1: Tokenization
# ---------------------------------------------------------------------------
def tokenize(text):
    """Split the text into tokens on whitespace.

    str.split() with no arguments splits on any run of whitespace (spaces,
    tabs, newlines) and drops empty strings. Punctuation stays attached to the
    words at this stage (e.g. "(NLP)", "linguistics,") and is handled in step 3.
    """
    return text.split()


# ---------------------------------------------------------------------------
# Step 2: Lowercasing
# ---------------------------------------------------------------------------
def lowercase(tokens):
    """Convert every token to lowercase so "NLP" and "nlp" are the same word."""
    return [token.lower() for token in tokens]


# ---------------------------------------------------------------------------
# Step 3: Punctuation removal
# ---------------------------------------------------------------------------
def remove_punctuation(tokens):
    """Remove every punctuation character from each token.

    We keep only characters that are not in PUNCTUATION. This removes
    surrounding punctuation ("(nlp)" -> "nlp", "language." -> "language") and
    also inner punctuation ("it's" -> "its", "real-world" -> "realworld").
    Tokens that become empty (a token that was only punctuation) are dropped.
    """
    cleaned = []
    for token in tokens:
        stripped = "".join(ch for ch in token if ch not in PUNCTUATION)
        if stripped:
            cleaned.append(stripped)
    return cleaned


# ---------------------------------------------------------------------------
# Step 4: Stop word removal
# ---------------------------------------------------------------------------
def remove_stop_words(tokens, stop_words=STOP_WORDS):
    """Drop tokens that appear in the stop word set.

    A set is used so each membership check is O(1).
    """
    return [token for token in tokens if token not in stop_words]


# ---------------------------------------------------------------------------
# Step 5: Stemming
# ---------------------------------------------------------------------------
# Suffixes ordered from longest to shortest so the most specific rule wins
# (e.g. "ations" is tried before "s").
SUFFIXES = [
    "ational", "ations", "ation", "ments", "ment", "ness", "ing", "ers",
    "ies", "ied", "ion", "ed", "er", "ly", "es", "s",
]


def simple_stem(word, min_stem_length=3):
    """Reduce a word to a rough root by stripping one common suffix.

    Rules:
      * Try suffixes from longest to shortest and strip the first that matches.
      * Only strip if at least `min_stem_length` characters remain, so short
        words like "is" or "was" are not destroyed.
      * Do not strip "s" from words ending in "ss", "us" or "is"
        (e.g. "class", "status", "analysis"); they are not plurals.
      * Strip "es" only after s, x, z, ch, sh ("boxes" -> "box"); otherwise
        only the "s" is removed ("machines" -> "machine").
      * "ies"/"ied" are replaced with "y" (e.g. "studies" -> "study").
      * A final doubled consonant is reduced ("running" -> "runn" -> "run").
    This is a deliberately simplified Porter-style stemmer.
    """
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem_length:
            if suffix == "s" and word[-2:] in ("ss", "us", "is"):
                continue
            if suffix == "es" and not word[:-2].endswith(("s", "x", "z", "ch", "sh")):
                continue
            stem = word[: -len(suffix)]
            if suffix in ("ies", "ied"):
                stem += "y"
            # Undo consonant doubling: "runn" -> "run", "stopp" -> "stop".
            if (
                len(stem) >= 2
                and stem[-1] == stem[-2]
                and stem[-1] not in "aeiouls"
            ):
                stem = stem[:-1]
            return stem
    return word


def stem_tokens(tokens):
    return [simple_stem(token) for token in tokens]


# ---------------------------------------------------------------------------
# Bonus: dictionary-based lemmatization
# ---------------------------------------------------------------------------
LEMMA_DICT = {
    "is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "being": "be",
    "has": "have", "had": "have", "having": "have",
    "used": "use", "using": "use", "uses": "use",
    "allowing": "allow", "allows": "allow", "allowed": "allow",
    "concerned": "concern",
    "computers": "computer",
    "machines": "machine",
    "interactions": "interaction",
    "applications": "application",
    "chatbots": "chatbot",
    "linguistics": "linguistics", 
    "including": "include", "includes": "include", "included": "include",
    "many": "many",
    "better": "good", "best": "good",
    "children": "child", "men": "man", "women": "woman",
    "ran": "run", "running": "run",
}


def simple_lemmatize(word):
    """Look the word up in LEMMA_DICT; if it is missing, return it unchanged.

    Leaving unknown words unchanged is safer than guessing: a lemmatizer
    should always return a real word.
    """
    return LEMMA_DICT.get(word, word)


def lemmatize_tokens(tokens):
    return [simple_lemmatize(token) for token in tokens]


# ---------------------------------------------------------------------------
# Run the pipeline and print the output of every step
# ---------------------------------------------------------------------------
def print_step(title, tokens):
    print(f"{title} ({len(tokens)} tokens):")
    print(tokens)
    print()


def main():
    # Step 1
    tokens = tokenize(TEXT)
    print_step("1. Tokens after tokenization", tokens)

    # Steps 2 and 3
    cleaned = remove_punctuation(lowercase(tokens))
    print_step("2. Tokens after lowercasing and punctuation removal", cleaned)

    # Step 4
    no_stop = remove_stop_words(cleaned)
    print_step("3. Tokens after stop word removal", no_stop)

    # Step 5
    stemmed = stem_tokens(no_stop)
    print_step("4. Tokens after stemming", stemmed)

    # Bonus
    lemmatized = lemmatize_tokens(no_stop)
    print_step("Bonus. Tokens after lemmatization", lemmatized)

    # Compare stemming and lemmatization side by side for the words that
    # differ, to show where the two approaches disagree.
    print("Bonus. Stemming vs. lemmatization (words where they differ):")
    print(f"{'word':<15}{'stem':<15}{'lemma':<15}")
    print("-" * 45)
    seen = set()
    for word, stem, lemma in zip(no_stop, stemmed, lemmatized):
        if word in seen or stem == lemma:
            continue
        seen.add(word)
        print(f"{word:<15}{stem:<15}{lemma:<15}")
    print()

    # The lemmatizer only runs after stop word removal above, so show it on
    # the words that were removed as stop words too, where it matters most.
    print("Bonus. Lemmatization of verb forms removed as stop words:")
    for word in ["is", "are", "has"]:
        print(f"  {word} -> lemma: {simple_lemmatize(word)}, stem: {simple_stem(word)}")



if __name__ == "__main__":
    main()
