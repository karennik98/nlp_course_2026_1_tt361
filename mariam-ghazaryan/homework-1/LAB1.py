"""Homework 1: text preprocessing and a stemming/lemmatization comparison."""
import string

text = """Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial
intelligence concerned with the interactions between computers and human language. It's used to
analyze text, allowing machines to understand, interpret, and manipulate human language. NLP has
many real-world applications, including machine translation, sentiment analysis, and chatbots."""

# Step 1: split the paragraph into whitespace-separated tokens.
tokens_step1 = text.split()

# Steps 2 and 3: lowercase tokens and remove ASCII punctuation.
translator = str.maketrans('', '', string.punctuation)
tokens_step2 = [tok.lower().translate(translator) for tok in tokens_step1]
tokens_step2 = [tok for tok in tokens_step2 if tok]

# Step 4: remove common function words with this small educational list.
stop_words = {"the", "a", "an", "in", "on", "at", "for", "to",
              "of", "and", "is", "are", "with", "has", "it",
              "its", "between"}
tokens_step4 = [tok for tok in tokens_step2 if tok not in stop_words]


def simple_stem(word):
    """Strip one matching suffix; this demonstration can yield nonwords."""
    suffixes = ["ational", "ization", "ing", "edly", "ed",
                "ies", "ied", "ly", "es", "s"]
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


# Step 5: apply the suffix-based stemmer to the filtered tokens.
tokens_step5 = [simple_stem(tok) for tok in tokens_step4]

# Bonus: a small dictionary with lemmas appropriate to this paragraph.
# Nouns keep their part of speech: translation and analysis stay unchanged.
# Verb forms such as allowing and including map to their base verbs.
lemma_dict = {
    "is": "be", "are": "be", "was": "be", "were": "be",
    "concerned": "concern", "interactions": "interaction",
    "computers": "computer", "allowing": "allow",
    "machines": "machine", "applications": "application",
    "interpret": "interpret", "including": "include",
    "translation": "translation", "analysis": "analysis",
    "chatbots": "chatbot",
}


def simple_lemmatize(word):
    """Return a paragraph-specific lemma; unknown words remain unchanged."""
    return lemma_dict.get(word, word)


# Lemmatize the same filtered tokens so the two results are comparable.
tokens_lemmatized = [simple_lemmatize(tok) for tok in tokens_step4]

print("Step 1 - tokenization:", tokens_step1)
print("Steps 2 and 3 - lowercase and punctuation removed:", tokens_step2)
print("Step 4 - stop words removed:", tokens_step4)
print("Step 5 - stemming:", tokens_step5)
print("Bonus - lemmatization:", tokens_lemmatized)

# Compare representative words from the paragraph, including both noun fixes.
print("\nComparison of stemming and lemmatization:")
print(f"{'Word':<16}{'Stem':<16}{'Lemma':<16}")
for word in ("machines", "including", "applications", "translation", "analysis"):
    print(f"{word:<16}{simple_stem(word):<16}{simple_lemmatize(word):<16}")
print("\nStemming removes suffixes mechanically and may produce nonwords "
      "such as 'machin', 'includ', and 'analysi'.")
print("Lemmatization returns a base dictionary form while preserving the "
      "intended part of speech: 'machines' becomes 'machine', while the "
      "nouns 'translation' and 'analysis' remain unchanged.")
print("These are simple educational implementations: the stemmer uses "
      "suffix rules and the lemmatizer covers only this paragraph, "
      "rather than performing general part-of-speech analysis.")
