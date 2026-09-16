import string


text = """Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial
intelligence concerned with the interactions between computers and human language. It's used to
analyze text, allowing machines to understand, interpret, and manipulate human language. NLP has
many real-world applications, including machine translation, sentiment analysis, and chatbots."""

# Step 1: split the text on whitespace.
tokens = text.split()

# Steps 2 and 3: lowercase each token and remove punctuation.
punctuation_table = str.maketrans("", "", string.punctuation)
clean_tokens = [token.lower().translate(punctuation_table) for token in tokens]
clean_tokens = [token for token in clean_tokens if token]

# Step 4: remove common stop words.
stop_words = {
    "a", "an", "and", "are", "at", "between", "for", "has", "in",
    "is", "it", "its", "of", "on", "the", "to", "with",
}
filtered_tokens = [token for token in clean_tokens if token not in stop_words]


# Step 5: remove a few common suffixes with simple rules.
def simple_stem(word):
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        return stem + "e" if stem.endswith("clud") else stem
    if word.endswith("ed") and len(word) > 3:
        stem = word[:-2]
        return stem + "e" if stem.endswith("us") else stem
    if word.endswith("s") and len(word) > 3 and not word.endswith(("is", "ss", "us")):
        return word[:-1]
    return word


stemmed_tokens = [simple_stem(token) for token in filtered_tokens]

# Bonus: map known words to their base forms.
lemma_map = {
    "is": "be",
    "are": "be",
    "has": "have",
    "used": "use",
    "interactions": "interaction",
    "computers": "computer",
    "allowing": "allow",
    "machines": "machine",
    "applications": "application",
    "including": "include",
    "chatbots": "chatbot",
}
lemmatized_tokens = [lemma_map.get(token, token) for token in filtered_tokens]

print("1. Tokens:", tokens)
print("\n2. Lowercased tokens without punctuation:", clean_tokens)
print("\n3. Tokens without stop words:", filtered_tokens)
print("\n4. Stemmed tokens:", stemmed_tokens)
print("\nBonus - lemmatized tokens:", lemmatized_tokens)
