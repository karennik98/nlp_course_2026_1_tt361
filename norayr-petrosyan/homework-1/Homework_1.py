text = """Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. It's used to analyze text, allowing machines to understand, interpret, and manipulate human language. NLP has many real-world applications, including machine translation, sentiment analysis, and chatbots."""

#1. Split at spaces and keep the original spelling
tokens = text.split()
print("1. Original tokens:", tokens)

#2-3. Lowercase each word and drop punctuation, including the hyphen
cleaned = []
for token in tokens:
    word = "".join(char for char in token.lower() if char.isalnum())
    if word:
        cleaned.append(word)
print("2-3. Cleaned tokens:", cleaned)

#4. "It's" became "its" when the apostrophe was removed
stop_words = {"a", "an", "the", "is", "are", "and", "of", "to", "with",
              "between", "its", "has", "in", "on", "at", "for"}
filtered = [word for word in cleaned if word not in stop_words]
print("4. After stop word removal:", filtered)


#5. Strip common endings. Leave words like "analysis" and "linguistics" alone.
def stem(word):
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "is", "ics")):
        return word[:-1]
    return word


print("5. Stems:", [stem(word) for word in filtered])
# Bonus
base_forms = {
    "is": "be", "are": "be", "has": "have", "processing": "process",
    "concerned": "concern", "interactions": "interaction", "computers": "computer",
    "used": "use", "allowing": "allow", "machines": "machine",
    "applications": "application", "including": "include", "chatbots": "chatbot"
}


def lemmatize(word):
    return base_forms.get(word, word)


print("Bonus. Lemmas:", [lemmatize(word) for word in filtered])
print("Words with different results (word, stem, lemma):")
for word in filtered:
    if stem(word) != lemmatize(word):
        print(word, stem(word), lemmatize(word))
        
# 1. Original tokens: ['Natural', 'Language', 'Processing', '(NLP)', 'is', 'a', 'subfield', 'of', 'linguistics,', 'computer', 'science,', 'and', 'artificial', 'intelligence', 'concerned', 'with', 'the', 'interactions', 'between', 'computers', 'and', 'human', 'language.', "It's", 'used', 'to', 'analyze', 'text,', 'allowing', 'machines', 'to', 'understand,', 'interpret,', 'and', 'manipulate', 'human', 'language.', 'NLP', 'has', 'many', 'real-world', 'applications,', 'including', 'machine', 'translation,', 'sentiment', 'analysis,', 'and', 'chatbots.']
# 2-3. Cleaned tokens: ['natural', 'language', 'processing', 'nlp', 'is', 'a', 'subfield', 'of', 'linguistics', 'computer', 'science', 'and', 'artificial', 'intelligence', 'concerned', 'with', 'the', 'interactions', 'between', 'computers', 'and', 'human', 'language', 'its', 'used', 'to', 'analyze', 'text', 'allowing', 'machines', 'to', 'understand', 'interpret', 'and', 'manipulate', 'human', 'language', 'nlp', 'has', 'many', 'realworld', 'applications', 'including', 'machine', 'translation', 'sentiment', 'analysis', 'and', 'chatbots']
# 4. After stop word removal: ['natural', 'language', 'processing', 'nlp', 'subfield', 'linguistics', 'computer', 'science', 'artificial', 'intelligence', 'concerned', 'interactions', 'computers', 'human', 'language', 'used', 'analyze', 'text', 'allowing', 'machines', 'understand', 'interpret', 'manipulate', 'human', 'language', 'nlp', 'many', 'realworld', 'applications', 'including', 'machine', 'translation', 'sentiment', 'analysis', 'chatbots']
# 5. Stems: ['natural', 'language', 'process', 'nlp', 'subfield', 'linguistics', 'computer', 'science', 'artificial', 'intelligence', 'concern', 'interaction', 'computer', 'human', 'language', 'used', 'analyze', 'text', 'allow', 'machine', 'understand', 'interpret', 'manipulate', 'human', 'language', 'nlp', 'many', 'realworld', 'application', 'includ', 'machine', 'translation', 'sentiment', 'analysis', 'chatbot']
# Bonus. Lemmas: ['natural', 'language', 'process', 'nlp', 'subfield', 'linguistics', 'computer', 'science', 'artificial', 'intelligence', 'concern', 'interaction', 'computer', 'human', 'language', 'use', 'analyze', 'text', 'allow', 'machine', 'understand', 'interpret', 'manipulate', 'human', 'language', 'nlp', 'many', 'realworld', 'application', 'include', 'machine', 'translation', 'sentiment', 'analysis', 'chatbot']
# Words with different results (word, stem, lemma):
# used used use
# including includ include
