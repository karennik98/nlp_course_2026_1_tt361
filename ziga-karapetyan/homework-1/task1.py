text = """Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial
intelligence concerned with the interactions between computers and human language. It's used to
analyze text, allowing machines to understand, interpret, and manipulate human language. NLP has
many real-world applications, including machine translation, sentiment analysis, and chatbots."""

import string

#Split the original text into separate words using spaces
raw_tokens = text.split()
print("Step 1-Tokenization:\n", raw_tokens, "\n")

#Removed punctuation marks from the text before splitting it into clean tokens
translator = str.maketrans("", "", string.punctuation)
cleaned_text = text.translate(translator)

#Converted all words to lowercase and split the text into tokens
tokens = cleaned_text.lower().split()
print("Steps 2 & 3-Lowercasing and Punctuation removal:\n", tokens, "\n")

#Created a set of common words that I want to remove
Stop_Words = {"a", "an", "in", "on", "at", "for", "to", "of", "and", "is", "are", "with", "the"}
    
#removed common stop words from the token list
filter_nlp_words = [nlp for nlp in tokens if nlp not in Stop_Words]
print("Step 4-Stop Word Removal:\n", filter_nlp_words, "\n")

#Removed common suffixes from words to make them shorter
def stemming_func(word):
    roots = ('ation', 'able', 'ment', 'ness', 'tion', 'ing', 'ful', 'ed', 'er', 'ic', 'y')             
    for root in roots:
        if word.endswith(root):
            return word[:-len(root)]

    #Kept the word unchanged if no suffix matches
    return word


#Applied the stemming function to every word
stemmed_words_result = [stemming_func(word) for word in filter_nlp_words]
print("Step 5-Stemming:\n", stemmed_words_result, "\n")

#Used a dictionary to change some words to their base form
lemmatization_dict = {
    "interactions": "interaction",
    "computers": "computer",
    "its": "it",
    "machines": "machine",
    "has": "have",
    "applications": "application",
    "chatbots": "chatbot",
    "is": "be",
    "are": "be",
    "am": "be"
}

#Look for each word in the dictionary
#If the word is found, replace it with its base form
#If it is not found, keep the original word
def lemmatization_func(word):
    return lemmatization_dict.get(word, word)

#Apply lemmatization to the tokens
lemmatized_words_result = [lemmatization_func(word) for word in tokens]
print("Lemmatization:\n", lemmatized_words_result)