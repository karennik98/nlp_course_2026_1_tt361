# Preprocessor class for text preprocessing tasks such as tokenization, lowercasing, punctuation removal, stopword removal, stemming, and lemmatization.

class Preprocessor:
    def __init__(self, text=""):
        self.text = text
        # Initialize the stopwords, apostrophes vocabulary, stemming set, and lemmatization vocabulary.
        # You can expand these sets and dictionaries with more words as needed.
        self.stopwords =  {"the","a","an","in","on","at","for","to","of","and","is","are"}
        # Initialize the apostrophes vocabulary with common contractions and their expanded forms. You can expand this dictionary with more contractions as needed.
        self.apostrophesvocab = {"'s" : "is", "'ve" : "have", "'re" : "are", "'ll" : "will", "'d" : "had", "'m" : "am", "n't" : "not"}
        # Initialize the stemming set with common suffixes. You can expand this set with more suffixes as needed.
        self.stemmingset = {'ingly', 'edly', 'ation', 'ness', 'able', 'ible', 'less', 'ship','ing', 'ion', 'est', 'ful', 'ous', 'ity', 'ive', 'ies', 'ied', 'ed', 'es', 'ly', 'er', 'or', 's'}
        self.stemmingset = sorted(self.stemmingset, key=lambda x: len(x), reverse=True)  # Sort the suffixes by length in descending order for proper stemming.
        # Initialize the lemmatization vocabulary (like irregular verbs, plural forms, etc.) with common irregular forms and their base forms. You can expand this dictionary with more words as needed.
        self.lemmatizationvocab = {
            'used': 'use',
            'running': 'run',
            'ran': 'run',
            'better': 'good',
            'best': 'good',
            'worse': 'bad',
            'worst': 'bad',
            'children': 'child',
            'mice': 'mouse',
            'geese': 'goose',
            'feet': 'foot',
            'teeth': 'tooth',
            'men': 'man',
            'women': 'woman',
            'am': 'be',
            'is': 'be',
            'are': 'be',
            'was': 'be',
            'were': 'be',
            'has': 'have',
            'had': 'have',
            'does': 'do',
            'did': 'do',
        }
    # Tokenization method to split the text into tokens based on whitespace.
    def tokenization(self, text=None):
        if text is None:
            text = self.text
        return text.split()
    
    # Lowercasing method to convert all tokens to lowercase.
    def lowercasing(self, text=None):
        if text is None:
            text = self.text
        if type(text) is str:
            tokens = self.tokenization(text)
        else:
            tokens = text
        lowcased_text = [token.lower() for token in tokens]
        return lowcased_text
    
    # List check method to verify if the input is a list. If it is, return it; otherwise, return an empty list.
    def listcheck(self, text=None):
        if text is None:
            text = self.text
        if type(text) is list:
            return text
        else:
            return self.lowercasing(self.tokenization(text))
    
    # Remove punctuation method to clean tokens by removing punctuation and handling contractions based on the apostrophes vocabulary.
    def remove_punctuation(self, text=None):
        if text is None:
            text = self.text
        tokens = self.listcheck(text)
        remove_punctuation_text = []
        for token in tokens:
            # Handle contractions and punctuation removal
            if "'" in token:
                i = token.index("'")
                if token[i:] in self.apostrophesvocab:
                    remove_punctuation_text.append(token[:i])
                    remove_punctuation_text.append(self.apostrophesvocab[token[i:]])
                else:
                    remove_punctuation_text.append(token)
            elif token.isalnum():
                remove_punctuation_text.append(token)
            else:
                newtoken = ""
                for char in token:
                    if char.isalnum():
                        newtoken += char
                    else:
                        if newtoken:
                            remove_punctuation_text.append(newtoken)
                            newtoken = ""
                if newtoken:
                    remove_punctuation_text.append(newtoken)           
        return remove_punctuation_text
    
    # Remove stopwords method to filter out common words that do not contribute much to the meaning of the text.
    def remove_stopwords(self, text=None):
        if text is None:
            text = self.text
        tokens = self.listcheck(text)
        remove_stopwords_text = [token for token in tokens if token not in self.stopwords]
        return remove_stopwords_text
    
    # Stemming method to reduce words to their root form.
    def stemming(self, text=None):
        if text is None:
            text = self.text
        tokens = self.listcheck(text)
        stemmed_text = []
        for token in tokens:
            stemmed_text.append(token)
            for suffix in self.stemmingset:
                # Check if the token ends with the suffix and has more than 2 characters before the suffix to avoid over-stemming.
                if token.endswith(suffix) and len(token) - len(suffix) > 2:
                    stemmed_text[-1] = token[:-len(suffix)]
                    break
        return stemmed_text
    
    # Lemmatization method to reduce words to their base or dictionary form using the lemmatization vocabulary.
    def lemmatization(self, text=None):
        if text is None:
            text = self.text
        tokens = self.listcheck(text)
        lemmatized_text = [self.lemmatizationvocab.get(token, token) for token in tokens]
        return lemmatized_text


# Example usage of the Preprocessor class with a sample text.
if __name__ == "__main__":
    example_text = """Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial
    intelligence concerned with the interactions between computers and human language. It's used to
    analyze text, allowing machines to understand, interpret, and manipulate human language. NLP has
    many real-world applications, including machine translation, sentiment analysis, and chatbots.
    """
    text_preprocessor = Preprocessor(example_text)
    tokenized_text = text_preprocessor.tokenization()
    lowcased_text = text_preprocessor.lowercasing(tokenized_text)
    removed_punctuation_text = text_preprocessor.remove_punctuation(lowcased_text)
    removed_stopwords_text = text_preprocessor.remove_stopwords(removed_punctuation_text)
    stemmed_text = text_preprocessor.stemming(removed_stopwords_text)
    lemmatized_text = text_preprocessor.lemmatization(removed_stopwords_text)
    print(":: Step 1 Tokenization :: \n-------------------------------\n", tokenized_text, "\n")
    print(":: Step 2 Lowercasing :: \n-------------------------------\n", lowcased_text, "\n")
    print(":: Step 3 Remove Punctuation :: \n-------------------------------\n", removed_punctuation_text, "\n")
    print(":: Step 4 Remove Stopwords :: \n-------------------------------\n", removed_stopwords_text, "\n")
    print(":: Final Step 5 Stemming :: \n-------------------------------\n", stemmed_text, "\n")
    print(":: Bonus Step 6 Lemmatization :: \n-------------------------------\n", lemmatized_text, "\n")