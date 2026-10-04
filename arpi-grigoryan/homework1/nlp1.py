class TextPreprocessor: 
    def __init__(self, text): 
        # Store the original text inside the object 
        self.text = text 
 
        # Words that we consider unnecessary/common in the text. 
        self.stop_words = [ 
            "the", "a", "an", "in", "on", "at", 
            "for", "to", "of", "and", "is", "are" 
        ] 
 
        # Punctuation marks that we want to remove from the text. 
        self.punctuation = [ 
            ".", ",", "!", "?", "-", ":", 
            ";", "(", ")", "'" 
        ] 
 
        # Common suffixes used for simple stemming. 
        self.suffixes = [ 
            "ness", "ment", "tion", "able", "ible", 
            "ful", "less", "ous", "ive", "ing", 
            "est", "es", "ed", "ly", "er", "s", "y" 
        ] 
 
        # A small manually-created dictionary for lemmatization. 
        # Unlike stemming, lemmatization tries to convert 
        # a word to its meaningful base/dictionary form. 
        self.lemma_dictionary = { 
            "is": "be", 
            "are": "be", 
            "am": "be", 
            "was": "be", 
            "were": "be", 
            "has": "have", 
            "had": "have", 
            "having": "have", 
            "went": "go", 
            "going": "go", 
            "ran": "run", 
            "running": "run", 
            "better": "good", 
            "best": "good", 
            "children": "child", 
            "machines": "machine", 
            "computers": "computer", 
            "applications": "application" 
        } 
 
 
    def tokenize(self): 
        # TOKENIZATION 
        # Split the text into individual pieces (tokens). 
        # .split() separates the text wherever there is whitespace. 
        self.tokens = self.text.split() 
        print("Step 1 - Tokenization:") 
        print(self.tokens) 
        print() 
 
        return self.tokens 
 
 
    def lowercase(self): 
        # LOWERCASE 
        # Convert every token to lowercase. 
        for i in range(len(self.tokens)): 
            self.tokens[i] = self.tokens[i].lower() 
 
        print("Step 2 - Lowercasing:") 
        print(self.tokens) 
        print() 
 
        return self.tokens 
 
 
    def remove_punctuation(self): 
        # PUNCTUATION REMOVAL 
        # Go through every token and remove punctuation marks. 
        for i in range(len(self.tokens)): 
            for mark in self.punctuation: 
                self.tokens[i] = self.tokens[i].replace(mark, "") 
 
        print("Step 3 - Punctuation Removal:") 
        print(self.tokens) 
        print() 
 
        return self.tokens 
 
 
    def remove_stop_words(self): 
        # STOP-WORD REMOVAL 
        # Remove common words that usually don't carry much 
        # useful information for some NLP tasks. 
        self.tokens = [ 
            word for word in self.tokens 
            if word not in self.stop_words 
        ] 
 
        print("Step 4 - Stop Word Removal:") 
        print(self.tokens) 
        print() 
 
        return self.tokens 
 
    def stem(self): 
        # STEMMING 
        # Remove common suffixes from words to get a simpler word form. 
        for i in range(len(self.tokens)): 
            # Check every suffix in our suffix list. 
            for suffix in self.suffixes: 
 
                # Check whether the word ends with this suffix. 
                if self.tokens[i].endswith(suffix): 
                    # Make sure that removing the suffix does not 
                    # make the word too short. 
                    if len(self.tokens[i]) > len(suffix) + 2: 
                        # Remove the suffix. 
                        self.tokens[i] = self.tokens[i][:-len(suffix)] 
                     # Stop checking suffixes for this word. 
                    # We only remove ONE suffix. 
                    break 
 
        print("Step 5 - Stemming:") 
        print(self.tokens) 
        print() 
        return self.tokens 
 
 
    def lemmatize(self): 
        # LEMMATIZATION 
        # Convert words to their base/dictionary form. 
        for i in range(len(self.tokens)): 
            # Check whether the current word exists 
            # in our lemma dictionary. 
            if self.tokens[i] in self.lemma_dictionary: 
                # Replace the current word with its lemma. 
                self.tokens[i] = self.lemma_dictionary[self.tokens[i]] 
 
                print("Lemmatization:") 
                print(self.tokens) 
                print() 
 
                return self.tokens 
 
text = """ Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial  
intelligence concerned with the interactions between computers and human language. It's used to  
analyze text, allowing machines to understand, interpret, and manipulate human language. NLP has  
many real-world applications, including machine translation, sentiment analysis, and chatbots. """ 
 
 
# Create an object of our TextPreprocessor class 
preprocessor = TextPreprocessor(text) 
 
## 1. Split the text into individual tokens 
preprocessor.tokenize() 
 
# 2. Convert all tokens to lowercase 
preprocessor.lowercase() 
 
# 3. Remove punctuation marks 
preprocessor.remove_punctuation() 
 
# 4. Remove common/unnecessary stop words 
preprocessor.remove_stop_words() 
 
# 5. Convert words to their dictionary/base forms 
preprocessor.lemmatize() 
#output 
# ['natural', 'language', 'processing', 'nlp', 'subfield', 'linguistics', 'computer',  
#  'science', 'artificial', 'intelligence', 'concerned', 'with',  
#  'interactions', 'between', 'computer', 'human', 'language', 'its',  
#  'used', 'analyze', 'text', 'allowing', 'machines', 'understand',  
#  'interpret', 'manipulate', 'human', 'language', 'nlp', 'has', 'many', 
#  'realworld', 'applications', 'including', 'machine', 'translation', 
#  'sentiment', 'analysis', 'chatbots'] 
 
# 6. Remove common suffixes to create simplified word forms 
preprocessor.stem() 
#output 
# ['natural', 'language', 'process', 'nlp', 'subfield',  
#  'linguistic', 'comput', 'science', 'artificial', 'intelligence', 
#  'concern', 'with', 'interaction', 'between', 'computer', 'human', 
#  'language', 'its', 'used', 'analyze', 'text', 'allow', 'machin',  
#  'understand', 'interpret', 'manipulate', 'human', 'language', 'nlp', 
#  'has', 'man', 'realworld', 'application', 'includ', 'machine', 'transla', 
#  'senti', 'analysi', 'chatbot'] 





     
  

  




    