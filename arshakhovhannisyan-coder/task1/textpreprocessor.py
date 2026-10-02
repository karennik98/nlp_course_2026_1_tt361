import string


class Preprocessor:
    def __init__(self, text=""):
        self.text = text
        self.stopwords = {
            "the", "a", "an", "in", "on", "at", "for",
            "to", "of", "and", "is", "are"
        }

    def tokenization(self, text=None):
        if text is None:
            text = self.text
        return text.split()

    def lowercasing(self, text=None):
        if text is None:
            text = self.text

        if isinstance(text, str):
            tokens = self.tokenization(text)
        else:
            tokens = text

        return [token.lower() for token in tokens]

    def remove_punctuation(self, text=None):
        if text is None:
            text = self.text

        tokens = self.lowercasing(text)
        result = []

        for token in tokens:
            new_token = ""

            for char in token:
                if char not in string.punctuation:
                    new_token += char

            if new_token:
                result.append(new_token)

        return result

    def remove_stopwords(self, text=None):
        if text is None:
            text = self.text

        if isinstance(text, str):
            tokens = self.remove_punctuation(text)
        else:
            tokens = text

        return [token for token in tokens if token not in self.stopwords]

    def stemming(self, text=None):
        if text is None:
            text = self.text

        tokens = self.remove_stopwords(text)
        stemmed_text = []

        for token in tokens:
            if token.endswith("ing"):
                token = token[:-3]
            elif token.endswith("ed"):
                token = token[:-2]
            elif token.endswith("ly"):
                token = token[:-2]
            elif token.endswith("s") and len(token) > 3:
                token = token[:-1]

            stemmed_text.append(token)

        return stemmed_text

    def lemmatization(self, text=None):
        if text is None:
            text = self.text

        tokens = self.stemming(text)

        lemmatization_vocab = {
            "used": "use",
            "is": "be",
            "are": "be",
            "was": "be",
            "has": "have",
            "had": "have"
        }

        return [lemmatization_vocab.get(token, token) for token in tokens]


if __name__ == "__main__":
    example_text = """Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. It's used to analyze text, allowing machines to understand, interpret, and manipulate human language. NLP has many real-world applications, including machine translation, sentiment analysis, and chatbots."""

    text_preprocessor = Preprocessor(example_text)

    tokenized_text = text_preprocessor.tokenization()
    lowcased_text = text_preprocessor.lowercasing(tokenized_text)
    removed_punctuation_text = text_preprocessor.remove_punctuation(lowcased_text)
    removed_stopwords_text = text_preprocessor.remove_stopwords(removed_punctuation_text)
    stemmed_text = text_preprocessor.stemming(removed_stopwords_text)
    lemmatized_text = text_preprocessor.lemmatization(stemmed_text)

    print("Քայլ 1")
    print(tokenized_text)
    print()

    print("Քայլ 2")
    print(lowcased_text)
    print()

    print("Քայլ 3")
    print(removed_punctuation_text)
    print()

    print("Քայլ 4")
    print(removed_stopwords_text)
    print()

    print("Քայլ 5")
    print(stemmed_text)
    print()

    print("Bonus")
    print(lemmatized_text)
