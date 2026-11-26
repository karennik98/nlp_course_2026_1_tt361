a = ("Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
     "concerned with the interactions between computers and human language. It`s used to analyze text, allowing "
     "machines to understand, interpret, and manipulate human language. NLP has many real-world applications, "
     "including machine translation, sentiment analysis, and chatbots")

tokens = a.split()
print(tokens)

tokens = [word.lower() for word in tokens]
print(tokens)

punct = " <,>.?/:;|{[}()*&^%$#@!]+=_-`~' "
tokens = [word.strip(punct) for word in tokens]
print(tokens)

stop = {
    "the", "is", "a", "an", "and", "or", "of", "to", "in",
    "with", "between", "it", "it`s", "has", "many", "used",
    "for", "from", "by", "on", "as", "that", "this"
        }
tokens = [word for word in tokens if word not in stop]
print(tokens)

def simple(word):
    verj = ["ingly", "edly", "ing", "ed", "ly", "es", "s", "`s"]

    for v in verj:
        if word.endswith(v) and len(word) > len(v) + 2:
            return word[:-len(v)]

    return word

tokens = [simple(word) for word in tokens]
print(tokens)
