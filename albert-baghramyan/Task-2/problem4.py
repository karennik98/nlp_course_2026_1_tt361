import re

def find_repeated_words(text):
    pattern = r'\b(\w+)\s+\1\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return matches

text = "The the quick brown fox jumps over the the lazy dog."
result = find_repeated_words(text)
print("Repeated words:", end=" ")
for word in result:
    print(word, word, end=" ")
print()