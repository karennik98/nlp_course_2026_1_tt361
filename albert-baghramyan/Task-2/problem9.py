import re

def extract_quoted_text(text):
    pattern = r'"([^"]*)"'
    quotes = re.findall(pattern, text)
    return quotes

text = 'He said, "Hello, world!" and she replied, "Hi there!"'
result = extract_quoted_text(text)
for quote in result:
    print(quote)