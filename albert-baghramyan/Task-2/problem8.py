import re

def normalize_spaces(text):
    pattern = r' +'
    normalized = re.sub(pattern, ' ', text)
    return normalized

text = "This    text   has     multiple  spaces."
result = normalize_spaces(text)
print(result)