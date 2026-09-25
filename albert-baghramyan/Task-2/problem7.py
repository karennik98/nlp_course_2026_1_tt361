import re

def extract_urls(text):
    pattern = r'https?://[^\s,]+'
    urls = re.findall(pattern, text)
    return urls

text = "Visit our website at https://www.example.com or check out http://blog.example.org for updates."
result = extract_urls(text)
for url in result:
    print(url)