import re

def extract_hashtags(text):
    pattern = r'#\w+'
    hashtags = re.findall(pattern, text)
    return hashtags

text = "Check out our new products: #Sale2024, #NewArrival, and #Discounts!"
result = extract_hashtags(text)
for tag in result:
    print(tag)