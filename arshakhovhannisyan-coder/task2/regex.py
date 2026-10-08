import re


print("Խնդիր 1")

text = "Contact us at support@example.com or sales@company.org. For personal inquiries, email john.doe123@university.edu."

emails = re.findall(r'[\w.-]+@[\w.-]+\.\w+', text)

for email in emails:
    print(email)


print("\nԽնդիր 2")

text = "Valid: 123-456-7890, 987-654-3210. Invalid: 12-345-67890, 1234567890."

phones = re.findall(r'\b\d{3}-\d{3}-\d{4}\b', text)

for phone in phones:
    print(phone)


print("\nԽնդիր 3")

text = "Important dates: 25/12/2023, 01-01-2024, 31/05/2023, 15-10-2024."

dates = re.findall(r'\d{2}[/-]\d{2}[/-]\d{4}', text)

for date in dates:
    print(date)


print("\nԽնդիր 4")

text = "The the quick brown fox jumps over the the lazy dog."

repeated_words = re.findall(r'\b(\w+)\s+\1\b', text, re.IGNORECASE)

for word in repeated_words:
    print(word)


print("\nԽնդիր 5")

text = "Check out our new products: #Sale2024, #NewArrival, and #Discounts!"

hashtags = re.findall(r'#\w+', text)

for hashtag in hashtags:
    print(hashtag)


print("\nԽնդիր 6")

passwords = [
    "Password123",
    "Secure456",
    "weak",
    "password",
    "Password"
]

for password in passwords:
    if len(password) >= 8 and re.search(r'[A-Z]', password) and re.search(r'[a-z]', password) and re.search(r'\d', password):
        print(password)


print("\nԽնդիր 7")

text = "Visit our website at https://www.example.com or http://blog.example.org for updates."

urls = re.findall(r'https?://\S+', text)

for url in urls:
    print(url)


print("\nԽնդիր 8")

text = "This text has    multiple     spaces."

new_text = re.sub(r'\s+', ' ', text)

print(new_text)


print("\nԽնդիր 9")

text = 'He said, "Hello, world!" and she replied, "Hi there!"'

quoted_text = re.findall(r'"(.*?)"', text)

for sentence in quoted_text:
    print(sentence)


print("\nԽնդիր 10")

text = "Valid: 192.168.1.1, 10.0.0.255. Invalid: 256.1.2.3, 192.168.01.1, 192.168.1."

possible_ips = re.findall(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', text)

for ip in possible_ips:
    numbers = ip.split(".")

    valid = True

    for number in numbers:
        if int(number) > 255:
            valid = False

        if len(number) > 1 and number[0] == "0":
            valid = False

    if valid:
        print(ip)
