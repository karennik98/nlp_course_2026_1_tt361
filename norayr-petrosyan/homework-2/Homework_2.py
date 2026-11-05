import re

#1. Email addresses
text = ("Contact us at support@example.com or sales@company.org for assistance. "
        "For personal inquiries, email john.doe123@university.edu.")
for email in re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}", text):
    print(email)

#2. Phone numbers
numbers = ["123-456-7890", "987-654-3210", "12-345-67890", "1234567890", "123-45-6789"]
print("\nValid phone numbers:")
for number in numbers:
    if re.fullmatch(r"\d{3}-\d{3}-\d{4}", number):
        print(number)

#3. Dates: the second separator must match the first one.
text = "Important dates: 25/12/2023, 01-01-2024, 31/05/2023, and 15-10-2024."
print()
for match in re.finditer(r"\b\d{2}([/-])\d{2}\1\d{4}\b", text):
    print(match.group())

#4. Consecutive repeated words
text = "The the quick brown fox jumps over the the lazy dog."
print("\nRepeated words:")
for match in re.finditer(r"\b(\w+)\s+\1\b", text, re.IGNORECASE):
    print(match.group().lower())

#5. Hashtags
text = "Check out our new products: #Sale2024, #NewArrival, and #Discounts!"
print()
for tag in re.findall(r"#\w+", text):
    print(tag)

#6. Password strength
passwords = ["Password123", "Secure456", "weak", "password", "Password"]
print("\nValid passwords:")
for password in passwords:
    if (len(password) >= 8 and re.search(r"[A-Z]", password)
            and re.search(r"[a-z]", password) and re.search(r"\d", password)):
        print(password)

#7. URLs
text = "Visit our website at https://www.example.com or check out http://blog.example.org for updates."
print()
for url in re.findall(r'https?://[^\s<>"]+', text):
    print(url.rstrip(".,!?"))

#8. Multiple spaces
text = "This    text  has   multiple    spaces."
print()
print(re.sub(r" {2,}", " ", text))

#9. Text inside double quotes
text = 'He said, "Hello, world!" and she replied, "Hi there!"'
print()
for quote in re.findall(r'"([^"]*)"', text):
    print(quote)

#10. IP addresses: each part is 0 to 255, with no leading zeros.
addresses = ["192.168.1.1", "10.0.0.255", "256.1.2.3", "192.168.01.1", "192.168.1"]
part = r"(?:0|[1-9][0-9]{0,2})"
print("\nValid IP addresses:")
for address in addresses:
    if re.fullmatch(rf"{part}(?:\.{part}){{3}}", address):
        if all(int(value) <= 255 for value in address.split(".")):
            print(address)
