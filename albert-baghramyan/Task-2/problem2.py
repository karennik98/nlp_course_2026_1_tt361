import re

def validate_phone_numbers(text):
    pattern = r'\b\d{3}-\d{3}-\d{4}\b'
    valid_numbers = re.findall(pattern, text)
    return valid_numbers

text = """Valid: 123-456-7890, 987-654-3210
Invalid: 12-345-67890, 1234567890, 123-45-6789"""
result = validate_phone_numbers(text)

print("Valid phone numbers:")
for number in result:
    print(number)