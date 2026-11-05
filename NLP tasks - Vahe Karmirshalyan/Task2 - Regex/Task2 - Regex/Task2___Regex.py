
import re

def extract_emails(text):
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    matches = re.findall(pattern, text)
    return matches

def Validate_PhoneNumber(phone_number):
    pattern = r'^\d{3}-\d{3}-\d{4}$'
    return re.match(pattern, phone_number) is not None

def Extract_URLs(text):
    pattern = r'https?://[^\s]+'
    matches = re.findall(pattern, text)
    return matches

def extract_dates(text):
    pattern = r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'
    matches = re.findall(pattern, text)
    return matches

def find_repeated_words(text):
    pattern = r'\b(\w+)\s+\1\b'
    matches = re.finditer(pattern, text, re.IGNORECASE)
    return [match.group() for match in matches]

def extract_hashtags(text):
    pattern = r'#\w+'
    matches = re.findall(pattern, text)
    return matches

def validate_password(password):
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$'
    return re.match(pattern, password) is not None

def replace_multiple_spaces(text):
    pattern = r' +'
    result = re.sub(pattern, ' ', text)
    return result

def extract_quoted_text(text):
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, text)
    return matches

def validate_ip_address(ip):
    pattern = r'^(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}$'
    return re.match(pattern, ip) is not None



test = "Contact us at support@example.com or sales@company.org for assistance."
print(extract_emails(test))


test_number = "123-456-788890"
print(Validate_PhoneNumber(test_number))

test_date = "The event is scheduled for 12/31/2023 and 01-01-2024."
print(extract_dates(test_date))

test_urls = "Visit our website at https://www.example.com or check out http://blog.example.org for updates."
print(Extract_URLs(test_urls))

test_repeated = "The the quick brown fox jumps over the the lazy dog."
print(find_repeated_words(test_repeated))

test_hashtags = "Check out our new products: #Sale2024, #NewArrival, and #Discounts!"
print(extract_hashtags(test_hashtags))

test_password = "Password123"
print(validate_password(test_password))

test_spaces = "This    text   has    multiple     spaces."
print(replace_multiple_spaces(test_spaces))

test_quotes = 'He said, "Hello, world!" and she replied, "Hi there!"'
print(extract_quoted_text(test_quotes))

test_ip = "192.168.1.1"
print(validate_ip_address(test_ip))