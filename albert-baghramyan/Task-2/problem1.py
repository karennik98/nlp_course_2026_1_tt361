import re

def extract_emails(text):
    pattern = r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}'
    emails = re.findall(pattern, text)
    return emails

text = """Contact us at support@example.com or sales@company.org for assistance.
For personal inquiries, email john.doe123@university.edu."""
result = extract_emails(text)
for email in result:
    print(email)