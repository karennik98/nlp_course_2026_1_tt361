"""Homework 2: 10 RegEx problems - Mariam Ghazaryan.

Each problem uses the sample text from the assignment and prints a label
before its result so the output can be compared with the expected output.
"""
import re


def problem_1_emails():
    """Extract all email addresses."""
    text = ("Contact us at support@example.com or sales@company.org for assistance.\n"
            "For personal inquiries, email john.doe123@university.edu.")
    # local part, "@", domain labels, then a top-level domain of 2+ letters.
    # The TLD must be letters, so the sentence's final period is not included.
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
    print("Problem 1 - Email addresses:")
    for email in re.findall(pattern, text):
        print(email)


def problem_2_phone_numbers():
    """Keep only phone numbers in the format XXX-XXX-XXXX."""
    text = ("Valid: 123-456-7890, 987-654-3210\n"
            "Invalid: 12-345-67890, 1234567890, 123-45-6789")
    # The lookarounds stop a match from starting or ending inside a longer
    # number, e.g. the "345-6789" part of other formats.
    pattern = r"(?<![\d-])\d{3}-\d{3}-\d{4}(?![\d-])"
    print("Problem 2 - Valid phone numbers:")
    for phone in re.findall(pattern, text):
        print(phone)


def problem_3_dates():
    """Extract dates written as DD/MM/YYYY or DD-MM-YYYY."""
    text = "Important dates: 25/12/2023, 01-01-2024, 31/05/2023, and 15-10-2024."
    # ([/-]) captures the first separator and \1 requires the same one again,
    # so a mixed form such as 25/12-2023 is rejected.
    pattern = r"\b\d{2}([/-])\d{2}\1\d{4}\b"
    print("Problem 3 - Dates:")
    for match in re.finditer(pattern, text):
        print(match.group())


def problem_4_repeated_words():
    """Find a word immediately repeated, e.g. "the the"."""
    text = "The the quick brown fox jumps over the the lazy dog."
    # (\w+) captures a word and \1 matches the same word after whitespace.
    # IGNORECASE lets "The the" count as a repetition.
    pattern = r"\b(\w+)\s+\1\b"
    print("Problem 4 - Repeated words:")
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        print(match.group().lower())


def problem_5_hashtags():
    """Extract hashtags (words starting with #)."""
    text = "Check out our new products: #Sale2024, #NewArrival, and #Discounts!"
    pattern = r"#\w+"
    print("Problem 5 - Hashtags:")
    for tag in re.findall(pattern, text):
        print(tag)


def problem_6_passwords():
    """Keep passwords with 8+ characters, an uppercase, a lowercase and a digit."""
    passwords = ["Password123", "Secure456", "weak", "password", "Password"]
    # Each (?=...) lookahead checks one rule without consuming characters;
    # .{8,} then checks the length over the whole string.
    pattern = r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d).{8,}$"
    print("Problem 6 - Valid passwords:")
    for password in passwords:
        if re.fullmatch(pattern, password):
            print(password)


def problem_7_urls():
    """Extract URLs that start with http:// or https://."""
    text = ("Visit our website at https://www.example.com or check out "
            "http://blog.example.org for updates.")
    # https? makes the "s" optional; \S+ takes everything up to a space.
    # The final character must not be punctuation, so a trailing "." is dropped.
    pattern = r"https?://\S+?(?=[.,!?]?(?:\s|$))"
    print("Problem 7 - URLs:")
    for url in re.findall(pattern, text):
        print(url)


def problem_8_multiple_spaces():
    """Replace runs of spaces with a single space."""
    text = "This    text   has     multiple      spaces."
    # " {2,}" matches two or more spaces; each run becomes one space.
    print("Problem 8 - Single spaces:")
    print(re.sub(r" {2,}", " ", text))


def problem_9_quoted_text():
    """Extract the text inside double quotes."""
    text = 'He said, "Hello, world!" and she replied, "Hi there!"'
    # [^"]* is the text between the quotes; the group returns it without them.
    pattern = r'"([^"]*)"'
    print("Problem 9 - Quoted text:")
    for quote in re.findall(pattern, text):
        print(quote)


def problem_10_ip_addresses():
    """Keep IPv4 addresses whose four parts are 0-255 without leading zeros."""
    text = ("Valid: 192.168.1.1, 10.0.0.255\n"
            "Invalid: 256.1.2.3, 192.168.01.1, 192.168.1")
    # One octet: 250-255 | 200-249 | 100-199 | 0-99 (no leading zero).
    octet = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    # The lookarounds reject addresses that are part of a longer number,
    # such as "56.1.2.3" inside 256.1.2.3 or "1.1" inside 192.168.01.1.
    pattern = rf"(?<![\d.]){octet}(?:\.{octet}){{3}}(?![\d.]*\d)"
    print("Problem 10 - Valid IP addresses:")
    for ip in re.findall(pattern, text):
        print(ip)


if __name__ == "__main__":
    problems = [
        problem_1_emails, problem_2_phone_numbers, problem_3_dates,
        problem_4_repeated_words, problem_5_hashtags, problem_6_passwords,
        problem_7_urls, problem_8_multiple_spaces, problem_9_quoted_text,
        problem_10_ip_addresses,
    ]
    for solve in problems:
        solve()
        print()
