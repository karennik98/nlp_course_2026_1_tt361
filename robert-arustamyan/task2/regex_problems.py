import re


def problem_1():  # emails: local part @ domain . tld
    text = (
        "Contact us at support@example.com or sales@company.org for assistance.\n"
        "For personal inquiries, email john.doe123@university.edu."
    )
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    for email in re.findall(pattern, text):
        print(email)


def problem_2():  # phones: \b...\b so only exact XXX-XXX-XXXX matches
    text = (
        "Valid: 123-456-7890, 987-654-3210\n"
        "Invalid: 12-345-67890, 1234567890, 123-45-6789"
    )
    pattern = r"\b\d{3}-\d{3}-\d{4}\b"
    print("Valid phone numbers:")
    for phone in re.findall(pattern, text):
        print(phone)


def problem_3():  # dates: \2 forces the same separator (/ or -) both times
    text = "Important dates: 25/12/2023, 01-01-2024, 31/05/2023, and 15-10-2024."
    pattern = r"\b(\d{2}([/-])\d{2}\2\d{4})\b"
    for date, _separator in re.findall(pattern, text):
        print(date)


def problem_4():  # repeated words: \1 backreference, case-insensitive
    text = "The the quick brown fox jumps over the the lazy dog."
    pattern = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
    print("Repeated words:")
    for match in pattern.finditer(text):
        print(match.group(0).lower())


def problem_5():  # hashtags: # followed by word characters
    text = "Check out our new products: #Sale2024, #NewArrival, and #Discounts!"
    pattern = r"#\w+"
    for hashtag in re.findall(pattern, text):
        print(hashtag)


def problem_6():  # passwords: lookaheads for upper/lower/digit + length >= 8
    passwords = ["Password123", "Secure456", "weak", "password", "Password"]
    pattern = re.compile(r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d).{8,}$")
    print("Valid passwords:")
    for password in passwords:
        if pattern.match(password):
            print(password)


def problem_7():  # urls: http(s):// up to whitespace, without trailing punctuation
    text = (
        "Visit our website at https://www.example.com or check out "
        "http://blog.example.org for updates."
    )
    pattern = r"https?://[^\s]+[^\s.,!?;:]"
    for url in re.findall(pattern, text):
        print(url)


def problem_8():  # collapse 2+ spaces into one with re.sub
    text = "This    text   has     multiple      spaces."
    print(re.sub(r" {2,}", " ", text))


def problem_9():  # quoted text: capture everything between two double quotes
    text = 'He said, "Hello, world!" and she replied, "Hi there!"'
    pattern = r'"([^"]*)"'
    for quoted in re.findall(pattern, text):
        print(quoted)


def problem_10():  # ips: each octet 0-255 without leading zeros, lookarounds reject partial matches
    text = (
        "Valid: 192.168.1.1, 10.0.0.255\n"
        "Invalid: 256.1.2.3, 192.168.01.1, 192.168.1"
    )
    octet = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d|\d)"
    pattern = rf"(?<![\d.]){octet}(?:\.{octet}){{3}}(?![\d]|\.\d)"
    print("Valid IP addresses:")
    for ip in re.findall(pattern, text):
        print(ip)


def main():  # run all 10 problems
    problems = [
        problem_1, problem_2, problem_3, problem_4, problem_5,
        problem_6, problem_7, problem_8, problem_9, problem_10,
    ]
    for number, problem in enumerate(problems, start=1):
        print(f"=== Problem {number} ===")
        problem()
        print()


if __name__ == "__main__":
    main()
