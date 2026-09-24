import re

def task1():
    text = """Contact us at support@example.com or sales@company.org for assistance. For personal inquiries, email john.doe123@university.edu."""       
    regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(regex, text)
    print("task 1: Extract emails\n")
    for address in emails:
        print(address)

def task2():
    text = "Valid: 123-456-7890, 987-654-3210\nInvalid: 12-345-67890, 1234567890, 123-45-6789"
    regex = r"\b\d{3}-\d{3}-\d{4}\b"
    valid_nums = re.findall(regex, text)
    print("\ntask 2: Validate phone numbers\n")
    print("Valid phone numbers:\n")
    for phone in valid_nums:
        print(phone)

def task3():
    text = "Important dates: 25/12/2023, 01-01-2024, 31/05/2023, and 15-10-2024."
    regex = r"\b\d{2}[/-]\d{2}[/-]\d{4}\b"
    dates = re.findall(regex, text)
    print("\ntask 3: Extract dates\n")
    for date in dates:
        print(date)

def task4():
    text = "The the quick brown fox jumps over the the lazy dog."
    regex = r"\b([a-zA-Z]+)\s+\1\b"
    matches = re.finditer(regex, text, re.IGNORECASE)
    print("\ntask 4: Find repeated words\n")
    print("Repeated words:\n")
    for match in matches:
        print(match.group(0))

def task5():
    text = "Check out our new products: #Sale2024, #NewArrival, and #Discounts!"
    regex = r"#\w+"
    hashtags = re.findall(regex, text)
    print("\ntask 5: Extract hashtags\n")
    for tag in hashtags:
        print(tag)

def task6():
    passwords = ["Password123", "Secure456", "weak", "password", "Password"]
    regex = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$"
    print("\ntask 6: Validate Passwords\n")
    print("Valid passwords:\n")
    for pwd in passwords:
        if re.match(regex, pwd):
            print(pwd)

def task7():
    text = "Visit our website at https://www.example.com or check out http://blog.example.org for updates."
    regex = r"https?://[^\s]+"
    urls = re.findall(regex, text)
    print("\ntask 7: Extract URLs\n")
    for url in urls:
        print(url)

def task8():
    text = """This   
text    
has   
multiple    
spaces."""
    result = re.sub(r"\s+", " ", text).strip()
    print("\ntask 8: Replace spaces\n")
    print(result)

def task9():
    text = 'He said, "Hello, world!" and she replied, "Hi there!"'
    regex = r'"([^"]*)"'
    quoted_text = re.findall(regex, text)
    print("\ntask 9: Extract quoted text\n")
    for quote in quoted_text:
        print(quote)

def task10():
    text = "Valid: 192.168.1.1, 10.0.0.255\nInvalid: 256.1.2.3, 192.168.01.1, 192.168.1"
    ip_octet = r"(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"
    regex = rf"\b(?:{ip_octet}\.){{3}}{ip_octet}\b"
    valid_ips = re.findall(regex, text)
    print("\ntask 10: Validate IP Addresses\n")
    print("Valid IP addresses:\n")
    for ip in valid_ips:
        print(ip)

if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    task7()
    task8()
    task9()
    task10()