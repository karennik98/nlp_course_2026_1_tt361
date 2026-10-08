import re

def validate_ip_addresses(text):
    octet = r'(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])'
    pattern = r'\b' + octet + r'\.' + octet + r'\.' + octet + r'\.' + octet + r'\b'

    ips = re.findall(pattern, text)
    return ips

text = "Valid: 192.168.1.1, 10.0.0.255 Invalid: 256.1.2.3, 192.168.01.1, 192.168.1"
result = validate_ip_addresses(text)
print("Valid IP addresses:")
for ip in result:
    print(ip)