import re

def validate_password(password):
    pattern = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d).{8,}$'
    return bool(re.match(pattern, password))


passwords = ["Password123", "Secure456", "weak", "password", "Password"]
valid_passwords = [pw for pw in passwords if validate_password(pw)]
print("Valid passwords:")
for pw in valid_passwords:
    print(pw)