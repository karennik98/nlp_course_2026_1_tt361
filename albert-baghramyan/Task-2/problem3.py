import re

def extract_dates(text):
    pattern = r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'
    dates = re.findall(pattern, text)
    return dates

text = "Important dates: 25/12/2023, 01-01-2024, 31/05/2023, and 15-10-2024."
result = extract_dates(text)
for date in result:
    print(date)