import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

CSV_PATH = "/home/albert/Downloads/Coachella-2015-2-DFE.csv"
TEXT_COLUMN = None

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
print("Stop words loaded:", len(STOP_WORDS))

if not os.path.exists(CSV_PATH):
print("File not found:", CSV_PATH)
raise SystemExit(1)

try:
df = pd.read_csv(CSV_PATH)
except UnicodeDecodeError:
df = pd.read_csv(CSV_PATH, encoding="latin-1")

print("Loaded rows:", len(df))
print("Columns:", list(df.columns))

if TEXT_COLUMN is None:
for candidate in ["text", "tweet_text", "tweet", "Tweet", "Text", "content"]:
if candidate in df.columns:
TEXT_COLUMN = candidate
break
else:
text_cols = df.select_dtypes(include="object").columns
TEXT_COLUMN = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())

print("Using text column:", TEXT_COLUMN)
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

df["hashtags"] = df[TEXT_COLUMN].apply(lambda t: re.findall(r"#\w+", t))
df["emails"] = df[TEXT_COLUMN].apply(
lambda t: re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)
)
print("Tweets with hashtags:", (df["hashtags"].str.len() > 0).sum())
print("Tweets with emails:", (df["emails"].str.len() > 0).sum())


def remove_usernames(text):
return re.sub(r"(?<!\w)@\w+", "", text)


def remove_links(text):
return re.sub(r"https?://\S+", "", text)


def remove_non_ascii_symbols(text):
return text.encode("ascii", errors="ignore").decode("ascii")


def to_lower(text):
return text.lower()


def remove_stop_words(text):
return " ".join(w for w in text.split() if w not in STOP_WORDS)


def remove_digits(text):
return re.sub(r"\d+", "", text)


def remove_special_characters(text):
return re.sub(r"[^a-zA-Z\s]", "", text)


df["clean_text"] = df[TEXT_COLUMN]
for func in [
remove_usernames,
remove_links,
remove_non_ascii_symbols,
to_lower,
remove_digits,
remove_special_characters,
remove_stop_words,
]:
df["clean_text"] = df["clean_text"].apply(func)
print("Applied:", func.name)

for i in range(min(5, len(df))):
print(f"\nTweet #{i + 1}")
print("Original:", df[TEXT_COLUMN].iloc[i])
print("Cleaned :", df["clean_text"].iloc[i])
print("Hashtags:", df["hashtags"].iloc[i])
print("Emails :", df["emails"].iloc[i])

output_path = os.path.join(os.path.dirname(CSV_PATH), "coachella_tweets_cleaned.csv")
df.to_csv(output_path, index=False)
print("\nSaved to:", output_path)