import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Որպեսզի թվիթները ամբողջությամբ երևան, ոչ թե կտրված "..."-ով
pd.set_option("display.max_colwidth", None)

raw_df = pd.read_csv("Coachella-2015-2-DFE.csv", encoding="latin-1")


print(raw_df.shape)          # քանի տող, քանի սյունակ
print(raw_df.columns)        # սյունակների անունները
raw_df.info()                # տիպերը և քանի դատարկ արժեք կա
raw_df.head()                # առաջին 5 տողը



for t in raw_df["text"].sample(10, random_state=42):
    print(t)
    print("-" * 80)


# ============================================================
# STEP 1: copy only the column we actually need from raw_df
# ============================================================
df = raw_df[["text"]].copy()

print("\n=== STEP 1: df with only the 'text' column ===")
print(df.shape)
print(df.columns)
print(df.head())


# ============================================================
# STEP 2: extract hashtags into a new column, on the copy
# ============================================================
HASHTAG_RE = re.compile(r"#\w+")

df["hashtags"] = df["text"].astype(str).apply(HASHTAG_RE.findall)

print("\n=== STEP 2: hashtags extracted ===")
print(df[["text", "hashtags"]].sample(5, random_state=42).to_string())


# ============================================================
# STEP 3: extract emails into a new column, on the copy
# ============================================================
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

df["emails"] = df["text"].astype(str).apply(EMAIL_RE.findall)

print("\n=== STEP 3: emails extracted ===")
print(df[["text", "emails"]].sample(5, random_state=42).to_string())


# ============================================================
# STEP 4: cleaning functions, each one small and single-purpose
# ============================================================
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def remove_usernames(text):
    """Remove @mentions, e.g. @coachella, @beyonce."""
    return re.sub(r"@\w+", "", text)


def remove_links(text):
    """Remove http/https links."""
    return re.sub(r"https?://\S+", "", text)


def remove_non_ascii_symbols(text):
    """Remove emojis, accented letters, and other non-ASCII characters."""
    return text.encode("ascii", "ignore").decode("ascii")


def to_lower(text):
    """Lowercase all letters."""
    return text.lower()


def remove_stop_words(text):
    """Remove common English stop words (the, a, an, and, ...)."""
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


def remove_digits(text):
    """Remove all numeric digits."""
    return re.sub(r"\d+", "", text)


def remove_special_characters(text):
    """Remove punctuation and other non-alphanumeric, non-space characters."""
    return re.sub(r"[^a-zA-Z\s]", "", text)


# ============================================================
# STEP 5: build the new "clean_tweet" column on the copy, one
# function at a time, in the order the task asks for
# ============================================================
df["clean_tweet"] = df["text"].astype(str)
df["clean_tweet"] = df["clean_tweet"].apply(remove_usernames)
df["clean_tweet"] = df["clean_tweet"].apply(remove_links)
df["clean_tweet"] = df["clean_tweet"].apply(remove_non_ascii_symbols)
df["clean_tweet"] = df["clean_tweet"].apply(to_lower)
df["clean_tweet"] = df["clean_tweet"].apply(remove_stop_words)
df["clean_tweet"] = df["clean_tweet"].apply(remove_digits)
df["clean_tweet"] = df["clean_tweet"].apply(remove_special_characters)
df["clean_tweet"] = df["clean_tweet"].str.replace(r"\s+", " ", regex=True).str.strip()

print("\n=== STEP 5: final df (copy) with all new columns ===")
print(df.columns.tolist())
print(df.sample(10, random_state=42).to_string())
