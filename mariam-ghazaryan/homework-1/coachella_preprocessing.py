"""Task 1: extract metadata and preprocess the Coachella tweets."""
import argparse
import csv
import json
import re
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# Download only the corpus used by this assignment, if absent.
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True, raise_on_error=True)
    STOP_WORDS = set(stopwords.words('english'))
TOKENIZER = TweetTokenizer()
HASHTAG_PATTERN = re.compile(r'(?<!\w)#[\w]+', re.UNICODE)
EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+")


def extract_hashtags(text):
    """Retain the #, spelling, order and repeated occurrences."""
    return HASHTAG_PATTERN.findall(text)


def extract_emails(text):
    return EMAIL_PATTERN.findall(text)


def remove_usernames(text):
    """Remove mentions without treating an email domain as a mention."""
    return re.sub(r'(?<![\w@.+-])@[A-Za-z0-9_]+', ' ', text)


def remove_links(text):
    return re.sub(r'https?://\S+', ' ', text, flags=re.IGNORECASE)


def remove_non_ascii_symbols(text):
    return text.encode('ascii', errors='ignore').decode('ascii')


def to_lower(text):
    return text.lower()


def remove_stop_words(text):
    """TweetTokenizer needs no Punkt download and handles punctuation."""
    return ' '.join(token for token in TOKENIZER.tokenize(text)
                    if token.lower() not in STOP_WORDS)


def remove_digits(text):
    return re.sub(r'\d+', '', text)


def remove_special_characters(text):
    # Use spaces so punctuation-separated words do not become one word.
    return ' '.join(re.sub(r'[^a-zA-Z\s]', ' ', text).split())


def preprocess_text(text):
    text = '' if text is None else str(text)
    for function in (remove_usernames, remove_links, remove_non_ascii_symbols,
                     to_lower, remove_stop_words, remove_digits,
                     remove_special_characters):
        text = function(text)
    # Punctuation removal can expose stopwords (e.g. parts of contractions).
    return remove_stop_words(text)


def process_dataset(input_path, output_path):
    # This supplied file is not valid UTF-8 or Windows-1252. Latin-1 is
    # a lossless one-byte mapping; existing mojibake is removed by ASCII cleaning.
    with open(input_path, encoding='latin-1', newline='') as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if not columns or 'text' not in columns:
            raise ValueError("The CSV must contain a 'text' column.")
        rows = list(reader)
    for row in rows:
        original = row['text'] or ''
        row['hashtags'] = json.dumps(extract_hashtags(original), ensure_ascii=False)
        row['emails'] = json.dumps(extract_emails(original), ensure_ascii=False)
        row['new_tweets_text'] = preprocess_text(original)
    with open(output_path, 'w', encoding='utf-8', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=columns +
                                ['hashtags', 'emails', 'new_tweets_text'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(rows):
    return {
        'tweets': len(rows),
        'tweets_with_hashtags': sum(bool(json.loads(r['hashtags'])) for r in rows),
        'hashtag_occurrences': sum(len(json.loads(r['hashtags'])) for r in rows),
        'tweets_with_emails': sum(bool(json.loads(r['emails'])) for r in rows),
        'email_occurrences': sum(len(json.loads(r['emails'])) for r in rows),
        'empty_cleaned_tweets': sum(not r['new_tweets_text'] for r in rows),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', default='Coachella-2015-2-DFE.csv')
    parser.add_argument('--output', default='coachella_processed.csv')
    args = parser.parse_args()
    print(json.dumps(summarize(process_dataset(args.input, args.output)), indent=2))
