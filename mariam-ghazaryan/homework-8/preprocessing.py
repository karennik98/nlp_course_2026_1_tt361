"""Shared text preprocessing used by both training and inference."""

from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess


def preprocess(text):
    """Tokenize, lowercase, drop gensim stopwords and words shorter than 3 chars."""
    # simple_preprocess tokenizes, lowercases and strips punctuation/numbers
    tokens = simple_preprocess(text, deacc=True, min_len=3, max_len=30)
    return [t for t in tokens if t not in STOPWORDS]
