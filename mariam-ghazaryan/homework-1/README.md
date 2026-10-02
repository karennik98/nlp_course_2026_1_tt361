# Homework 1 Coachella text preprocessing

Mariam Ghazaryan

Open Coachella_Homework.ipynb to see the completed explanation, code, examples and executed checks.

Files:
- Coachella_Homework.ipynb: completed notebook with saved outputs.
- coachella_preprocessing.py: reusable implementation of all required functions.
- coachella_processed.csv: all original columns plus hashtags, emails and new_tweets_text.
- Coachella-2015-2-DFE.csv: unchanged source dataset.

Run from this folder:

```bash
pip install -r requirements.txt
python coachella_preprocessing.py
```

NLTK downloads its English stopword corpus on first use if it is missing. Internet access is needed for that initial download.

Results: 3,846 tweets; 3,832 tweets with hashtags; 5,475 hashtag occurrences; no email matches; no empty cleaned tweets. The notebook includes a synthetic email example to verify the email function independently.

The original CSV has invalid UTF-8 and Windows-1252 bytes and pre-existing corrupted characters. Latin-1 loading preserves every byte. The output is UTF-8; the new cleaned text is ASCII. No tweets are discarded, and original metadata is retained as strings (including tweet IDs already stored in scientific notation).

