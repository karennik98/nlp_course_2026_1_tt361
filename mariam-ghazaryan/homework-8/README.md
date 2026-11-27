# Homework 8: LDA Topic Modeling

LDA topic model trained on the first 1000 training documents of **20 Newsgroups** (headers, footers and quotes removed), then used to classify new documents.

## Files

| File | Purpose |
|------|---------|
| `preprocessing.py` | Shared preprocessing: tokenize, lowercase, remove gensim stopwords, keep words with 3+ characters |
| `train_lda.py` | **Part 1.** Builds the dictionary (drops words in < 5 docs or > 50% of docs), BoW corpus, trains LDA (10 topics, 15 passes, `alpha='auto'`, `eta='auto'`), saves to `models/`, prints top 15 words per topic |
| `label_topics.py` | **Part 2.** Shows the top 20 words + probabilities for each topic, asks for a name (Enter keeps `Topic N`), saves `models/topic_labels.json`, prints a summary |
| `inference.py` | **Part 3.** `classify()` returns the top 3 topics for a text. Prints a topic summary on start, then classifies the 5 sample documents (preview, topic names, probabilities, top 5 words per topic) |
| `output_*.txt` | Printed output of each script |

## How to run

Python 3.10–3.12 is recommended (gensim may not have wheels for newer versions).

```bash
pip install -r requirements.txt
```

```bash
python train_lda.py
```

```bash
python label_topics.py
```

```bash
python inference.py
```

## Results

Vocabulary: 18,376 words before filtering, 2,544 after.

| # | Label | Top words |
|---|-------|-----------|
| 0 | Sports (Hockey) | max, period, play, power, vancouver, scoring, chicago, pittsburgh, detroit |
| 1 | Software & Ethics | know, person, people, problem, windows, software, moral, code |
| 2 | Computers & Software | use, problem, program, files, disk, windows, file, help, image |
| 3 | Religion | jesus, god, people, matthew, christian, man |
| 4 | Health & Medicine | health, father, son, spirit, car, medical, insurance |
| 5 | Space & Science | space, nasa, shuttle, data, scsi, chip, mission, launch, orbit, earth |
| 6 | Games & Mixed | good, excellent, new, missing, israel, cover, games |
| 7 | Electronics for Sale | edu, thanks, mail, card, sale, video, interested |
| 8 | Politics & War | armenian, turkish, government, genocide, soviet, war, greek |
| 9 | Philosophy & Debate | argument, true, believe, example, god, truth, conclusion |

### Sample classifications (top topic)

| Sample | Top topic | Probability |
|--------|-----------|-------------|
| 1. Graphics card / GPU | Software & Ethics (2nd: Electronics for Sale) | 42.8% |
| 2. Exoplanet discovery | Games & Mixed | 70.2% |
| 3. Basketball championship | Games & Mixed | 83.1% |
| 4. Congress healthcare bill | Politics & War | 73.9% |
| 5. Italian cooking | Health & Medicine | 47.9% |

Full output, including the 2nd and 3rd topics, is in `output_inference.txt`.

## Discussion

- **Works well:** the politics sample goes clearly to *Politics & War*, and the GPU sample lands in the computer/electronics topics.
- **Sample 2 (space)** missed *Space & Science* because its key words (`exoplanet`, `orbiting`, `habitable`, `planetary`) are not in the vocabulary. Only generic words like `new`, `star`, `research` and `team` were left.
- **Sample 5 (cooking)** has no matching topic in the training data. Only 5 words were known (`love`, `food`, `home`, ...), so the probabilities are spread out and not meaningful.
- **Limits:** 1000 short documents and 10 topics give some mixed topics (e.g. 1 and 6). More documents, lemmatization, or tuning the number of topics with coherence scores would give cleaner topics.
