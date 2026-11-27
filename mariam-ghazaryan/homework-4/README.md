# BBC News classification homework

Mariam Ghazaryan

Open BBC_News_Homework.ipynb for the completed tasks, explanations, saved outputs and analysis.

## Contents

- BBC_News_Homework.ipynb: fully executed solution.
- bbc_classification.py: standalone custom vectorizers and classification experiment.
- bbc-text.csv: unchanged supplied dataset.
- requirements.txt: dependencies.
- results/: cleaned articles, train/test membership, predictions, metrics, confusion matrices and analysis tables.

## Run

From this folder:

```bash
pip install -r requirements.txt
python bbc_classification.py
```

For the notebook, start Jupyter and run its cells in order. NLTK downloads English stopwords on first use if unavailable; this first download requires internet access.

## Measured results

The source contains 2,225 articles. Removing identical preprocessed documents leaves 2,119 articles: 1,695 training and 424 testing. The training vocabulary has 25,566 terms.

| Representation | Test accuracy | Macro F1 |
| --- | --- | --- |
| Custom BoW | 96.93% | 0.9686 |
| Custom TF-IDF with L2 normalization | 96.46% | 0.9644 |

Both use Multinomial Naive Bayes with alpha=1, a stratified 80/20 split and random seed 42. No library vectorizer or TF-IDF transformer is used. TF is raw count; IDF is ln(N/DF) fitted on training documents only. Category averages use unnormalized TF-IDF on training data. The notebook explains the chosen definitions, common/rare word cutoffs, results and limitations.

All hand-calculated vectorizer checks passed, every notebook code cell executed without errors, and all five categories have ten ranked keywords. Performance is from one split; near duplicates and distribution shifts remain limitations.
