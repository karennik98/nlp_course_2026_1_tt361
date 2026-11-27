"""Custom BoW and TF-IDF for BBC News classification.

Only the split, classifier and evaluation use scikit-learn. No library text
vectorizer or TF-IDF transformer is used. Run from this script's directory.
"""
from collections import Counter
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True, raise_on_error=True)
    STOP_WORDS = set(stopwords.words('english'))


def preprocess(text):
    """Lowercase, replace punctuation/digits with spaces, tokenize, filter."""
    return [word for word in re.findall(r'[a-z]+', str(text).lower())
            if word not in STOP_WORDS]


def build_vocabulary(documents):
    """Assign deterministic column indices to training words only."""
    return {word: i for i, word in enumerate(sorted(
        {word for document in documents for word in document}))}


def document_to_bow(tokens, vocabulary):
    """One dense count vector; unseen words are ignored."""
    vector = np.zeros(len(vocabulary), dtype=np.int32)
    for word, count in Counter(tokens).items():
        if word in vocabulary:
            vector[vocabulary[word]] = count
    return vector


def bow_matrix(documents, vocabulary):
    """Construct a sparse matrix directly from custom counts."""
    values, columns, offsets = [], [], [0]
    for tokens in documents:
        for word, count in sorted(Counter(tokens).items()):
            if word in vocabulary:
                columns.append(vocabulary[word])
                values.append(count)
        offsets.append(len(values))
    return csr_matrix((np.array(values, dtype=np.float64), columns, offsets),
                      shape=(len(documents), len(vocabulary)))


def calculate_tf(tokens):
    """TF is raw count, matching the core worked examples in the PDF."""
    return dict(Counter(tokens))


def calculate_idf(training_documents, vocabulary):
    """IDF(t) = ln(N / DF(t)); a document counts each term once."""
    document_frequency = np.zeros(len(vocabulary), dtype=np.int32)
    for tokens in training_documents:
        for word in set(tokens):
            if word in vocabulary:
                document_frequency[vocabulary[word]] += 1
    if len(training_documents) == 0 or np.any(document_frequency == 0):
        raise ValueError('Vocabulary must come from a nonempty training corpus.')
    return np.log(len(training_documents) / document_frequency), document_frequency


def document_to_tfidf(tokens, vocabulary, idf):
    """Unnormalized TF-IDF vector: raw term count times training IDF."""
    vector = np.zeros(len(vocabulary), dtype=float)
    for word, frequency in calculate_tf(tokens).items():
        if word in vocabulary:
            index = vocabulary[word]
            vector[index] = frequency * idf[index]
    return vector


def tfidf_matrix(counts, idf, normalize=False):
    """Multiply nonzero custom counts by IDF; optionally L2-normalize."""
    result = counts.copy().astype(float)
    result.data *= idf[result.indices]
    result.eliminate_zeros()
    if normalize:
        norms = np.sqrt(np.asarray(result.multiply(result).sum(axis=1)).ravel())
        for row in range(result.shape[0]):
            if norms[row] > 0:
                start, end = result.indptr[row:row + 2]
                result.data[start:end] /= norms[row]
    return result


def verify_vectorizers():
    """Hand-calculated fixture checks counts, DF, IDF, OOV and empty text."""
    docs = [['cat', 'cat', 'dog'], ['dog', 'bird'], []]
    vocabulary = build_vocabulary(docs)
    assert list(vocabulary) == ['bird', 'cat', 'dog']
    assert calculate_tf(docs[0]) == {'cat': 2, 'dog': 1}
    assert document_to_bow(docs[0], vocabulary).tolist() == [0, 2, 1]
    idf, df = calculate_idf(docs, vocabulary)
    np.testing.assert_array_equal(df, [1, 1, 2])
    np.testing.assert_allclose(idf, [np.log(3), np.log(3), np.log(1.5)])
    counts = bow_matrix(docs, vocabulary)
    np.testing.assert_array_equal(counts.toarray(), [[0, 2, 1], [1, 0, 1], [0, 0, 0]])
    vectors = tfidf_matrix(counts, idf)
    np.testing.assert_allclose(vectors[0].toarray().ravel(), [0, 2*np.log(3), np.log(1.5)])
    np.testing.assert_allclose(document_to_tfidf(docs[0], vocabulary, idf), vectors[0].toarray().ravel())
    assert not document_to_bow(['unseen'], vocabulary).any()
    normalized = tfidf_matrix(counts, idf, normalize=True)
    np.testing.assert_allclose(np.asarray(normalized.multiply(normalized).sum(axis=1)).ravel(), [1, 1, 0])
    assert preprocess('The CAT, and dog! 123') == ['cat', 'dog']
    print('All hand-calculated vectorizer checks passed.')


def run_experiment(data_path='bbc-text.csv', output_dir='results'):
    verify_vectorizers()
    output = Path(output_dir)
    output.mkdir(exist_ok=True, parents=True)
    raw = pd.read_csv(data_path)
    if raw[['text', 'category']].isna().any().any():
        raise ValueError('Missing texts or labels must be handled before training.')
    data = raw.copy()
    data['source_row'] = np.arange(len(data))
    # Deduplicate the actual preprocessed input to prevent exact overlap.
    data['tokens'] = data.text.map(preprocess)
    data['clean_text'] = data.tokens.map(' '.join)
    if data.groupby('clean_text').category.nunique().max() > 1:
        raise ValueError('Identical cleaned documents have conflicting labels.')
    data = data.drop_duplicates('clean_text').reset_index(drop=True)
    train, test = train_test_split(data, test_size=0.2, random_state=42,
                                   stratify=data.category)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    assert set(train.clean_text).isdisjoint(test.clean_text)
    vocabulary = build_vocabulary(train.tokens)
    words = np.array(list(vocabulary))
    idf, df = calculate_idf(train.tokens.tolist(), vocabulary)
    train_bow = bow_matrix(train.tokens, vocabulary)
    test_bow = bow_matrix(test.tokens, vocabulary)
    # Unnormalized weights are used for interpretable category averages.
    raw_tfidf = tfidf_matrix(train_bow, idf)
    train_tfidf = tfidf_matrix(train_bow, idf, normalize=True)
    test_tfidf = tfidf_matrix(test_bow, idf, normalize=True)
    scores, reports, predictions = [], {}, test[['source_row', 'category']].copy()
    labels = sorted(data.category.unique())
    for name, x_train, x_test in [('BoW', train_bow, test_bow),
                                  ('TF-IDF', train_tfidf, test_tfidf)]:
        model = MultinomialNB(alpha=1.0).fit(x_train, train.category)
        predicted = model.predict(x_test)
        scores.append({'representation': name, 'accuracy': accuracy_score(test.category, predicted),
                       'macro_f1': f1_score(test.category, predicted, average='macro')})
        reports[name] = classification_report(test.category, predicted, output_dict=True, zero_division=0)
        predictions[name] = predicted
        pd.DataFrame(confusion_matrix(test.category, predicted, labels=labels),
                     index=labels, columns=labels).to_csv(output/f'{name.lower().replace("-", "_")}_confusion_matrix.csv')
    top_rows = []
    for category in labels:
        averages = np.asarray(raw_tfidf[train.category.eq(category).to_numpy()].mean(axis=0)).ravel()
        for rank, index in enumerate(np.argsort(-averages, kind='stable')[:10], 1):
            top_rows.append({'category': category, 'rank': rank, 'word': words[index],
                             'average_tfidf': averages[index]})
    top_words = pd.DataFrame(top_rows)
    # Mean raw TF over ALL training documents, including absent terms as zero.
    stats = pd.DataFrame({'word': words, 'mean_tf': np.asarray(train_bow.mean(axis=0)).ravel(),
                          'document_frequency': df, 'idf': idf})
    # State explicit relative cutoffs rather than calling arbitrary words high/low.
    low_idf = stats.idf.quantile(0.10)
    high_idf = stats.idf.quantile(0.90)
    common = stats[stats.idf <= low_idf].sort_values(['mean_tf', 'word'], ascending=[False, True]).head(10)
    rare = stats[stats.idf >= high_idf].sort_values(['mean_tf', 'word']).head(10)
    results = pd.DataFrame(scores)
    summary = {'source_rows': len(raw), 'exact_raw_duplicates': int(raw.text.duplicated().sum()),
               'duplicates_removed_after_preprocessing': len(raw)-len(data), 'unique_documents': len(data),
               'train_documents': len(train), 'test_documents': len(test), 'vocabulary_size': len(vocabulary),
               'seed': 42, 'idf_low_cutoff': low_idf, 'idf_high_cutoff': high_idf,
               'test_oov_token_count': sum(w not in vocabulary for doc in test.tokens for w in doc),
               'test_token_count': sum(map(len, test.tokens))}
    results.to_csv(output/'classification_metrics.csv', index=False)
    top_words.to_csv(output/'top_category_words.csv', index=False)
    stats.to_csv(output/'training_word_statistics.csv', index=False)
    common.to_csv(output/'high_tf_low_idf.csv', index=False)
    rare.to_csv(output/'low_tf_high_idf.csv', index=False)
    predictions.to_csv(output/'test_predictions.csv', index=False)
    data[['source_row','category','clean_text']].to_csv(output/'preprocessed_articles.csv',index=False)
    pd.concat([train[['source_row','category']].assign(split='train'),
               test[['source_row','category']].assign(split='test')]).to_csv(output/'split_manifest.csv',index=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    (output/'classification_reports.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps(summary, indent=2))
    print('\nClassification results\n', results.to_string(index=False))
    return summary, results, top_words, common, rare, reports


if __name__ == '__main__':
    run_experiment()
