# SMS Spam Classification using Markov Chains and Bayes' Theorem

This project implements an SMS spam classifier from scratch in Python by combining **First-Order Markov Chains** (to model sequential word transitions) with **Bayes' Theorem** (to compute posterior class probabilities).

---

## Mathematical Formulation

Given an SMS message represented as a sequence of $T$ words $M = [s_1, s_2, \dots, s_T]$, we classify the message by comparing the posterior probabilities $\log P(\text{spam} \mid M)$ and $\log P(\text{ham} \mid M)$.

### 1. Bayes' Theorem in Log Space
Using Bayes' theorem and dropping the shared denominator $P(M)$:
$$\log P(c \mid M) \propto \log P(M \mid c) + \log P(c)$$
where $c \in \{\text{spam}, \text{ham}\}$.

### 2. Prior Probabilities
$$\log P(\text{spam}) = \log\left(\frac{N_{\text{spam}}}{N_{\text{total}}}\right), \quad \log P(\text{ham}) = \log\left(\frac{N_{\text{ham}}}{N_{\text{total}}}\right)$$

### 3. Markov Chain Likelihood
Assuming the Markov property (each word depends only on the immediately preceding word), the log-likelihood of a message sequence is:
$$\log P(M \mid c) = \log \hat{\pi}_{s_1}^{(c)} + \sum_{t=2}^{T} \log \hat{A}_{s_{t-1}, s_t}^{(c)}$$

### 4. Parameter Estimation with Additive Smoothing
To prevent zero probabilities for unseen words or word transitions, Laplace/Lidstone smoothing ($\epsilon$) is applied across vocabulary size $M$:

* **Initial State Probabilities:**
  $$\hat{\pi}_i = \frac{\text{count}(s_1 = i) + \epsilon}{N_c + \epsilon M}$$

* **Transition Probabilities:**
  $$\hat{A}_{ij} = \frac{\text{count}(i \rightarrow j) + \epsilon}{\text{count}(i) + \epsilon M}$$
  *(Note: $\text{count}(i)$ represents the number of times word $i$ appears as a prefix to another word in class $c$).*

---

## Project Structure

* `spam_classifier.py`: Complete, dependency-free Python implementation of data preprocessing, train/test splitting, Markov-Bayes model training, and evaluation.
* `spam.csv`: Input dataset containing `label` (`spam` or `ham`) and `message` columns.
* `README.md`: Documentation and theoretical overview.

---

## Usage

### Prerequisites
* Python 3.7+ (Uses only standard library modules: `csv`, `math`, `random`, `re`, `collections`).

### Running the Classifier
1. Ensure `spam.csv` is located in the same directory as `spam_classifier.py`.
2. Run the script from your terminal:
   ```bash
   python spam_classifier.py
   ```

### Expected Output
The script will load and clean the dataset, perform an 80/20 train-test split, train the Markov models, and output evaluation metrics (Accuracy, Precision, Recall, F1-Score, and Confusion Matrix) along with predictions on sample SMS messages.