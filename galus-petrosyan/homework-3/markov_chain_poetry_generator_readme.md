# Markov Chain Poetry Generator

A Python implementation of a **second-order Markov Chain** poetry generator trained on the works of Robert Frost (starting with *"The Road Not Taken"*). The program learns statistical word transition patterns from a source text file and generates new, original stanzas in a similar poetic style.

---

## Features

- **Three-Tier Probability Model:**
  1. **Initial Word Probabilities** $P(w_1)$: Tracks the likelihood of each word starting a line.
  2. **First-Order Transitions** $P(w_t \mid w_{t-1})$: Tracks the likelihood of a word following a single preceding word.
  3. **Second-Order Transitions** $P(w_t \mid w_{t-2}, w_{t-1})$: Tracks the likelihood of a word following a two-word sequence, including line-ending `'END'` tokens.
- **Text Preprocessing:** Automatically strips punctuation, normalizes text to lowercase, and processes lines individually to preserve poetic line structure.
- **Cumulative Probability Sampling:** Uses a number-line cumulative distribution approach to sample words proportionally to their learned probabilities, producing varied outputs on each run.
- **Graceful Fallback & Error Handling:** Safely handles missing or unreadable files and falls back to lower-order distributions if an unseen word pair is encountered.

---

## Mathematical Overview

During the **Training Phase**, the model calculates probabilities across all non-empty lines in the corpus:

- **Initial Probability:**
  $$P(w_1) = \frac{\text{count}(w_1 \text{ as first word})}{\text{total number of lines}}$$

- **First-Order Transition Probability:**
  $$P(w_t \mid w_{t-1}) = \frac{\text{count}(w_{t-1}, w_t)}{\text{count}(w_{t-1})}$$

- **Second-Order Transition Probability:**
  $$P(w_t \mid w_{t-2}, w_{t-1}) = \frac{\text{count}(w_{t-2}, w_{t-1}, w_t)}{\text{count}(w_{t-2}, w_{t-1})}$$

During the **Generation Phase**, each line is constructed step-by-step:
1. Sample $w_1$ from $P(w_1)$.
2. Sample $w_2$ from $P(w_2 \mid w_1)$.
3. Iteratively sample $w_t$ from $P(w_t \mid w_{t-2}, w_{t-1})$ using cumulative probability intervals until the `'END'` token is selected.

---

## Project Structure

```text
.
├── markov_poetry.py      # Main Python script containing MarkovPoetryGenerator
├── robert_frost.txt      # Training corpus of Robert Frost poems
└── README.md             # Project documentation
```

---

## Getting Started

### Prerequisites

- **Python 3.9+** (uses only standard library modules: `random`, `re`, and `collections`)

### Installation & Usage

1. Clone or download the project files into a single directory.
2. Ensure your training text file (e.g., `robert_frost.txt`) is placed in the same directory as the script.
3. Update the `filename` variable in `markov_poetry.py` if your text file has a different name:
   ```python
   filename = "robert_frost.txt"
   ```
4. Run the generator from your terminal:
   ```bash
   python markov_poetry.py
   ```

---

## Example Output

Because word selection relies on cumulative probability sampling, each execution produces a unique 4-line poem:

```text
Գեներացված 4 տողանոց բանաստեղծությունը․

Two roads diverged in a yellow wood
And sorry i could not travel both
To watch his woods fill up with snow
And miles to go before i sleep
```

---

## Customization

- **Change Poem Length:** Adjust `num_lines` in `generator.generate_poem(num_lines=4)` to generate longer or shorter stanzas.
- **Prevent Runaway Lines:** Adjust `max_words` in `generate_line(max_words=25)` to cap the maximum number of words per generated line.
- **Train on Other Authors:** Replace `robert_frost.txt` with any plain-text poetry collection to generate verses in a different style.