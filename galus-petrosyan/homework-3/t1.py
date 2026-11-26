import random
import re
from collections import defaultdict


class MarkovPoetryGenerator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Երեք հիմնական հավանականային բառարանները
        self.initial_probs: dict[str, float] = {}
        self.first_order_probs: dict[str, dict[str, float]] = {}
        self.second_order_probs: dict[tuple[str, str], dict[str, float]] = {}

    def _read_lines(self) -> list[str]:
        """Կարդում է ֆայլը տող առ տող և մշակում հնարավոր սխալները:"""
        lines = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                for line in file:
                    cleaned_line = line.strip()
                    if cleaned_line:
                        lines.append(cleaned_line)
        except FileNotFoundError:
            print(f"Սխալ․ '{self.file_path}' ֆայլը չի գտնվել։")
            return []
        except OSError as e:
            print(f"Սխալ ֆայլը կարդալիս․ {e}")
            return []
        return lines

    @staticmethod
    def _preprocess_line(line: str) -> list[str]:
        """Հեռացնում է կետադրական նշանները, բերում փոքրատառի և բաժանում բառերի:"""
        no_punct = re.sub(r"[^\w\s]", "", line.lower())
        return no_punct.split()

    @staticmethod
    def _counts_to_probs(counts: dict[str, int]) -> dict[str, float]:
        """Հաշվում է P(word) = count(word) / total_count հավանականությունները:"""
        total = sum(counts.values())
        if total == 0:
            return {}
        return {word: count / total for word, count in counts.items()}

    def train(self) -> bool:
        """Ուսուցման փուլ․ կառուցում է նախնական, առաջին և երկրորդ կարգի հավանականությունները[cite: 1]:"""
        raw_lines = self._read_lines()
        if not raw_lines:
            return False

        initial_counts: dict[str, int] = defaultdict(int)
        first_order_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        second_order_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

        total_lines = 0

        for raw_line in raw_lines:
            words = self._preprocess_line(raw_line)
            if not words:
                continue

            total_lines += 1

            # A. Առաջին բառի հաշվարկ՝ տողի սկզբում[cite: 1]
            w1 = words[0]
            initial_counts[w1] += 1

            # Եթե տողն ունի ընդամենը 1 բառ, միանգամից անցնում է END թոքենին[cite: 1]
            if len(words) == 1:
                first_order_counts[w1]["END"] += 1
                continue

            # B. Առաջին կարգի անցումներ (w1 -> w2)[cite: 1]
            w2 = words[1]
            first_order_counts[w1][w2] += 1

            if len(words) == 2:
                second_order_counts[(w1, w2)]["END"] += 1
                continue

            # C. Երկրորդ կարգի անցումներ (w_{t-2}, w_{t-1} -> w_t)[cite: 1]
            for i in range(2, len(words)):
                prev_pair = (words[i - 2], words[i - 1])
                curr_word = words[i]
                second_order_counts[prev_pair][curr_word] += 1

            # Տողի ավարտը նշվում է 'END' թոքենով[cite: 1]
            last_pair = (words[-2], words[-1])
            second_order_counts[last_pair]["END"] += 1

        if total_lines == 0:
            return False

        # Պահպանում ենք հավանականությունները երեք բառարաններում[cite: 1]
        self.initial_probs = {
            word: count / total_lines for word, count in initial_counts.items()
        }
        self.first_order_probs = {
            prev_word: self._counts_to_probs(next_counts)
            for prev_word, next_counts in first_order_counts.items()
        }
        self.second_order_probs = {
            prev_pair: self._counts_to_probs(next_counts)
            for prev_pair, next_counts in second_order_counts.items()
        }

        return True

    @staticmethod
    def _select_cumulative(prob_dist: dict[str, float]) -> str:
        """Ընտրում է բառ՝ օգտագործելով կուտակային հավանականության մեթոդը[cite: 1]:"""
        if not prob_dist:
            return "END"

        r = random.random()  # Պատահական թիվ [0.0, 1.0) միջակայքում[cite: 1]
        cumulative = 0.0
        last_word = ""

        for word, prob in prob_dist.items():
            cumulative += prob
            last_word = word
            if r < cumulative:
                return word

        return last_word

    def _fallback_next_word(self, word: str) -> str:
        """Պահուստային ընտրություն, եթե տվյալ բառը կամ զույգը չի հանդիպել որպես նախորդող[cite: 1]:"""
        if word in self.first_order_probs:
            return self._select_cumulative(self.first_order_probs[word])
        return self._select_cumulative(self.initial_probs)

    def generate_line(self, max_words: int = 25) -> str:
        """Գեներացնում է մեկ տող՝ մինչև 'END' թոքենի ընտրվելը[cite: 1]:"""
        if not self.initial_probs:
            return ""

        # 1. Ընտրում ենք առաջին բառը (w1) նախնական հավանականություններով[cite: 1]
        w1 = self._select_cumulative(self.initial_probs)
        if w1 == "END":
            return ""

        line_words = [w1]

        # 2. Ընտրում ենք երկրորդ բառը (w2) առաջին կարգի անցումներով[cite: 1]
        if w1 in self.first_order_probs:
            w2 = self._select_cumulative(self.first_order_probs[w1])
        else:
            w2 = self._select_cumulative(self.initial_probs)

        if w2 == "END":
            return " ".join(line_words).capitalize()

        line_words.append(w2)

        # 3. Հաջորդ բառերը (wt) ընտրում ենք երկրորդ կարգի անցումներով մինչև 'END'[cite: 1]
        while len(line_words) < max_words:
            prev_pair = (line_words[-2], line_words[-1])
            if prev_pair in self.second_order_probs:
                next_word = self._select_cumulative(self.second_order_probs[prev_pair])
            else:
                next_word = self._fallback_next_word(line_words[-1])

            if next_word == "END":
                break

            line_words.append(next_word)

        return " ".join(line_words).capitalize()

    def generate_poem(self, num_lines: int = 4) -> list[str]:
        """Գեներացնում է նշված քանակությամբ տողերից բաղկացած բանաստեղծություն[cite: 1]:"""
        poem = []
        while len(poem) < num_lines:
            line = self.generate_line()
            if line:
                poem.append(line)
        return poem


if __name__ == "__main__":
    # Տեքստային ֆայլի անունը (կարող եք փոխել 'robert_frost.txt'-ի, եթե ֆայլն այդպես է կոչվում)[cite: 1, 2]
    filename = "robert_frost.txt"

    generator = MarkovPoetryGenerator(filename)
    if generator.train():
        poem_lines = generator.generate_poem(num_lines=4)
        print("\nԳեներացված 4 տողանոց բանաստեղծությունը․\n")
        for line in poem_lines:
            print(line)