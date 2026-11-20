import random
import re
from collections import defaultdict

def read_poem(path="robert_frost.txt"):
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                text = re.sub(r"[^\w\s']", "", raw.lower()).strip()
                words = [w.strip("'") for w in text.split()]
                words = [w for w in words if w]
                if words:
                    lines.append(words)
    except FileNotFoundError:
        print(f"Սխալ: '{path}' ֆայլը չի գտնվել")
        raise SystemExit(1)
    except Exception as e:
        print(f"Ֆայլը կարդալիս սխալ: {e}")
        raise SystemExit(1)
    return lines

def train(lines):
    initial_c = defaultdict(int)
    first_c = defaultdict(lambda: defaultdict(int))
    second_c = defaultdict(lambda: defaultdict(int))

    for words in lines:
        words = words + ["END"]
        initial_c[words[0]] += 1
        for i in range(len(words) - 1):
            first_c[words[i]][words[i + 1]] += 1
        for i in range(len(words) - 2):
            second_c[(words[i], words[i + 1])][words[i + 2]] += 1

    total_lines = len(lines)
    initial = {w: c / total_lines for w, c in initial_c.items()}
    first = {w: {n: c / sum(d.values()) for n, c in d.items()} for w, d in first_c.items()}
    second = {k: {n: c / sum(d.values()) for n, c in d.items()} for k, d in second_c.items()}
    return initial, first, second

def pick_cumulative(dist):
    r = random.random()
    cumulative = 0.0
    for word, p in dist.items():
        cumulative += p
        if r < cumulative:
            return word
    return list(dist)[-1]

def pick_deterministic(dist):
    return max(dist, key=dist.get)

def generate_line(initial, first, second, pick, max_words=15):
    w1 = pick(initial)
    line = [w1]
    if w1 not in first:
        return " ".join(line)
    w2 = pick(first[w1])
    if w2 == "END":
        return " ".join(line)
    line.append(w2)

    while len(line) < max_words:
        key = (line[-2], line[-1])
        if key in second:
            nxt = pick(second[key])
        else:
            nxt = pick(first[line[-1]])
        if nxt == "END":
            break
        line.append(nxt)
    return " ".join(line)

if __name__ == "__main__":
    lines = read_poem("robert_frost.txt")
    initial, first, second = train(lines)

    print("Cumulative (պատահական)")
    for _ in range(4):
        print(generate_line(initial, first, second, pick_cumulative))

    print("\nDeterministic (ամենամեծ հավանականությամբ)")
    for _ in range(4):
        print(generate_line(initial, first, second, pick_deterministic))
