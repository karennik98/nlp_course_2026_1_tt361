import csv
import re
from html import unescape
from pathlib import Path

folder = Path(__file__).resolve().parent
source = next(folder.glob("Coachella-2015-2-DFE*.csv"))
output = folder / "coachella_cleaned.csv"

stop_words = set("""
a an the and or but if as at by for from in into of on to with about between
i me my mine we us our ours you your yours he him his she her hers it its
they them their theirs this that these those am is are was were be been being
have has had do does did can could will would should shall may might
so than then there here when where who whom which what how why all any both
each few more most other some such only own same too very just it's i'm
you're we're they're i've i'll that's there's rt
""".split())


def remove_usernames(text):
    return re.sub(r"(?<![\w.+-])@\w+", " ", text)


def remove_links(text):
    return re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)


def remove_non_ascii_symbols(text):
    return text.encode("ascii", errors="ignore").decode("ascii")


def to_lower(text):
    return text.lower()


def remove_stop_words(text):
    words = re.findall(r"[a-z]+(?:'[a-z]+)*", text)
    return " ".join(word for word in words if word not in stop_words)


def remove_digits(text):
    return re.sub(r"\d+", " ", text)


def remove_special_characters(text):
    return " ".join(re.sub(r"[^a-zA-Z0-9\s]", " ", text).split())


with source.open(encoding="latin-1", newline="") as infile, \
        output.open("w", encoding="utf-8", newline="") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames + ["hashtags", "emails", "cleaned_text"])
    writer.writeheader()

    for row in reader:
        text = row["text"]
        
        row["hashtags"] = ", ".join(re.findall(r"(?<!\w)#\w+", text))
        row["emails"] = ", ".join(re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}", text))

        cleaned = remove_links(unescape(text))
        cleaned = remove_usernames(cleaned)
        cleaned = remove_non_ascii_symbols(cleaned)
        cleaned = to_lower(cleaned)
        cleaned = remove_digits(cleaned)
        cleaned = remove_stop_words(cleaned)
        row["cleaned_text"] = remove_special_characters(cleaned)
        writer.writerow(row)
