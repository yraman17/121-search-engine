from collections import defaultdict

from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer


def tokenize(text: str) -> tuple[dict[str, int], dict[str, list[int]]]:
    # tokenize text into stemmed terms; return token counts and start positions per term
    counts: defaultdict[str, int] = defaultdict(int)
    starts: defaultdict[str, list[int]] = defaultdict(list)

    if not text:
        return counts, starts

    tokenizer = WordPunctTokenizer()
    stemmer = PorterStemmer()

    for start, end in tokenizer.span_tokenize(text):
        raw = text[start:end]
        if not raw or not raw.isalnum() or not raw.isascii():
            continue
        raw = raw.lower()
        stemmed = stemmer.stem(raw)

        counts[stemmed] += 1
        starts[stemmed].append(start)

    return counts, starts
