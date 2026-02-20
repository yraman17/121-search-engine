from collections import defaultdict

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer

def tokenize(text):
    counts = defaultdict(int)
    starts = defaultdict(list)

    if not text:
        return counts, starts

    for (start, end) in WordPunctTokenizer().span_tokenize(text):
        raw = text[start:end]
        if not raw or not raw.isalnum():
            continue
        raw = raw.lower()
        stemmed = PorterStemmer().stem(raw)

        counts[stemmed] += 1
        starts[stemmed].append(start)

    return counts, starts
