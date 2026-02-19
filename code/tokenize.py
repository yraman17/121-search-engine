from collections import defaultdict

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer

def tokenize(text: str):
    counts = defaultdict(int)
    starts = defaultdict(list)

    stemmer = PorterStemmer()
    tokenizer = WordPunctTokenizer()
    spans = tokenizer.span_tokenize(text)
    tokens = nltk.word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens]
    for token, span in zip(stemmed, spans):
        counts[token] += 1
        starts[token].append(span[0])
    