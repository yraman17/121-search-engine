import bisect
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

from lib.index import Importance

# suppress BeautifulSoup XML and URL warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")


def extract_text(html_text: str) -> tuple[str, str]:
    # extract plain text from HTML; return (body text, important text from title/headings/bold)
    if not html_text:
        return "", []

    soup = BeautifulSoup(html_text, "lxml")
    IMPORTANT_TAGS = {"h1", "h2", "h3", "b", "strong", "title"}

    text_chunks = []
    spans = []
    offset = 0

    # iterate through all string elements
    for element in soup.find_all(string=True):
        text = element.strip()
        if not text:
            continue

        # relate this element to its nearest higher importance
        importance = Importance.NORMAL
        for parent in element.parents:
            if parent.name in IMPORTANT_TAGS:
                importance = (
                    Importance.TITLE
                    if parent.name == "title"
                    else Importance.BOLD_OR_HEADING
                )
                break

        # adjust offset and store start and end of chunk so importance can be assigned to individual tokens
        start = offset
        end = offset + len(text)
        spans.append((start, end, importance))
        text_chunks.append(text)
        offset = end + 1
    # reconstruct full text
    full_text = " ".join(text_chunks)
    return full_text, spans


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


def assign_importance(
    starts: dict[str, list[int]], spans: list[tuple[int, int, Importance]]
) -> dict[str, list[int, Importance]]:
    # get where each chunk in spans starts
    span_starts = [span[0] for span in spans]
    # dict tokens to list of starts with importances
    token_importance = {}

    for token, positions in starts.items():
        token_importance[token] = []
        for pos in positions:
            best = Importance.NORMAL
            # get start after which pos falls (pos between start-end in spans)
            idx = bisect.bisect_right(span_starts, pos) - 1
            if idx >= 0:
                start, end, importance = spans[idx]
                # make sure pos in found tuple and is more important than current best
                if start <= pos <= end and importance > best:
                    best = importance
            token_importance[token].append((pos, importance))

    return token_importance
