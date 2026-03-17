import bisect
import warnings
from collections import defaultdict

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk import bigrams
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

from lib.index import Importance

# suppress BeautifulSoup XML and URL warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

_tokenizer = WordPunctTokenizer()
_stemmer = PorterStemmer()
_stem_cache = {}  # cache for stemmed tokens to avoid redundant stemming


def extract_text(html_text: str) -> tuple[str, list[tuple[int, int, Importance]]]:
    # extract plain text from HTML; return (body text, important text from title/headings/bold)
    if not html_text:
        return "", []

    soup = BeautifulSoup(html_text, "lxml")
    important_tags = {"h1", "h2", "h3", "b", "strong", "title"}

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
            if parent.name in important_tags:
                importance = Importance.TITLE if parent.name == "title" else Importance.BOLD_OR_HEADING
                break

        # adjust offset and store start and end of chunk so importance can be assigned to individual tokens
        # ex: <h1>hello world</h1> -> text chunk "hello world" with importance TITLE and span (0, 10, TITLE)
        # or <p>this is an example </p> -> text chunk "this is an example" with importance NORMAL and span (11, 32, NORMAL) if it comes after the previous chunk
        start = offset
        end = offset + len(text)
        spans.append((start, end, importance))
        text_chunks.append(text)
        offset = end + 1
    # reconstruct full text
    full_text = " ".join(text_chunks)
    return full_text, spans


def tokenize(text: str) -> dict[str, list[int]]:
    # get bigrams and tokens and their positions, then combine into dict of token -> list of positions
    # stems and stores in cache to avoid redundant stemming of the same token across documents and queries, which is a bottleneck during search
    starts: defaultdict[str, list[int]] = defaultdict(list)

    if not text:
        return starts

    stemmed_list = []
    token_idx = 0
    for token in _tokenizer.tokenize(text.lower()):
        if not token or not token.isalnum() or not token.isascii():
            continue
        if token not in _stem_cache:
            _stem_cache[token] = _stemmer.stem(token)
        stemmed = _stem_cache[token]
        starts[stemmed].append(token_idx)
        stemmed_list.append((stemmed, token_idx))
        token_idx += 1

    bigrams_list = list(bigrams(stemmed_list))  # (((token1, pos1), (token2, pos2)), ((token2, pos2), (token3, pos3)), ...)
    for bigram in bigrams_list:
        bi_string = " ".join([t[0] for t in bigram])
        starts[bi_string].append(bigram[0][1])  # position of first token in bigram
    return starts


def assign_importance(
    starts: dict[str, list[int]], spans: list[tuple[int, int, Importance]]
) -> dict[str, list[tuple[int, Importance]]]:
    # Assigns importance to individual tokens basedo on chunks in spans
    # get where each chunk in spans starts
    span_starts = [span[0] for span in spans]
    pos_importance = {}  # position -> importance for quick lookup
    all_starts = set()
    for positions in starts.values():
        all_starts.update(positions)

    for pos in all_starts:
        idx = bisect.bisect_right(span_starts, pos) - 1
        if idx >= 0:
            start, end, importance = spans[idx]
            pos_importance[pos] = importance if start <= pos <= end else Importance.NORMAL
        else:
            pos_importance[pos] = Importance.NORMAL

    return {token: [(pos, pos_importance[pos]) for pos in positions] for token, positions in starts.items()}
