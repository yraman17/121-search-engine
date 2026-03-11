import heapq
import math
import sys
import time
from collections import defaultdict
from functools import lru_cache

from lib.common import read_doc_mapping
from lib.globals import FINAL_INDEX_PATH, MIN_IDF, RETURN_SIZE
from lib.index import IndexEntry, build_norms, build_token_info, fetch_from_index
from lib.parse_text import tokenize

TOKEN_INFO = build_token_info()
DOC_MAPPING = read_doc_mapping()
NUM_DOCS = len(DOC_MAPPING)
DOC_NORMS = build_norms()
INDEX_FILE = open(FINAL_INDEX_PATH, "r", encoding="utf-8")  # noqa: SIM115


@lru_cache(maxsize=1024)
def _fetch_from_entry_cached(token, query_mode=False) -> IndexEntry:
    entry = fetch_from_index(token, query_mode, TOKEN_INFO, INDEX_FILE)
    if entry and entry.doc_postings:
        return entry
    return IndexEntry(token)  # return empty entry for tokens not found or with no postings to avoid repeated lookups


def query_parser(query: str) -> list[tuple[int, float]]:
    tokens = tokenize(query)
    counts = {token: len(positions) for token, positions in tokens.items()}
    unigram_tokens = [t for t in counts if " " not in t]
    query_len = len(unigram_tokens)  # count single tokens only

    results = {}  # doc_id -> best score
    if query_len <= 2:  # noqa: PLR2004
        for token in unigram_tokens:
            for doc_id, score in exact_search(token):
                if doc_id not in results or score > results[doc_id]:
                    results[doc_id] = score

    # get entries for query tokens and doc_id: # of query tokens in doc
    entries = {}
    doc_token_counts: dict[int, int] = defaultdict(int)
    query_entry_weights: dict[str, float] = defaultdict(float)
    query_norm = 0.0
    for token, count in counts.items():
        if token in TOKEN_INFO and TOKEN_INFO[token][1] < MIN_IDF:
            continue
        entry = _fetch_from_entry_cached(token)
        if not entry or not entry.doc_postings:
            continue
        entries[token] = entry
        if " " not in token:  # only count single tokens for doc_token_counts
            for doc_id in entry.doc_postings:
                doc_token_counts[doc_id] += 1
        weight = (1 + math.log10(count)) * entry.idf if count else 0
        query_entry_weights[token] = weight
        query_norm += weight**2

    query_len = sum(1 for t in entries if " " not in t)
    query_norm = math.sqrt(query_norm)
    valid_doc_counts = {
        i: {doc_id for doc_id, count in doc_token_counts.items() if count >= i}
        for i in range(1, query_len + 1)
    }

    # Iterate in reverse from query_len to 1, getting docs with at least that many query tokens and adding to results until we have enough results
    min_tokens_in_doc = max(1, query_len - 1)
    while len(results) < RETURN_SIZE and min_tokens_in_doc > 0:
        valid_doc_ids = valid_doc_counts.get(min_tokens_in_doc, set())
        for doc_id, score in vector_search(query_entry_weights, query_norm, entries, valid_doc_ids):
            if doc_id not in results:
                results[doc_id] = score
        min_tokens_in_doc -= 1

    return heapq.nlargest(RETURN_SIZE, results.items(), key=lambda x: x[1])


def exact_search(exact_string: str) -> list[tuple[int, float]]:
    entry = _fetch_from_entry_cached(exact_string)
    if not entry or not entry.doc_postings:
        return []
    return [(doc_id, posting.log_tf) for doc_id, posting in entry.doc_postings.items()]


def vector_search(
    query_entry_weights: dict[str, float], query_norm: float, entries: dict[str, IndexEntry], valid_doc_ids: set[int]
) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for token, entry in entries.items():
        for doc_id, posting in entry.doc_postings.items():
            if doc_id in valid_doc_ids:
                scores[doc_id] += posting.log_tf * query_entry_weights[token]
    for doc_id in scores:
        doc_norm = DOC_NORMS.get(doc_id, 1)
        scores[doc_id] /= (query_norm * doc_norm) or 1.0
    return list(scores.items())


def main(args: list[str]) -> None:
    query = args[0]
    output_num = 1
    print(f"Searching for: '{query}'\n")
    start = time.perf_counter()
    results = query_parser(query)
    elapsed = time.perf_counter() - start
    for doc_id, score in results:
        print(f"{output_num}. URL: {DOC_MAPPING.get(doc_id, 'Unknown')}, Score: {abs(score):.4f}")
        output_num += 1
    print(f"Search completed in {elapsed * 1000:.2f}ms")
    INDEX_FILE.close()


if __name__ == "__main__":
    main(sys.argv[1:])
