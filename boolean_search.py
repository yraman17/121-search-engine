import heapq
import math
import sys
import time
from collections import defaultdict

from lib.common import read_doc_mapping
from lib.globals import MAX_DF, RETURN_SIZE
from lib.index import build_norms, build_offsets, fetch_from_index
from lib.parse_text import tokenize


OFFSETS = build_offsets()
DOC_MAPPING = read_doc_mapping()
NUM_DOCS = len(DOC_MAPPING)
DOC_NORMS = build_norms()

def query_parser(query: str) -> list[tuple[int, float]]:
    tokens = tokenize(query)
    counts = {token: len(positions) for token, positions in tokens.items()}
    query_len = len([t for t in counts if " " not in t])  # count single tokens only

    results = {}  # doc_id -> best score
    if query_len <= 2:  # noqa: PLR2004
        for token in counts:
            if token.count(" ") == query_len - 1:  # matches bigram for query_len==2, unigram for 1
                for doc_id, score in exact_search(token):
                    if doc_id not in results or score > results[doc_id]:
                        results[doc_id] = score

    if len(results) < RETURN_SIZE:
        for doc_id, score in vector_search(counts):
            if doc_id not in results:
                results[doc_id] = score

    return heapq.nlargest(RETURN_SIZE, results.items(), key=lambda x: x[1])

def exact_search(exact_string: str) -> list[tuple[int, float]]:
    entry = fetch_from_index(exact_string, OFFSETS)
    if not entry.doc_postings:
        return []
    return [(posting.doc_id, entry.get_log_tf(posting.doc_id)) for posting in entry.doc_postings]


def vector_search(tokens: dict[str, int]) -> list[tuple[int, float]]:
    squared_query_norm = 0
    scores: dict[int, float] = defaultdict(float)
    for token, count in tokens.items():
        entry = fetch_from_index(token, OFFSETS)
        if not entry.doc_postings or entry.df >= MAX_DF:
            continue

        token_idf = math.log10(NUM_DOCS / entry.df) if entry.df else 0
        query_weight = (1 + math.log10(count)) * token_idf if count else 0
        squared_query_norm += query_weight**2

        for posting in entry.doc_postings:
            scores[posting.doc_id] += entry.get_log_tf(posting.doc_id) * query_weight

    query_norm = math.sqrt(squared_query_norm)
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


if __name__ == "__main__":
    main(sys.argv[1:])
