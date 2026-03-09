import heapq
import math
import sys
import time
from collections import defaultdict

from lib.common import read_doc_mapping
from lib.globals import MIN_IDF, RETURN_SIZE
from lib.index import Index, build_norms, build_offsets, fetch_from_index
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

    min_tokens_in_doc = max(1, query_len - 1)  # require at least query_len-1 tokens in doc for vector search
    while len(results) < RETURN_SIZE and min_tokens_in_doc > 0:
        for doc_id, score in vector_search(counts, min_tokens_in_doc):
            if doc_id not in results:
                results[doc_id] = score
        min_tokens_in_doc -= 1

    return heapq.nlargest(RETURN_SIZE, results.items(), key=lambda x: x[1])


def exact_search(exact_string: str) -> list[tuple[int, float]]:
    entry = fetch_from_index(exact_string, OFFSETS)
    if not entry.doc_postings:
        return []
    return [(posting.doc_id, entry.get_log_tf(posting.doc_id)) for posting in entry.doc_postings]


def vector_search(tokens: dict[str, int], min_tokens_in_doc: int = 1) -> list[tuple[int, float]]:
    squared_query_norm = 0
    scores: dict[int, float] = defaultdict(float)
    query_index = Index()

    # only proceed with docs that have at least min_tokens_in_doc query tokens to avoid scoring all docs with low token overlap
    doc_token_counts: dict[int, int] = defaultdict(int)
    for token in tokens:
        entry = fetch_from_index(token, OFFSETS)
        if not entry or entry.doc_postings or entry.idf <= MIN_IDF:
            continue
        query_index.add_entry(entry)
        for posting in entry.doc_postings:
            doc_token_counts[posting.doc_id] += 1

    valid_doc_ids = {doc_id for doc_id, count in doc_token_counts.items() if count >= min_tokens_in_doc}
    for token, count in tokens.items():
        entry = query_index.get_entry(token)
        if not entry or entry.doc_postings or entry.idf <= MIN_IDF:
            continue

        print(token, count, entry.idf)
        query_weight = (1 + math.log10(count)) * entry.idf if count else 0
        squared_query_norm += query_weight**2

        for doc_id in valid_doc_ids:
            scores[doc_id] += entry.get_log_tf(doc_id) * query_weight

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
