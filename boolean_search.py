import bisect
from collections import defaultdict
import heapq
import json
import os
from enum import Enum
from re import T
import sys
import time
from typing import Iterable, List, Tuple
from lib.common import read_doc_mapping
from lib.globals import FINAL_INDEX_PATH, RETURN_SIZE
from lib.index import DocPosting, IndexEntry, build_norms, build_offsets, fetch_from_index
from lib.parse_text import tokenize
import math

OFFSETS = build_offsets()
DOC_MAPPING = read_doc_mapping()
NUM_DOCS = len(DOC_MAPPING)
DOC_NORMS = build_norms()

class SearchType(Enum):
    AND = "AND"
    OR = "OR"
        

def merge_postings(
    postings_lists: Iterable[Iterable["DocPosting"]], search_type: "SearchType"
) -> List[int]:
    # merge postings according to given SearchType

    postings_lists = list(postings_lists)
    if not postings_lists:
        return []

    doc_sets = []
    for postings in postings_lists:
        doc_ids = {p.doc_id for p in postings}
        if not doc_ids and search_type == SearchType.AND:
            return []
        doc_sets.append(doc_ids)

    if not doc_sets:
        return []

    if search_type == SearchType.AND:
        shared = set.intersection(*doc_sets)
    elif search_type == SearchType.OR:
        shared = set.union(*doc_sets)
    else:
        shared = set()

    return sorted(shared)


def process_query(raw_query: str) -> List[str]:
    # tokenize and stem the raw query
    starts = tokenize(raw_query)
    if not starts:
        return []
    return sorted(starts.keys()) # ! does this need to be sorted?


def score_doc(unique_query_tokens: set[str], doc_id: int, token_entries: List["IndexEntry"], num_docs: int = NUM_DOCS) -> float:
    # score the document with bonus for higher importance
    score = 0.0
    for entry in token_entries:
        if entry.token in unique_query_tokens:
            score += entry.get_tf_idf(doc_id, num_docs)
                
    return score


def search(query: str, search_type: SearchType = SearchType.AND) -> list[Tuple[int, float]]:
    tokens = tokenize(query)
    if not tokens:
        return []

    squared_query_norm = 0
    counts = {token: len(tokens[token]) for token in tokens}
    scores: dict[int, float] = defaultdict(float)
    for token, count in counts.items():
        entry = fetch_from_index(token, OFFSETS)
        if not entry.doc_postings:
            continue

        token_idf = math.log10(NUM_DOCS / entry.df) if entry.df else 0
        query_weight = math.log10(count + 1) * token_idf
        squared_query_norm += query_weight ** 2

        for posting in entry.doc_postings:
            scores[posting.doc_id] += entry.get_log_tf(posting.doc_id) * query_weight
    
    query_norm = math.sqrt(squared_query_norm)
    for doc_id in scores:
        doc_norm = DOC_NORMS.get(doc_id, 1)
        scores[doc_id] /= (query_norm * doc_norm) if (query_norm * doc_norm) else 1.0
    return heapq.nlargest(RETURN_SIZE, scores.items(), key=lambda x: x[1])

    # token_entries: List[IndexEntry] = []
    # postings_lists: List[Iterable[DocPosting]] = []

    # for token in tokens:
    #     entry = fetch_from_index(token, OFFSETS)
    #     if not entry.doc_postings:
    #         # if AND, if token has no postings, can't be satisfied
    #         if search_type == SearchType.AND:
    #             return []
    #         # if OR skip tokens w empty postings
    #         continue

    #     token_entries.append(entry)
    #     postings_lists.append(entry.doc_postings)

    # if not postings_lists:
    #     return []

    # matching_doc_ids = merge_postings(postings_lists, search_type)
    # if not matching_doc_ids:
    #     return []

    # scored_results: List[Tuple[str, float]] = []
    # unique_query_tokens = set(tokens)
    # for doc_id in matching_doc_ids:
    #     score = score_doc(unique_query_tokens, doc_id, token_entries)
    #     scored_results.append((DOC_MAPPING[doc_id], score))

    # scored_results.sort(key=lambda x: x[1], reverse=True)
    # return scored_results

def main(args: List[str]) -> None:
    build_offsets()
    query = args[0]
    output_num = 1
    start = time.perf_counter()
    results = search(query)
    elapsed = time.perf_counter() - start
    for doc_id, score in results:
        print(f"{output_num}. URL: {DOC_MAPPING.get(doc_id, 'Unknown')}, Score: {abs(score):.4f}")
        output_num += 1
    print(f"Search completed in {elapsed * 1000:.2f}ms")
        

if __name__ == "__main__":
    main(sys.argv[1:])
