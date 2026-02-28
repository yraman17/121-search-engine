import bisect
import json
import os
from enum import Enum
from re import T
import sys
import time
from typing import Iterable, List, Tuple
from lib.globals import FINAL_INDEX_DIR
from lib.index import IndexEntry
from lib.index import Posting
from lib.parse_text import tokenize
import math

OFFSETS = {}

class SearchType(Enum):
    AND = "AND"
    OR = "OR"

def build_offsets():
    with open(os.path.join(FINAL_INDEX_DIR, "offset.json"), "r", encoding="utf-8") as offset_file:
        for line in offset_file:
            OFFSETS.update(json.loads(line))

def fetch_from_index(token) -> IndexEntry:
    offset = OFFSETS.get(token)
    if offset is None:
        return IndexEntry(token)
    with open(os.path.join(FINAL_INDEX_DIR, f"{token[0]}.jsonl")) as file:
        file.seek(offset)
        return IndexEntry.from_dict(json.loads(file.readline()))
        

def merge_postings(
    postings_lists: Iterable[Iterable["Posting"]], search_type: "SearchType"
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


def score_doc(doc_id: int, token_entries: List["IndexEntry"], num_docs: int) -> float:
    # score the document with bonus for higher importance
    score = 0.0
    for entry in token_entries:
        # importance already factored into tf calculation, so just calculate tf-idf
        tf_raw = entry.get_tf(doc_id)
        # skip if doc_id doesn't occur in this entries postings
        if tf_raw == 0:
            continue
        tf = (1 + math.log(tf_raw, 10)) 
        idf = math.log((float(num_docs)/entry.df), 10)
        score += tf * idf
                
    return score


def search(query: str, search_type: SearchType = SearchType.AND) -> List[Tuple[int, float]]:
    tokens = process_query(query)
    if not tokens:
        return []

    token_entries: List[IndexEntry] = []
    postings_lists: List[Iterable[Posting]] = []

    for token in tokens:
        entry = fetch_from_index(token)
        if not entry.postings:
            # if AND, if token has no postings, can't be satisfied
            if search_type == SearchType.AND:
                return []
            # if OR skip tokens w empty postings
            continue

        token_entries.append(entry)
        postings_lists.append(entry.postings)

    if not postings_lists:
        return []

    matching_doc_ids = merge_postings(postings_lists, search_type)
    if not matching_doc_ids:
        return []

    with open(os.path.join(FINAL_INDEX_DIR, "doc_mapping.json"), "r") as f:
        doc_mapping = json.load(f)
    num_docs = len(doc_mapping)
    scored_results: List[Tuple[int, float]] = []
    for doc_id in matching_doc_ids:
        score = score_doc(doc_id, token_entries, num_docs)
        scored_results.append((doc_mapping[str(doc_id)], score))

    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results

def main(args: List[str]) -> None:
    build_offsets()
    query = args[0]
    output_num = 1
    start = time.perf_counter()
    results = search(query)
    elapsed = time.perf_counter() - start
    for result in results:
        print(f"{output_num}. URL: {result[0]}, Score: {result[1]}")
        output_num += 1
    print(f"Search completed in {elapsed * 1000:.2f}ms")
        

if __name__ == "__main__":
    main(sys.argv[1:])
