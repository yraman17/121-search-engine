import json
import os
from enum import Enum
from typing import Iterable, List, Tuple
from lib.globals import FINAL_INDEX_DIR
from lib.index import Index, IndexEntry
from lib.index import Posting
from lib.tokenizer import tokenize

class SearchType(Enum):
    AND = "and"
    OR = "or"

def fetch_from_index(token) -> IndexEntry:
    starting_letter = token[0]
    with open(os.path.join(FINAL_INDEX_DIR, f"{starting_letter}.jsonl")) as file:
        # build index from file for this starting letter
        index = Index()
        for line in file:
            entry = IndexEntry.from_dict(json.loads(line))
            index.entries.append(entry)
            index.token_to_entry[entry.token] = entry
        # fetch entry for this token
        entry = index.get_entry(token)
        return entry if entry else IndexEntry(token)

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
    counts, _ = tokenize(raw_query)
    if not counts:
        return []
    return sorted(counts.keys())


def score_doc(doc_id: int, token_entries: List["IndexEntry"]) -> float:
    # score the document with bonus for higher importance
    score = 0.0
    importance_weight = 0.5

    for entry in token_entries:
        for p in entry.postings:
            if p.doc_id == doc_id:
                score += p.tf
                score += importance_weight * int(p.importance)
                break

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

    scored_results: List[Tuple[int, float]] = []
    for doc_id in matching_doc_ids:
        score = score_doc(doc_id, token_entries)
        scored_results.append((doc_id, score))

    scored_results.sort(key=lambda x: (-x[1], x[0]))
    return scored_results

if __name__ == "__main__":
    query = "cristina lopes"
    print(search(query))
