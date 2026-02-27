import json
import os
from enum import Enum

from lib.globals import FINAL_INDEX_DIR
from lib.index import Index, IndexEntry

class SearchType(Enum):
    AND = "and"
    OR = "or"

def fetch_from_index(token) -> IndexEntry:
    starting_letter = token[0]
    with open(os.path.join(FINAL_INDEX_DIR, f"{starting_letter}.jsonl")) as file:
        # build index from file for this starting letter
        index = Index()
        for line in file.readline():
            entry = IndexEntry.from_dict(json.loads(line))
            index.entries.append(entry)
            index.token_to_entry[entry.token] = entry
        # fetch entry for this token
        entry = index.get_entry(token)
        return entry if entry else IndexEntry(token)

# def merge_postings(postings, search_type: SearchType):
#     shared_docs = []
#     if search_type == SearchType.AND:
#         for posting in postings:
