import json
import os
from index import Index, IndexEntry, Posting, Importance

def _posting_to_dict(p: Posting) -> dict:
    return {"doc_id": p.doc_id, "tf": p.tf, "importance": int(p.importance)}

def _dict_to_posting(d: dict) -> Posting:
    return Posting(
        doc_id=d["doc_id"],
        tf=d["tf"],
        importance=Importance(d.get("importance", 0)),
    )

def _entry_to_dict(entry: IndexEntry) -> dict:
    return {
        "token": entry.token,
        "postings": [_posting_to_dict(p) for p in entry.postings],
    }

def _dict_to_entry(d: dict) -> IndexEntry:
    entry = IndexEntry(token=d["token"])
    entry.postings = [_dict_to_posting(p) for p in d["postings"]]
    return entry

def write_partial_index(index: Index, path: str) -> None:
    # write in-memory index to JSON file
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {"entries": [_entry_to_dict(e) for e in index.entries]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)

def read_partial_index(path: str) -> Index:
    # read partial index from disk to Index
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    index = Index()
    for d in data["entries"]:
        entry = _dict_to_entry(d)
        index.entries.append(entry)
        index.token_to_entry[entry.token] = entry
    index.entries.sort(key=lambda x: x.token)
    return index

def merge_partial_indexes(partial_paths: list[str], final_path: str) -> Index:
    # read all partial index files, merge by token, write final index
    merged = Index()
    for path in partial_paths:
        part = read_partial_index(path)
        merged.merge(part)
    write_partial_index(merged, final_path)
    return merged

def write_doc_mapping(doc_id_to_url: dict[int, str], path: str) -> None:
    # persist doc_id -> url mapping
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {str(k): v for k, v in doc_id_to_url.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)

def read_doc_mapping(path: str) -> dict[int, str]:
    # load doc_id -> url mapping from file
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}
