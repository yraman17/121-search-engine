import bisect
import json
import os
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from contextlib import ExitStack
import heapq
from typing import TextIO

from lib.globals import FINAL_INDEX_DIR


class Importance(IntEnum):
    # higher is more important
    NORMAL = 0
    BOLD_OR_HEADING = 1
    TITLE = 2


@dataclass
class Posting:
    # one posting: token's occurrence in a single document
    doc_id: int
    tf: int  # term frequency
    importance: Importance = Importance.NORMAL

    # merge two postings for the same doc_id
    def merge_with(self, other: "Posting") -> None:
        if other.doc_id != self.doc_id:
            return
        self.tf += other.tf
        if other.importance > self.importance:
            self.importance = other.importance

    @classmethod
    def from_dict(cls, d: dict) -> "Posting":
        return cls(
            doc_id=d["doc_id"],
            tf=d["tf"],
            importance=Importance(d.get("importance", 0)),
        )


@dataclass
class IndexEntry:
    # inverted index entry: one token -> list of postings (sorted by doc_id)
    token: str
    postings: list[Posting] = field(
        default_factory=list
    )  # creates a brand new list for every instance of IndexEntry

    def add_or_update_posting(
        self, doc_id: int, tf_delta: int, importance: Importance
    ) -> None:
        # add tf to the posting for doc_id, or create one, merges importance
        for p in self.postings:
            if p.doc_id == doc_id:
                p.tf += tf_delta
                if importance > p.importance:
                    p.importance = importance
                return
        # new doc for this token
        self.postings.append(Posting(doc_id=doc_id, tf=tf_delta, importance=importance))
        self.postings.sort(key=lambda x: x.doc_id)

    def merge(self, other: "IndexEntry") -> None:
        # merge postings from another IndexEntry
        for p in other.postings:
            self.add_or_update_posting(p.doc_id, p.tf, p.importance)

    @classmethod
    def from_dict(cls, d: dict) -> "IndexEntry":
        entry = cls(token=d["token"])
        entry.postings = [Posting.from_dict(p) for p in d["postings"]]
        return entry


class Index:
    # inverted index: token (str) -> IndexEntry (list of Postings)
    def __init__(self):
        self.entries: list[IndexEntry] = []
        self.token_to_entry: dict[str, IndexEntry] = {}

    def add_token(
        self,
        token: str,
        doc_id: int,
        tf: int,
        importance: Importance = Importance.NORMAL,
    ) -> None:
        if tf <= 0:
            return
        if token not in self.token_to_entry:
            entry = IndexEntry(token=token)
            self.token_to_entry[token] = entry
            bisect.insort(self.entries, entry, key=lambda x: x.token)
        self.token_to_entry[token].add_or_update_posting(doc_id, tf, importance)

    def get_entry(self, token: str) -> IndexEntry | None:
        return self.token_to_entry.get(token)

    def __len__(self) -> int:
        return len(self.entries)

    def merge(self, other: "Index") -> None:
        # merge another index into this one
        for entry in other.entries:
            if entry.token not in self.token_to_entry:
                self.token_to_entry[entry.token] = entry
                bisect.insort(self.entries, entry, key=lambda x: x.token)
            else:
                self.token_to_entry[entry.token].merge(entry)

    def write_to_disk(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = [e.to_dict() for e in self.entries]
        with open(path, "w", encoding="utf-8") as f:
            for e in data:
                f.write(json.dumps(e, separators=(",", ":"), ensure_ascii=False))
                f.write("\n")

    @classmethod
    def from_dict(cls, d: dict) -> "Index":
        index = cls()
        for e in d["entries"]:
            entry = IndexEntry.from_dict(e)
            index.entries.append(entry)
            index.token_to_entry[entry.token] = entry
        index.entries.sort(key=lambda x: x.token)
        return index


@dataclass
class IndexStats:
    num_docs: int
    num_unique_tokens: int
    index_size_kb: float
    exact_dups_removed: int
    near_dups_removed: int

    def print_and_write(self, path: str) -> None:
        analytics = (
            f"Index analytics (for report):\n"
            f"  Number of indexed documents (after dedup): {self.num_docs}\n"
            f"  Number of unique tokens:     {self.num_unique_tokens}\n"
            f"  Total size of index on disk: {self.index_size_kb:.2f} KB\n"
            f"  Exact duplicates removed:    {self.exact_dups_removed}\n"
            f"  Near-duplicates removed:     {self.near_dups_removed}\n"
        )
        print(analytics)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(analytics)


def read_partial_index(path: str) -> Index:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    index = Index()
    for d in data["entries"]:
        entry = IndexEntry.from_dict(d)
        index.entries.append(entry)
        index.token_to_entry[entry.token] = entry
    index.entries.sort(key=lambda x: x.token)
    return index

def push_entry_to_heap(heap: list[tuple[str, IndexEntry, TextIO]], file: TextIO) -> list[tuple[str, IndexEntry, TextIO]]:
    line = file.readline()
    if line:
        entry = IndexEntry.from_dict(json.loads(line))
        heapq.heappush(heap, (entry.token, entry, file))
    return heap


def merge_partial_indexes(partial_paths: list[str]) -> None:
    with ExitStack() as stack:
        files = [stack.enter_context(open(path, "r", encoding="utf-8")) for path in partial_paths]
        
        heap = []
        for file in files:
            heap = push_entry_to_heap(heap, file)
            
        current_letter = 'a'
        out_file = None

        while heap:
            token, entry, file = heapq.heappop(heap)
            heap = push_entry_to_heap(heap, file)
            
            while heap and heap[0][0] == current_letter:
                _, next_entry, same_file = heapq.heappop(heap)
                entry.merge(next_entry)
                heap = push_entry_to_heap(heap, same_file)
            letter = token[0]
            if letter != current_letter:
                if out_file:
                    out_file.close()
                out_file = open(os.path.join(FINAL_INDEX_DIR, f"{current_letter}.json"), "w", encoding="utf-8")
                current_letter = letter
            
            out_file.write(json.dumps(entry.to_dict(), separators=(",", ":"), ensure_ascii=False) + "\n")

        if out_file:
            out_file.close()

def write_doc_mapping(doc_id_to_url: dict[int, str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {str(k): v for k, v in doc_id_to_url.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)


def read_doc_mapping(path: str) -> dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

