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
    start: int
    importance: Importance = Importance.NORMAL

    @classmethod
    def from_dict(cls, d: dict) -> "Posting":
        return cls(
            doc_id=d["doc_id"],
            start=d["start"],
            importance=Importance(d.get("importance", Importance.NORMAL)),
        )


@dataclass
class IndexEntry:
    # inverted index entry: one token -> list of postings (sorted by doc_id)
    token: str
    postings: list[Posting] = field(
        default_factory=list
    )  # creates a brand new list for every instance of IndexEntry
    df: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "IndexEntry":
        entry = cls(token=d["token"])
        entry.postings = [Posting.from_dict(p) for p in d["postings"]]
        entry.df = d["df"]
        return entry

    def add_posting(self, posting: Posting) -> None:
        bisect.insort(self.postings, posting, key=lambda x: x.doc_id)

    def merge(self, other: "IndexEntry") -> None:
        # merge postings from another IndexEntry
        for p in other.postings:
            self.add_posting(p)

    def calculate_df(self) -> None:
        unique_doc_ids = set()
        for p in self.postings:
            unique_doc_ids.add(p.doc_id)
        self.df = len(unique_doc_ids)

    def get_tf(self, doc_id: int) -> int:
        i = bisect.bisect_left(self.postings, doc_id, key=lambda x: x.doc_id)
        tf = 0
        while i < len(self.postings) and self.postings[i].doc_id == doc_id:
            tf += 1
            i += 1
        return tf


class Index:
    # inverted index: token (str) -> IndexEntry (list of Postings)
    def __init__(self):
        self.entries: list[IndexEntry] = []
        self.token_to_entry: dict[str, IndexEntry] = {}

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def from_dict(cls, d: dict) -> "Index":
        index = cls()
        for e in d["entries"]:
            entry = IndexEntry.from_dict(e)
            index.entries.append(entry)
            index.token_to_entry[entry.token] = entry
        index.entries.sort(key=lambda x: x.token)
        return index
     
    def insert_entry(self, entry):
        bisect.insort(self.entries, entry, key=lambda x: x.token)

    def add_token(
        self, token: str, doc_id: int, start:int, importance: Importance = Importance.NORMAL
    ) -> None:
        if token not in self.token_to_entry:
            entry = IndexEntry(token=token)
            self.token_to_entry[token] = entry
            self.insert_entry(entry)
        self.token_to_entry[token].add_posting(Posting(doc_id, start, importance))

    def get_entry(self, token: str) -> IndexEntry | None:
        # Return existing IndexEntry for token or empty IndexEntry
        return self.token_to_entry.get(token)

    def merge(self, other: "Index") -> None:
        # merge another index into this one
        for entry in other.entries:
            if entry.token not in self.token_to_entry:
                self.token_to_entry[entry.token] = entry
                self.insert_entry(entry)
            else:
                self.token_to_entry[entry.token].merge(entry)

    def write_to_disk(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = [asdict(e) for e in self.entries]
        with open(path, "w", encoding="utf-8") as f:
            for e in data:
                f.write(json.dumps(e, separators=(",", ":"), ensure_ascii=False))
                f.write("\n")


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


@dataclass(order=False)
class HeapEntry:
    token: str
    entry: IndexEntry
    file: TextIO

    def __lt__(self, other):
        return self.token < other.token

    def __le__(self, other):
        return self.token <= other.token

    def __eq__(self, other):
        return self.token == other.token

    def __iter__(self):
        return iter((self.token, self.entry, self.file))


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


def push_entry_to_heap(heap: list[HeapEntry], file: TextIO) -> list[HeapEntry]:
    line = file.readline()
    if line:
        entry = IndexEntry.from_dict(json.loads(line))
        heapq.heappush(heap, HeapEntry(entry.token, entry, file))
    return heap


def merge_partial_indexes(partial_paths: list[str]) -> None:
    with ExitStack() as stack:
        # Open all partial index files at once
        files = [
            stack.enter_context(open(path, "r", encoding="utf-8"))
            for path in partial_paths
        ]

        heap = []
        # Seed heap with first lines from each file
        for file in files:
            heap = push_entry_to_heap(heap, file)

        current_letter = "0"
        out_file = open(
            os.path.join(FINAL_INDEX_DIR, f"{current_letter}.jsonl"),
            "w",
            encoding="utf-8",
        )

        while heap:
            # fetch top element and push the next one from the same file
            token, entry, file = heapq.heappop(heap)
            heap = push_entry_to_heap(heap, file)

            # fetch and merge all the elements in heap that match the token popped initially
            # - Every time we pop, we push the next entry in that file to the heap to keep growing the heap
            while heap and heap[0].token == token:
                _, next_entry, same_file = heapq.heappop(heap)
                entry.merge(next_entry)
                heap = push_entry_to_heap(heap, same_file)
            # Check if new file needs to be made
            letter = token[0]
            if letter != current_letter:
                if out_file:
                    out_file.close()
                current_letter = letter
                out_file = open(
                    os.path.join(FINAL_INDEX_DIR, f"{current_letter}.jsonl"),
                    "w",
                    encoding="utf-8",
                )

            entry.calculate_df()
            out_file.write(
                json.dumps(asdict(entry), separators=(",", ":"), ensure_ascii=False)
                + "\n"
            )

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
