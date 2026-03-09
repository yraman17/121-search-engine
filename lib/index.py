import bisect
import heapq
import json
import math
import os
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import TextIO

from lib.globals import FINAL_INDEX_PATH, OFFSET_INDEX_PATH, DOC_NORM_PATH


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
class DocPosting:
    doc_id: int
    positions: list[tuple[int, Importance]]  # (start, importance) pairs

    @classmethod
    def from_dict(cls, d: dict) -> "DocPosting":
        return cls(
            doc_id=d["doc_id"],
            positions=[(p[0], Importance(p[1])) for p in d["positions"]],
        )

    def add_position(self, start: int, importance: Importance) -> None:
        index = bisect.bisect_left(self.positions, (start, importance))
        self.positions.insert(index, (start, importance))

    def add_positions(self, other: "DocPosting") -> None:
        for pos in other.positions:
            self.add_position(*pos)


@dataclass
class IndexEntry:
    # inverted index entry: one token -> list of postings (sorted by doc_id)
    token: str
    doc_postings: list[DocPosting] = field(default_factory=list)
    df: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "IndexEntry":
        entry = cls(token=d["token"])
        entry.doc_postings = [DocPosting.from_dict(p) for p in d["doc_postings"]]
        entry.df = d["df"]
        return entry

    def get_posting(self, doc_id: int) -> DocPosting | None:
        # binary search for posting with given doc_id
        i = bisect.bisect_left(self.doc_postings, doc_id, key=lambda x: x.doc_id)
        if i < len(self.doc_postings) and self.doc_postings[i].doc_id == doc_id:
            return self.doc_postings[i]
        return None

    def add_posting(self, doc_id: int, start: int, importance: Importance) -> None:
        existing_posting = self.get_posting(doc_id)
        if existing_posting:
            existing_posting.add_position(start, importance)
        else:
            new_posting = DocPosting(doc_id=doc_id, positions=[(start, importance)])
            index = bisect.bisect_left(self.doc_postings, doc_id, key=lambda x: x.doc_id)
            self.doc_postings.insert(index, new_posting)

    def merge(self, other: "IndexEntry") -> None:
        # merge postings from another IndexEntry
        for p in other.doc_postings:
            for start, importance in p.positions:
                self.add_posting(p.doc_id, start, importance)

    def calculate_df(self) -> None:
        unique_doc_ids = {p.doc_id for p in self.doc_postings}
        self.df = len(unique_doc_ids)

    def get_tf(self, doc_id: int) -> float:
        i = bisect.bisect_left(self.doc_postings, doc_id, key=lambda x: x.doc_id)
        tf = 0
        if i < len(self.doc_postings) and self.doc_postings[i].doc_id == doc_id:
            for _, importance in self.doc_postings[i].positions:
                tf += 1 + importance * 0.5
        return tf

    def get_log_tf(self, doc_id: int) -> float:
        tf = self.get_tf(doc_id)
        return 1 + math.log10(tf) if tf else 0

    def get_tf_idf(self, doc_id: int, num_docs: int) -> float:
        log_tf = self.get_log_tf(doc_id)
        if log_tf == 0:
            return 0
        idf = math.log10(float(num_docs) / self.df)
        return log_tf * idf


class Index:
    # inverted index: token (str) -> IndexEntry (list of Postings)
    def __init__(self):
        self.entries: list[IndexEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def from_dict(cls, d: dict) -> "Index":
        index = cls()
        for e in d["entries"]:
            entry = IndexEntry.from_dict(e)
            index.entries.append(entry)
        index.entries.sort(key=lambda x: x.token)
        return index

    def insert_entry(self, entry):
        bisect.insort(self.entries, entry, key=lambda x: x.token)

    def add_token(
        self,
        token: str,
        doc_id: int,
        start: int,
        importance: Importance = Importance.NORMAL,
    ) -> None:
        curr_entry = self.get_entry(token)
        if not curr_entry:
            entry = IndexEntry(token=token)
            self.insert_entry(entry)
            curr_entry = entry
        curr_entry.add_posting(doc_id, start, importance)

    def get_entry(self, token: str) -> IndexEntry | None:
        # Return existing IndexEntry for token or empty IndexEntry
        index = bisect.bisect_left(self.entries, token, key=lambda x: x.token)
        if index < len(self.entries) and self.entries[index].token == token:
            return self.entries[index]
        return None

    def merge(self, other: "Index") -> None:
        # merge another index into this one
        for entry in other.entries:
            existing_entry = self.get_entry(entry.token)
            if not existing_entry:
                self.insert_entry(entry)
            else:
                existing_entry.merge(entry)

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

    def __hash__(self):
        return hash(self.token)


def read_partial_index(path: str) -> Index:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    index = Index()
    for d in data["entries"]:
        entry = IndexEntry.from_dict(d)
        index.entries.append(entry)
    index.entries.sort(key=lambda x: x.token)
    return index


def _push_entry_to_heap(heap: list[HeapEntry], file: TextIO) -> list[HeapEntry]:
    line = file.readline()
    if line:
        entry = IndexEntry.from_dict(json.loads(line))
        heapq.heappush(heap, HeapEntry(entry.token, entry, file))
    return heap


def merge_partial_indexes(partial_paths: list[str]) -> None:
    with ExitStack() as stack:
        # Open all partial index files at once
        files = [stack.enter_context(open(path, "r", encoding="utf-8")) for path in partial_paths]

        heap = []
        offsets = {}
        doc_vectors: dict[int, float] = defaultdict(float)
        # Seed heap with first lines from each file
        for file in files:
            heap = _push_entry_to_heap(heap, file)

        with open(FINAL_INDEX_PATH, "w", encoding="utf-8") as out_file:
            while heap:
                # fetch top element and push the next one from the same file
                heap_entry = heapq.heappop(heap)
                token, entry, file = heap_entry.token, heap_entry.entry, heap_entry.file
                heap = _push_entry_to_heap(heap, file)

                # fetch and merge all the elements in heap that match the token popped initially
                # - Every time we pop, we push the next entry in that file to the heap to keep growing the heap
                while heap and heap[0].token == token:
                    next_heap_entry = heapq.heappop(heap)
                    next_entry, same_file = next_heap_entry.entry, next_heap_entry.file
                    entry.merge(next_entry)
                    heap = _push_entry_to_heap(heap, same_file)

                entry.calculate_df()
                for posting in entry.doc_postings:
                    doc_vectors[posting.doc_id] += entry.get_log_tf(posting.doc_id) ** 2
                offsets[token] = out_file.tell()
                d = asdict(entry)
                del d["token"]  # token is redundant since it's the key in the index
                out_file.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

        with open(OFFSET_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(offsets, f, ensure_ascii=False)
        with open(DOC_NORM_PATH, "w", encoding="utf-8") as f:
            doc_vectors = {doc_id: math.sqrt(norm) for doc_id, norm in doc_vectors.items()}
            json.dump(doc_vectors, f, ensure_ascii=False)


def build_offsets() -> dict[str, int]:
    offsets = {}
    with open(OFFSET_INDEX_PATH, "r", encoding="utf-8") as offset_file:
        for line in offset_file:
            offsets.update(json.loads(line))
    return offsets

def build_norms() -> dict[int, float]:
    with open(DOC_NORM_PATH, "r", encoding="utf-8") as norm_file:
        data = json.load(norm_file)
    return {int(k): v for k, v in data.items()}


def fetch_from_index(token, offsets: dict[str, int]) -> IndexEntry:
    offset = offsets.get(token)
    if offset is None:
        return IndexEntry(token)
    with open(FINAL_INDEX_PATH, "r", encoding="utf-8") as file:
        file.seek(offset)
        return IndexEntry.from_dict(json.loads(file.readline()))
