import bisect
import heapq
import json
import math
import os
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TextIO

from lib.globals import DOC_NORM_PATH, FINAL_INDEX_PATH, TOKEN_INFO_PATH


class Importance(IntEnum):
    # higher is more important
    NORMAL = 0
    BOLD_OR_HEADING = 1
    TITLE = 2


@dataclass
class DocPosting:
    doc_id: int
    positions: list[tuple[int, Importance]]  # (start, importance) pairs
    log_tf: float = -1

    @classmethod
    def from_dict(cls, d: dict, query_mode=False) -> "DocPosting":
        if not query_mode:
            return cls(
                doc_id=d["doc_id"],
                positions=[(p[0], Importance(p[1])) for p in d["positions"]],
                log_tf=d.get("log_tf", -1),
            )
        return cls(
            doc_id=d["doc_id"],
            positions=[],
            log_tf=d.get("log_tf", -1),
        )

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "positions": [[p[0], int(p[1])] for p in self.positions], "log_tf": round(self.log_tf, 4)}

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
    doc_postings: dict[int, DocPosting] = field(default_factory=dict)  # doc_id -> DocPosting for quick lookup
    idf: float = 0

    @classmethod
    def from_dict(cls, d: dict, with_token: bool = True, query_mode=False) -> "IndexEntry":
        token = d["token"] if with_token else ""
        entry = cls(token=token)
        entry.doc_postings = {p["doc_id"]: DocPosting.from_dict(p, query_mode) for p in d["doc_postings"]}
        entry.idf = d["idf"]
        return entry

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "doc_postings": [posting.to_dict() for posting in self.doc_postings.values()],
            "idf": round(self.idf, 4),
        }

    def get_posting(self, doc_id: int) -> DocPosting | None:
        return self.doc_postings.get(doc_id)

    def add_posting(self, doc_id: int, start: int, importance: Importance) -> None:
        existing_posting = self.get_posting(doc_id)
        if existing_posting:
            existing_posting.add_position(start, importance)
        else:
            new_posting = DocPosting(doc_id=doc_id, positions=[(start, importance)])
            self.doc_postings[doc_id] = new_posting

    def merge(self, other: "IndexEntry") -> None:
        # merge postings from another IndexEntry
        for doc_id, postings in other.doc_postings.items():
            for start, importance in postings.positions:
                self.add_posting(doc_id, start, importance)

    def get_tf(self, doc_id: int) -> float:
        posting = self.get_posting(doc_id)
        if not posting:
            return 0
        tf = 0
        for _, importance in posting.positions:
            tf += 1 + importance * 0.5
        return tf

    def calculate_log_tf(self, doc_id: int) -> None:
        tf = self.get_tf(doc_id)
        self.doc_postings[doc_id].log_tf = 1 + math.log10(tf) if tf else 0

    def calculate_idf(self, num_docs) -> None:
        unique_doc_ids = set(self.doc_postings.keys())
        self.idf = math.log10(num_docs / len(unique_doc_ids)) if unique_doc_ids else 0 / 0


class Index:
    # inverted index: token (str) -> IndexEntry (list of Postings)
    def __init__(self):
        self.token_to_entry: dict[str, IndexEntry] = {}

    def __len__(self) -> int:
        return len(self.token_to_entry)

    @classmethod
    def from_dict(cls, d: dict) -> "Index":
        index = cls()
        for e in d["entries"]:
            entry = IndexEntry.from_dict(e)
            index.token_to_entry[entry.token] = entry
        return index

    def add_entry(self, entry):
        self.token_to_entry[entry.token] = entry

    def get_entry(self, token: str) -> IndexEntry | None:
        # Return existing IndexEntry for token or empty IndexEntry
        return self.token_to_entry.get(token)

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
            self.add_entry(entry)
            curr_entry = entry
        curr_entry.add_posting(doc_id, start, importance)

    def merge(self, other: "Index") -> None:
        # merge another index into this one
        for token in other.token_to_entry:
            if token not in self.token_to_entry:
                self.token_to_entry[token] = other.token_to_entry[token]
            else:
                self.token_to_entry[token].merge(other.token_to_entry[token])

    def write_to_disk(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        sorted_entries = sorted(self.token_to_entry.values(), key=lambda x: x.token)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps(entry.to_dict(), separators=(",", ":"), ensure_ascii=False) + "\n"
                for entry in sorted_entries
            )


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
        index.token_to_entry[entry.token] = entry
    return index


def _push_entry_to_heap(heap: list[HeapEntry], file: TextIO):
    line = file.readline()
    if line:
        entry = IndexEntry.from_dict(json.loads(line))
        heapq.heappush(heap, HeapEntry(entry.token, entry, file))


def merge_partial_indexes(partial_paths: list[str], num_docs: int) -> None:
    with ExitStack() as stack:
        # Open all partial index files at once
        files = [stack.enter_context(open(path, "r", encoding="utf-8")) for path in partial_paths]

        heap = []
        offsets = {}
        doc_norms: dict[int, float] = defaultdict(float)

        # Seed heap with first lines from each file
        for file in files:
            _push_entry_to_heap(heap, file)

        with open(FINAL_INDEX_PATH, "w", encoding="utf-8") as out_file:
            while heap:
                # fetch top element and push the next entry from the same file
                heap_entry = heapq.heappop(heap)
                token, entry, file = heap_entry.token, heap_entry.entry, heap_entry.file
                _push_entry_to_heap(heap, file)

                # fetch and merge all the elements in heap that match the token popped initially
                # - Every time we pop, we push the next entry in that file to the heap to keep growing the heap
                while heap and heap[0].token == token:
                    next_heap_entry = heapq.heappop(heap)
                    next_entry, same_file = next_heap_entry.entry, next_heap_entry.file
                    entry.merge(next_entry)
                    _push_entry_to_heap(heap, same_file)

                entry.calculate_idf(num_docs)
                for doc_id in entry.doc_postings:
                    entry.calculate_log_tf(doc_id)
                    doc_norms[doc_id] += entry.doc_postings[doc_id].log_tf ** 2
                offsets[token] = (out_file.tell(), round(entry.idf, 4))
                d = entry.to_dict()
                del d["token"]  # token is redundant since it's the key in the index
                out_file.write(json.dumps(d, separators=(",", ":"), ensure_ascii=False) + "\n")

        with open(TOKEN_INFO_PATH, "w", encoding="utf-8") as f:
            json.dump(offsets, f, ensure_ascii=False)
        with open(DOC_NORM_PATH, "w", encoding="utf-8") as f:
            doc_norms = {doc_id: math.sqrt(norm) for doc_id, norm in doc_norms.items()}
            json.dump(doc_norms, f, ensure_ascii=False)


def build_token_info() -> dict[str, tuple[int, float]]:
    print("Loading token info...")
    with open(TOKEN_INFO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Token info loaded.\n")
    return {token: (v[0], v[1]) for token, v in data.items()}


def build_norms() -> dict[int, float]:
    print("Loading document norms...")
    with open(DOC_NORM_PATH, "r", encoding="utf-8") as norm_file:
        data = json.load(norm_file)
    print("Document norms loaded.\n")
    return {int(k): v for k, v in data.items()}


def fetch_from_index(token, query_mode, token_info: dict[str, tuple[int, float]], file) -> IndexEntry:
    offset = token_info[token][0] if token in token_info else None
    if offset is None:
        return IndexEntry(token)
    file.seek(offset)
    entry = IndexEntry.from_dict(json.loads(file.readline()), with_token=False, query_mode=query_mode)
    entry.token = token
    return entry
