import bisect
from dataclasses import dataclass, field
from enum import IntEnum

class Importance(IntEnum):
    # higher is more important
    NORMAL = 0
    BOLD_OR_HEADING = 1
    TITLE = 2

@dataclass
class Posting:
    # one posting: token's occurrence in a single document
    doc_id: int
    tf: int # term frequency
    importance: Importance = Importance.NORMAL

    # merge two postings for the same doc_id
    def merge_with(self, other: "Posting") -> None:
        if other.doc_id != self.doc_id:
            return
        self.tf += other.tf
        if other.importance > self.importance:
            self.importance = other.importance

@dataclass
class IndexEntry:
    # inverted index entry: one token -> list of postings (sorted by doc_id)
    token: str
    postings: list[Posting] = field(default_factory=list) # creates a brand new list for every instance of IndexEntry

    def add_or_update_posting(self, doc_id: int, tf_delta: int, importance: Importance) -> None:
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

class Index:
    # inverted index: token (str) -> IndexEntry (list of Postings)
    def __init__(self):
        self.entries: list[IndexEntry] = []
        self.token_to_entry: dict[str, IndexEntry] = {}

    def add_token(self, token: str, doc_id: int, tf: int, importance: Importance = Importance.NORMAL) -> None:
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
