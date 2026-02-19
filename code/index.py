import bisect
from dataclasses import dataclass
from enum import Enum

class IMPORTANCE(Enum):
    NORMAL = 0
    BOLD = 1
    TITLE = 2

@dataclass
class IndexNode:
    token: str
    count: int
    doc_ids: dict[int, list[int]]
    important: IMPORTANCE

class Index:
    def __init__(self):
        self.nodes = []
        self.token_to_node = {}

    def add_token(self, token: str, doc_id: int, position: int, important: IMPORTANCE):
        if token not in self.token_to_node:
            node = IndexNode(token, 0, {}, important)
            self.token_to_node[token] = node
            bisect.insort(self.nodes, node, key=lambda x: x.token)
        else:
            node = self.token_to_node[token]
            if important.value > node.important.value:
                node.important = important

        node.count += 1
        if doc_id not in node.doc_ids:
            node.doc_ids[doc_id] = []
        node.doc_ids[doc_id].append(position)
    
    def get_node(self, token: str):
        return self.token_to_node.get(token, None)

    def __len__(self):
        return len(self.nodes)
    
    def merge(self, other):
        for node in other.nodes:
            for doc_id, positions in node.doc_ids.items():
                for position in positions:
                    self.add_token(node.token, doc_id, position, node.important)