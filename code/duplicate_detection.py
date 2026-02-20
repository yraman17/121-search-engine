import hashlib

from simhash import (
    compute_simhash,
    hamming_distance,
    block_values,
)

HAMMING_K = 3
NUM_BITS = 64
NUM_BLOCKS = HAMMING_K + 1 


def content_hash(content: str) -> str:
    # hash of raw content for exact duplicate detection
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


class DuplicateDetector:
    # tracks seen content hashes (exact) and SimHash block indexes (near)
    # call check() before indexing a doc; if not duplicate, call add_doc() after assigning doc_id

    def __init__(self, hamming_k: int = HAMMING_K):
        self._seen_content_hashes: set[str] = set()
        self._hamming_k = hamming_k
        self._num_blocks = hamming_k + 1
        # block_index_i[block_value] = list of (full_simhash, doc_id)
        self._block_indexes: list[dict[int, list[tuple[int, int]]]] = [
            {} for _ in range(self._num_blocks)
        ]

    # check if document is an exact or near duplicate
        # returns (skip_reason, simhash_or_none):
        #   - ("exact", None) if exact duplicate -> skip
        #   - ("near", None) if near duplicate -> skip
        #   - (None, simhash) if not duplicate -> caller assigns doc_id then add_doc(simhash, doc_id)
    def check(
        self, content: str | None, token_counts: dict[str, int] | None
    ) -> tuple[str | None, int | None]:
        
        if content is None:
            return (None, None)
        # Exact duplicate check
        ch = content_hash(content)
        if ch in self._seen_content_hashes:
            return ("exact", None)
        # Near-duplicate check requires token counts for SimHash
        if token_counts is None:
            return (None, None)
        simhash = compute_simhash(token_counts, NUM_BITS)
        blocks = block_values(simhash, NUM_BITS, self._num_blocks)
        for i, block_val in enumerate(blocks):
            candidates = self._block_indexes[i].get(block_val, [])
            for other_simhash, _ in candidates:
                if hamming_distance(simhash, other_simhash, NUM_BITS) <= self._hamming_k:
                    return ("near", None)
        return (None, simhash)

    def add_doc(self, simhash: int, doc_id: int) -> None:
        # register a non-duplicate document
        blocks = block_values(simhash, NUM_BITS, self._num_blocks)
        for i, block_val in enumerate(blocks):
            if block_val not in self._block_indexes[i]:
                self._block_indexes[i][block_val] = []
            self._block_indexes[i][block_val].append((simhash, doc_id))

    def register_content_hash(self, content: str) -> None:
        # mark content as seen
        if content is not None:
            self._seen_content_hashes.add(content_hash(content))
