import hashlib
from lib.globals import HAMMING_K, NUM_BITS


def compute_simhash(token_counts: dict[str, int], num_bits: int = 64) -> int:
    bit_sums = [0] * num_bits
    for term, weight in token_counts.items():
        if weight <= 0:
            continue
        hash = int.from_bytes(
            hashlib.md5(term.encode("utf-8", errors="ignore")).digest()[:8],
            byteorder="big",
        )
        for i in range(num_bits):
            if (hash >> i) & 1:
                bit_sums[i] += weight
            else:
                bit_sums[i] -= weight
    fingerprint = 0
    for i in range(num_bits):
        if bit_sums[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int, bits: int = 64) -> int:
    x = a ^ b
    n = 0
    while x and bits > 0:
        n += x & 1
        x >>= 1
        bits -= 1
    return n


def block_values(
    fingerprint: int, num_bits: int = 64, num_blocks: int = 4
) -> list[int]:
    block_size = num_bits // num_blocks
    blocks = []
    for i in range(num_blocks):
        start_bit = i * block_size
        block = (fingerprint >> start_bit) & ((1 << block_size) - 1)
        blocks.append(block)
    return blocks


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


class DuplicateDetector:
    def __init__(self, hamming_k: int = HAMMING_K):
        self._seen_content_hashes: set[str] = set()
        self._hamming_k = hamming_k
        self._num_blocks = hamming_k + 1
        self._block_indexes: list[dict[int, list[tuple[int, int]]]] = [
            {} for _ in range(self._num_blocks)
        ]

    def check(
        self, content: str | None, token_counts: dict[str, int] | None
    ) -> tuple[str | None, int | None]:
        # Sanity checks
        if content is None:
            return None, None
        if token_counts is None:
            return None, None
        # Check for exact duplicate using content hash
        ch = content_hash(content)
        if ch in self._seen_content_hashes:
            return "exact", None
        # Check for near duplicate using SimHash
        simhash = compute_simhash(token_counts, NUM_BITS)
        blocks = block_values(simhash, NUM_BITS, self._num_blocks)
        for i, block_val in enumerate(blocks):
            candidates = self._block_indexes[i].get(block_val, [])
            for other_simhash, _ in candidates:
                if (
                    hamming_distance(simhash, other_simhash, NUM_BITS)
                    <= self._hamming_k
                ):
                    return "near", None
        return None, simhash

    def add_doc(self, simhash: int, doc_id: int) -> None:
        blocks = block_values(simhash, NUM_BITS, self._num_blocks)
        for i, block_val in enumerate(blocks):
            if block_val not in self._block_indexes[i]:
                self._block_indexes[i][block_val] = []
            self._block_indexes[i][block_val].append((simhash, doc_id))

    def register_content_hash(self, content: str) -> None:
        if content is not None:
            self._seen_content_hashes.add(content_hash(content))
