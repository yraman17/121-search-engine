import hashlib

def compute_simhash(token_counts: dict[str, int], num_bits: int = 64) -> int:
    # compute a SimHash fingerprint from a document's token counts
    V = [0] * num_bits
    for term, weight in token_counts.items():
        if weight <= 0:
            continue
        # 64-bit hash from first 8 bytes of hash
        h = int.from_bytes(
            hashlib.md5(term.encode("utf-8", errors="ignore")).digest()[:8],
            byteorder="big",
        )
        for i in range(num_bits):
            if (h >> i) & 1:
                V[i] += weight
            else:
                V[i] -= weight
    fingerprint = 0
    for i in range(num_bits):
        if V[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int, bits: int = 64) -> int:
    # return the number of diferieng bits between two integers
    x = a ^ b
    n = 0
    while x and bits > 0:
        n += x & 1
        x >>= 1
        bits -= 1
    return n


def block_values(fingerprint: int, num_bits: int = 64, num_blocks: int = 4) -> list[int]:
    # split fingerprint into num_blocks equal-sized blocks for faster comparison
    # operates off of the pigeonhole principle
    block_size = num_bits // num_blocks
    blocks = []
    for i in range(num_blocks):
        start_bit = i * block_size
        block = (fingerprint >> start_bit) & ((1 << block_size) - 1)
        blocks.append(block)
    return blocks
