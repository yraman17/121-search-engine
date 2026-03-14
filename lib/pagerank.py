from collections import defaultdict


def compute_pagerank(
    outlinks: dict[int, list[int]], all_doc_ids: set[int], damping: float = 0.85, max_iter: int = 100, tolerance: float = 1e-6) -> dict[int, float]:
    #Compute PageRank scores, damping factor is 0.85, max iterations is 100, tolerance is 1e-6
    rank = {pid: 1.0 for pid in all_doc_ids}
    num_pages = len(all_doc_ids)
    if num_pages == 0:
        return {}

    for _ in range(max_iter):
        next_rank: dict[int, float] = defaultdict(float)
        for source, targets in outlinks.items():
            if not targets:
                continue
            contribution = damping * (rank[source] / len(targets))
            for dest in targets:
                next_rank[dest] += contribution
        teleport = (1 - damping) / num_pages
        for pid in all_doc_ids:
            next_rank[pid] += teleport
        max_delta = max(abs(next_rank.get(pid, 0) - rank.get(pid, 0)) for pid in all_doc_ids)
        rank = dict(next_rank)
        if max_delta < tolerance:
            break
    return rank
