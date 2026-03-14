"""
Test many candidate queries and identify the top 10 that perform best.
Measures: number of results, total score, and runtime.
Requires: index/final/ with final_index.jsonl, offsets.json, doc_mapping.json, doc_norms.jsonl
Usage: python run_query_test.py
"""
import io
import sys
import time

# Suppress debug prints from boolean_search during import and runs
_devnull = io.StringIO()

# Candidate queries (broad mix: single words, bigrams, topics, names, programs)
CANDIDATES = [
    "cristina lopes",
    "machine learning",
    "master of software engineering",
    "ACM",
    "software engineering",
    "database",
    "informatics",
    "research lab",
    "phd program",
    "faculty",
    "graduate",
    "undergraduate",
    "course",
    "computer science",
    "artificial intelligence",
    "web",
    "networks",
    "distributed systems",
    "security",
    "data science",
    "machine",
    "learning",
    "software",
    "programming",
    "algorithm",
    "database systems",
    "machine learning research",
    "mswe",
    "mcs",
    "uci",
    "ics",
    "faculty directory",
    "admission",
    "applications",
    "requirements",
    "graduation",
    "curriculum",
    "thesis",
    "internship",
    "career",
    "conference",
    "publication",
    "research group",
    "lab",
    "project",
    "python",
    "java",
    "javascript",
    "html",
    "css",
    "sql",
]


def run_query(query: str) -> tuple[list, float]:
    """Run a single query, return (results, runtime_ms). Suppresses stdout."""
    from boolean_search import query_parser

    old_stdout = sys.stdout
    sys.stdout = _devnull
    try:
        start = time.perf_counter()
        results = query_parser(query)
        elapsed = (time.perf_counter() - start) * 1000
        return results, elapsed
    finally:
        sys.stdout = old_stdout


def main():
    print("Testing candidate queries (suppressing debug output)...\n")
    scores: list[tuple[str, int, float, float]] = []  # (query, num_results, total_score, runtime_ms)

    for i, query in enumerate(CANDIDATES):
        try:
            results, runtime_ms = run_query(query)
            num_results = len(results)
            total_score = sum(abs(s) for _, s in results) if results else 0
            scores.append((query, num_results, total_score, runtime_ms))
            status = "ok" if num_results > 0 else "empty"
            print(f"  [{i+1}/{len(CANDIDATES)}] {query!r}: {num_results} results, {runtime_ms:.1f}ms [{status}]")
        except Exception as e:
            print(f"  [{i+1}/{len(CANDIDATES)}] {query!r}: ERROR - {e}")
            scores.append((query, 0, 0.0, float("inf")))

    # Rank: prefer more results, higher scores, lower runtime
    # Score = num_results * 100 + total_score - runtime_ms * 0.1 (penalize slow)
    def rank_key(x):
        q, n, ts, rt = x
        return (n * 100 + ts - rt * 0.1)

    ranked = sorted(scores, key=rank_key, reverse=True)
    top10 = ranked[:10]

    print("\n" + "=" * 60)
    print("TOP 10 queries (by results + score - runtime penalty):")
    print("=" * 60)
    for i, (query, num_results, total_score, runtime_ms) in enumerate(top10, 1):
        print(f"\n{i}. {query!r}")
        print(f"   Results: {num_results}, Total score: {total_score:.4f}, Runtime: {runtime_ms:.1f}ms")


if __name__ == "__main__":
    main()
