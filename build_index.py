import os
import sys

from lib.common import read_doc_mapping, write_doc_mapping, write_pagerank
from lib.doc_loading import iter_documents
from lib.duplicate_detector import DuplicateDetector
from lib.globals import BATCH_SIZE, DATASET_DIR, FINAL_INDEX_DIR, PARTIAL_INDEX_DIR
from lib.index import (
    Index,
    merge_partial_indexes,
)
from lib.links import extract_outlinks, normalize_url
from lib.parse_text import assign_importance, extract_text, tokenize
from lib.pagerank import compute_pagerank


def _get_file_size_kb(path: str) -> float:
    return os.path.getsize(path) / 1024.0


def _print_progress(file_count, doc_count, exact_dups, near_dups, unique_tokens):
    if file_count % 1000 == 0:
        print(
            f"\tProcessed {file_count} total files, indexed {doc_count} documents "
            f"({exact_dups} exact, {near_dups} near duplicates skipped, {unique_tokens} unique tokens in current index)"
        )


def _offload_partial_index(index: Index, directory: str, paths: list[str], doc_id: int):
    part_path = os.path.join(directory, f"partial_{len(paths)}.jsonl")
    total_postings = sum(len(entry.doc_postings) for entry in index.token_to_entry.values())
    print(f"      Writing partial index #{len(paths)}:")
    print(f"         - {len(index)} unique tokens")
    print(f"         - {total_postings} total postings")
    print(f"         - {doc_id} documents indexed so far")
    print(f"         - Saving to: {part_path}")

    index.write_to_disk(part_path)

    file_size_kb = _get_file_size_kb(part_path)
    print(f"         - Partial index size: {file_size_kb:.2f} KB\n")
    paths.append(part_path)


# build inverted index, returns (num_docs, num_unique_tokens, index_size_kb, exact_dups_removed, near_dups_removed)
def build_index(
    dataset_dir: str = DATASET_DIR,
    partial_dir: str = PARTIAL_INDEX_DIR,
    final_dir: str = FINAL_INDEX_DIR,
) -> None:
    # Make directories if they don't exist
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(partial_dir, exist_ok=True)

    print("[1/5] Starting index construction...")
    print(f"\tDataset directory: {dataset_dir}")
    print(f"\tPartial index directory: {partial_dir}")
    print(f"\tFinal index directory: {final_dir}")
    print(f"\tBatch size: {BATCH_SIZE} documents per partial index\n")

    doc_id_to_url: dict[int, str] = {}
    partial_paths: list[str] = []
    current_index = Index()
    next_doc_id = 0
    file_count = 0
    exact_dups_removed = 0
    near_dups_removed = 0
    detector = DuplicateDetector()

    print("[2/5] Processing documents and building index...")
    for url, html in iter_documents(dataset_dir):
        if html is None:
            continue

        file_count += 1
        # progress printing (runs for every 1000 files)
        _print_progress(
            file_count,
            next_doc_id,
            exact_dups_removed,
            near_dups_removed,
            len(current_index),
        )
        # partial index offload (runs for every file, keyed on file_count)
        if file_count % BATCH_SIZE == 0 and current_index:
            _offload_partial_index(current_index, partial_dir, partial_paths, next_doc_id)
            current_index = Index()
        # text extraction and tokenization
        full_text, spans = extract_text(html)
        starts = tokenize(full_text)
        token_importance = assign_importance(starts, spans)
        # duplicate detection
        skip_reason, simhash_val = detector.check(html, {token: len(starts) for token, starts in starts.items()})
        if skip_reason == "exact":
            exact_dups_removed += 1
            continue
        if skip_reason == "near":
            near_dups_removed += 1
            continue

        detector.register_content_hash(html)
        doc_id = next_doc_id
        next_doc_id += 1
        if simhash_val is not None:
            detector.add_doc(simhash_val, doc_id)

        doc_id_to_url[doc_id] = url
        for token in token_importance:
            for start, importance in token_importance[token]:
                current_index.add_token(token, doc_id, start, importance)

    # write remaining in-memory index as last partial if non-empty
    if current_index:
        _offload_partial_index(current_index, partial_dir, partial_paths, next_doc_id)

    # persist doc_id -> URL mapping for report and future search
    print(f"[4/5] Writing document mapping ({len(doc_id_to_url)} documents)...")
    write_doc_mapping(doc_id_to_url)
    print("\tDocument mapping saved to disk\n")

    # merge all partial indexes into final index
    print(f"[3/5] Merging {len(partial_paths)} partial index(es) into final index...")
    print("\tNo partial indexes to merge (empty corpus)") if not partial_paths else print(
        "\tReading and merging partial indexes..."
    )
    doc_mapping = read_doc_mapping()
    merge_partial_indexes(partial_paths, len(doc_mapping))

    # Build link graph and compute PageRank
    if doc_mapping:
        print("[5/5] Building link graph and computing PageRank...")
        url_lookup = {normalize_url(u): did for did, u in doc_mapping.items()}
        link_graph: dict[int, list[int]] = {}
        for url, html in iter_documents(dataset_dir):
            if html is None:
                continue
            canon = normalize_url(url)
            did = url_lookup.get(canon)
            if did is None:
                continue
            link_graph[did] = extract_outlinks(html, url, url_lookup)
        doc_ids = set(doc_mapping.keys())
        rank_scores = compute_pagerank(link_graph, doc_ids)
        write_pagerank(rank_scores)
        print("\tPageRank scores saved.\n")


def main(arg) -> None:
    if arg:
        build_index()
    else:
        partial_paths = [os.path.join(PARTIAL_INDEX_DIR, f"partial_{num}.jsonl") for num in range(12)]
        merge_partial_indexes(partial_paths, len(read_doc_mapping()))


if __name__ == "__main__":
    main(int(sys.argv[1]))
