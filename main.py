import os

from lib.doc_loading import iter_documents
from lib.globals import DATASET_DIR, PARTIAL_INDEX_DIR, FINAL_INDEX_DIR, BATCH_SIZE
from lib.parse_text import extract_text
from lib.tokenizer import tokenize
from lib.index import (
    Index,
    Importance,
    IndexStats,
    merge_partial_indexes,
    write_doc_mapping,
)
from lib.duplicate_detector import DuplicateDetector

def _get_file_size_kb(path: str) -> float:
    return os.path.getsize(path) / 1024.0

def _print_progress(file_count, doc_count, exact_dups, near_dups, unique_tokens):
    if file_count % 1000 == 0:
        print(
            f"\tProcessed {file_count} total files, indexed {doc_count} documents "
            f"({exact_dups} exact, {near_dups} near duplicates skipped, {unique_tokens} unique tokens in current index)"
        )


def _offload_partial_index(index: Index, dir: str, paths: list[str], doc_id: int):
    part_path = os.path.join(dir, f"partial_{len(paths)}.json")
    total_postings = sum(len(entry.postings) for entry in index.entries)
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
) -> IndexStats:
    # Make directories if they don't exist
    os.makedirs(final_dir, exist_ok=True)

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
            _offload_partial_index(
                current_index, partial_dir, partial_paths, next_doc_id
            )
            current_index = Index()
        # text extraction and tokenization
        normal_text, important_text = extract_text(html)
        counts_normal, _ = tokenize(normal_text)
        counts_important, _ = tokenize(important_text)
        # duplicate detection
        skip_reason, simhash_val = detector.check(html, counts_normal)
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
        for token, tf in counts_normal.items():
            current_index.add_token(token, doc_id, tf, Importance.NORMAL)
        for token, tf in counts_important.items():
            current_index.add_token(token, doc_id, tf, Importance.BOLD_OR_HEADING)

    # write remaining in-memory index as last partial if non-empty
    if current_index:
        _offload_partial_index(current_index, partial_dir, partial_paths, next_doc_id)

    # prints completed processing stats
    print(
        f"\tCompleted processing {file_count} files ({next_doc_id} indexed, "
        f"{exact_dups_removed} exact duplicates, {near_dups_removed} near duplicates skipped)"
    )

    # merge all partial indexes into final index
    print(f"[3/5] Merging {len(partial_paths)} partial index(es) into final index...")
    final_index_path = os.path.join(final_dir, "index.json")
    if not partial_paths:
        print("\tNo partial indexes to merge (empty corpus)")
        num_unique_tokens = 0
    else:
        # prints merging partial indexes
        print("\tReading and merging partial indexes...")
    for i, part_path in enumerate(partial_paths):
        part_size_kb = _get_file_size_kb(part_path)
        print(
            f"\t\t[{i + 1}/{len(partial_paths)}] Merging {part_path} ({part_size_kb:.2f} KB)"
        )

    merged = merge_partial_indexes(partial_paths, final_index_path)
    num_unique_tokens = len(merged)

    # persist doc_id -> URL mapping for report and future search
    print(f"[4/5] Writing document mapping ({len(doc_id_to_url)} documents)...")
    doc_mapping_path = os.path.join(final_dir, "doc_mapping.json")
    write_doc_mapping(doc_id_to_url, doc_mapping_path)
    print(f"\tDocument mapping saved to {doc_mapping_path}\n")

    # analytics: index size on disk (final index file + doc mapping, or just index per spec)
    print("[5/5] Computing analytics...")
    index_size_kb = _get_file_size_kb(final_index_path)

    return IndexStats(
        num_docs=next_doc_id,
        num_unique_tokens=num_unique_tokens,
        index_size_kb=index_size_kb,
        exact_dups_removed=exact_dups_removed,
        near_dups_removed=near_dups_removed,
    )


def main() -> None:

    print("=" * 60)
    print("Milestone 1: Inverted Index Builder")
    print("=" * 60 + "\n")

    index_stats = build_index()

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    report_path = os.path.join(FINAL_INDEX_DIR, "index_report.txt")
    index_stats.print_and_write(report_path)

    print(f"Analytics saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
