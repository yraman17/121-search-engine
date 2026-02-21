import os

from lib.doc_loading import iter_documents
from lib.globals import DATASET_DIR, PARTIAL_INDEX_DIR, FINAL_INDEX_DIR, BATCH_SIZE
from lib.parse_text import extract_text
from lib.tokenizer import tokenize
from lib.index import (
    Index,
    Importance,
    IndexStats,
    write_partial_index,
    merge_partial_indexes,
    write_doc_mapping,
)
from lib.duplicate_detector import DuplicateDetector


def _print_progress(file_count, doc_count, exact_dups, near_dups, unique_tokens):
    if file_count % 1000 == 0:
        print(
            f"      Processed {file_count} total files, indexed {doc_count} documents "
            f"({exact_dups} exact, {near_dups} near duplicates skipped)... "
            f"(current index has {unique_tokens} unique tokens)"
        )
    elif file_count <= 500 and file_count % 100 == 0:
        print(
            f"      Processed {file_count} files, indexed {doc_count} documents "
            f"(current index has {unique_tokens} unique tokens)"
        )


def _offload_partial_index(current_index, part_path, partial_paths, next_doc_id):
    total_postings = sum(len(entry.postings) for entry in current_index.entries)
    print(f"      Writing partial index #{len(partial_paths)}:")
    print(f"         - {len(current_index)} unique tokens")
    print(f"         - {total_postings} total postings")
    print(f"         - {next_doc_id} documents indexed so far")
    print(f"         - Saving to: {part_path}")

    write_partial_index(current_index, part_path)

    file_size_kb = os.path.getsize(part_path) / 1024.0
    print(f"         - Partial index size: {file_size_kb:.2f} KB\n")


# build inverted index, returns (num_docs, num_unique_tokens, index_size_kb, exact_dups_removed, near_dups_removed)
def build_index(
    dataset_dir: str = DATASET_DIR,
    partial_dir: str = PARTIAL_INDEX_DIR,
    final_dir: str = FINAL_INDEX_DIR,
) -> IndexStats:
    # prints for visiblity
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

    print("[2/5] Processing documents and building index...")  # prints for visiblity
    for url, html in iter_documents(dataset_dir):
        file_count += 1

        # progress printing (runs for every file, before any continue)
        _print_progress(
            file_count,
            next_doc_id,
            exact_dups_removed,
            near_dups_removed,
            len(current_index),
        )

        # partial index offload (runs for every file, keyed on file_count)
        if file_count % BATCH_SIZE == 0 and current_index:
            part_path = os.path.join(partial_dir, f"partial_{len(partial_paths)}.json")
            _offload_partial_index(current_index, part_path, partial_paths, next_doc_id)
            partial_paths.append(part_path)
            current_index = Index()

        if html is None:
            continue

        normal_text, important_text = extract_text(html)
        counts_normal, _ = tokenize(normal_text)
        counts_important, _ = tokenize(important_text)

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

        if next_doc_id <= 3:
            print(
                f"\tProcessing document #{next_doc_id}: {url[:80]}{'...' if len(url) > 80 else ''}"
            )
            total_normal_tokens = sum(counts_normal.values())
            total_important_tokens = sum(counts_important.values())
            unique_normal = len(counts_normal)
            unique_important = len(counts_important)
            print(
                f"\t\tTokenized: {total_normal_tokens} normal tokens ({unique_normal} unique), "
                f"{total_important_tokens} important tokens ({unique_important} unique)"
            )
            if counts_normal:
                sample_tokens = list(counts_normal.keys())[:5]
                print(f"\t\tSample tokens: {', '.join(sample_tokens)}")

        for token, tf in counts_normal.items():
            current_index.add_token(token, doc_id, tf, Importance.NORMAL)
        for token, tf in counts_important.items():
            current_index.add_token(token, doc_id, tf, Importance.BOLD_OR_HEADING)

    # write remaining in-memory index as last partial if non-empty
    if current_index:
        part_path = os.path.join(partial_dir, f"partial_{len(partial_paths)}.json")
        _offload_partial_index(current_index, part_path, partial_paths, next_doc_id)
        partial_paths.append(part_path)

    # prints completed processing stats
    print(
        f"\tCompleted processing {file_count} files ({next_doc_id} indexed, "
        f"{exact_dups_removed} exact duplicates, {near_dups_removed} near duplicates skipped)"
    )
    print(f"\tCreated {len(partial_paths)} partial index(es)\n")

    # merge all partial indexes into final index
    print(
        f"[3/5] Merging {len(partial_paths)} partial index(es) into final index..."
    )  # prints for visiblity
    os.makedirs(final_dir, exist_ok=True)
    final_index_path = os.path.join(final_dir, "index.json")
    if not partial_paths:
        print(
            "\tNo partial indexes to merge (empty corpus)"
        )  # prints for visiblity
        write_partial_index(Index(), final_index_path)
        num_unique_tokens = 0
    else:
        # prints merging partial indexes
        print("\tReading and merging partial indexes...")
        for i, part_path in enumerate(partial_paths):
            part_size_kb = os.path.getsize(part_path) / 1024.0
            print(
                f"\t\t[{i + 1}/{len(partial_paths)}] Merging {part_path} ({part_size_kb:.2f} KB)"
            )

        merged = merge_partial_indexes(partial_paths, final_index_path)
        num_unique_tokens = len(merged)

        # prints final index stats
        total_postings = sum(len(entry.postings) for entry in merged.entries)
        final_size_kb = os.path.getsize(final_index_path) / 1024.0
        print("\tMerged into final index:")
        print(f"\t\t- {num_unique_tokens} unique tokens")
        print(f"\t\t- {total_postings} total postings")
        print(f"\t\t- Final index size: {final_size_kb:.2f} KB")
        print(f"\t\t- Saved to: {final_index_path}\n")

    # persist doc_id -> URL mapping for report and future search
    print(
        f"[4/5] Writing document mapping ({len(doc_id_to_url)} documents)..."
    )  # prints for visiblity
    doc_mapping_path = os.path.join(final_dir, "doc_mapping.json")
    write_doc_mapping(doc_id_to_url, doc_mapping_path)
    print(
        f"\tDocument mapping saved to {doc_mapping_path}\n"
    )  # prints for visiblity

    # analytics: index size on disk (final index file + doc mapping, or just index per spec)
    print("[5/5] Computing analytics...")  # prints for visiblity
    index_size_bytes = os.path.getsize(final_index_path)
    index_size_kb = index_size_bytes / 1024.0
    print(f"\tIndex size: {index_size_kb:.2f} KB\n")  # prints for visiblity

    return IndexStats(
        num_docs=next_doc_id,
        num_unique_tokens=num_unique_tokens,
        index_size_kb=index_size_kb,
        exact_dups_removed=exact_dups_removed,
        near_dups_removed=near_dups_removed,
    )


def main() -> None:
    # prints for visiblity
    print("=" * 60)
    print("Milestone 1: Inverted Index Builder")
    print("=" * 60 + "\n")

    index_stats = build_index()

    # prints for visiblity
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    report_path = os.path.join(FINAL_INDEX_DIR, "index_report.txt")
    index_stats.print_and_write(report_path)

    # prints for visiblity
    print(f"Analytics saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
