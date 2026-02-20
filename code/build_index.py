import os

import doc_loading
from parse_text import extract_text
from tokenizer import tokenize
from index import Index, Importance
from index_io import (
    write_partial_index,
    merge_partial_indexes,
    write_doc_mapping,
)
from duplicate_detection import DuplicateDetector

BATCH_SIZE = 10000

# build inverted index, returns (num_docs, num_unique_tokens, index_size_kb, exact_dups_removed, near_dups_removed)
def build_index(
    dataset_dir: str | None = None,
    partial_dir: str | None = None,
    final_dir: str | None = None,
) -> tuple[int, int, float, int, int]:
    dataset_dir = dataset_dir or doc_loading.DATASET_DIR
    partial_dir = partial_dir or doc_loading.PARTIAL_INDEX_DIR
    final_dir = final_dir or doc_loading.FINAL_INDEX_DIR

    # prints for visiblity
    print(f"[1/5] Starting index construction...")
    print(f"      Dataset directory: {dataset_dir}")
    print(f"      Partial index directory: {partial_dir}")
    print(f"      Final index directory: {final_dir}")
    print(f"      Batch size: {BATCH_SIZE} documents per partial index\n")

    doc_id_to_url: dict[int, str] = {}
    partial_paths: list[str] = []
    current_index = Index()
    next_doc_id = 0
    file_count = 0
    exact_dups_removed = 0
    near_dups_removed = 0
    detector = DuplicateDetector()

    print(f"[2/5] Processing documents and building index...") # prints for visiblity
    for _file_id, url, html in doc_loading.iter_documents(dataset_dir):
        file_count += 1

        # progress printing (runs for every file, before any continue)
        if file_count % 1000 == 0:
            print(f"      Processed {file_count} files, indexed {next_doc_id} documents "
                  f"({exact_dups_removed} exact, {near_dups_removed} near duplicates skipped)... "
                  f"(current index has {len(current_index)} unique tokens)")
        elif file_count <= 500 and file_count % 100 == 0:
            print(f"      Processed {file_count} files, indexed {next_doc_id} documents "
                  f"(current index has {len(current_index)} unique tokens)")

        # partial index offload (runs for every file, keyed on file_count)
        if file_count % BATCH_SIZE == 0 and current_index:
            part_path = os.path.join(partial_dir, f"partial_{len(partial_paths)}.json")

            total_postings = sum(len(entry.postings) for entry in current_index.entries)
            print(f"      Writing partial index #{len(partial_paths)}:")
            print(f"         - {len(current_index)} unique tokens")
            print(f"         - {total_postings} total postings")
            print(f"         - {next_doc_id} documents indexed so far")
            print(f"         - Saving to: {part_path}")

            write_partial_index(current_index, part_path)

            file_size_kb = os.path.getsize(part_path) / 1024.0
            print(f"         - Partial index size: {file_size_kb:.2f} KB\n")

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
            print(f"      Processing document #{next_doc_id}: {url[:80]}{'...' if len(url) > 80 else ''}")
            total_normal_tokens = sum(counts_normal.values())
            total_important_tokens = sum(counts_important.values())
            unique_normal = len(counts_normal)
            unique_important = len(counts_important)
            print(f"         Tokenized: {total_normal_tokens} normal tokens ({unique_normal} unique), "
                  f"{total_important_tokens} important tokens ({unique_important} unique)")
            if counts_normal:
                sample_tokens = list(counts_normal.keys())[:5]
                print(f"         Sample tokens: {', '.join(sample_tokens)}")

        for token, tf in counts_normal.items():
            current_index.add_token(token, doc_id, tf, Importance.NORMAL)
        for token, tf in counts_important.items():
            current_index.add_token(token, doc_id, tf, Importance.BOLD_OR_HEADING)

    # write remaining in-memory index as last partial if non-empty
    if current_index:
        part_path = os.path.join(partial_dir, f"partial_{len(partial_paths)}.json")

        # prints final partial index stats
        total_postings = sum(len(entry.postings) for entry in current_index.entries)
        print(f"      Writing final partial index #{len(partial_paths)}:")
        print(f"         - {len(current_index)} unique tokens")
        print(f"         - {total_postings} total postings")
        print(f"         - Saving to: {part_path}")

        write_partial_index(current_index, part_path)

        # prints final partial index size
        file_size_kb = os.path.getsize(part_path) / 1024.0
        print(f"         - Partial index size: {file_size_kb:.2f} KB\n")

        partial_paths.append(part_path)
    
    # prints completed processing stats
    print(f"      Completed processing {file_count} files ({next_doc_id} indexed, "
          f"{exact_dups_removed} exact duplicates, {near_dups_removed} near duplicates skipped)")
    print(f"      Created {len(partial_paths)} partial index(es)\n")

    # merge all partial indexes into final index
    print(f"[3/5] Merging {len(partial_paths)} partial index(es) into final index...") # prints for visiblity
    os.makedirs(final_dir, exist_ok=True)
    final_index_path = os.path.join(final_dir, "index.json")
    if not partial_paths:
        print(f"      No partial indexes to merge (empty corpus)") # prints for visiblity
        write_partial_index(Index(), final_index_path)
        num_unique_tokens = 0
    else:
        # prints merging partial indexes
        print(f"      Reading and merging partial indexes...")
        for i, part_path in enumerate(partial_paths):
            part_size_kb = os.path.getsize(part_path) / 1024.0
            print(f"         [{i+1}/{len(partial_paths)}] Merging {part_path} ({part_size_kb:.2f} KB)")

        merged = merge_partial_indexes(partial_paths, final_index_path)
        num_unique_tokens = len(merged)
        
        # prints final index stats
        total_postings = sum(len(entry.postings) for entry in merged.entries)
        final_size_kb = os.path.getsize(final_index_path) / 1024.0
        print(f"      Merged into final index:")
        print(f"         - {num_unique_tokens} unique tokens")
        print(f"         - {total_postings} total postings")
        print(f"         - Final index size: {final_size_kb:.2f} KB")
        print(f"         - Saved to: {final_index_path}\n")

    # persist doc_id -> URL mapping for report and future search
    print(f"[4/5] Writing document mapping ({len(doc_id_to_url)} documents)...") # prints for visiblity
    doc_mapping_path = os.path.join(final_dir, "doc_mapping.json")
    write_doc_mapping(doc_id_to_url, doc_mapping_path)
    print(f"      Document mapping saved to {doc_mapping_path}\n") # prints for visiblity

    # analytics: index size on disk (final index file + doc mapping, or just index per spec)
    print(f"[5/5] Computing analytics...") # prints for visiblity
    index_size_bytes = os.path.getsize(final_index_path)
    index_size_kb = index_size_bytes / 1024.0
    print(f"      Index size: {index_size_kb:.2f} KB\n") # prints for visiblity
 
    return len(doc_id_to_url), num_unique_tokens, index_size_kb, exact_dups_removed, near_dups_removed


def main() -> None:
    # prints for visiblity
    print("=" * 60)
    print("Milestone 1: Inverted Index Builder")
    print("=" * 60 + "\n")
    
    num_docs, num_tokens, size_kb, exact_dups, near_dups = build_index()
    
    # prints for visiblity
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    analytics = (
        f"Index analytics (for report):\n"
        f"  Number of indexed documents (after dedup): {num_docs}\n"
        f"  Number of unique tokens:     {num_tokens}\n"
        f"  Total size of index on disk: {size_kb:.2f} KB\n"
        f"  Exact duplicates removed:    {exact_dups}\n"
        f"  Near-duplicates removed:     {near_dups}\n"
    )
    print(analytics)
    report_path = os.path.join(doc_loading.FINAL_INDEX_DIR, "index_analytics.txt")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(analytics)

    # prints for visiblity
    print(f"Analytics saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()