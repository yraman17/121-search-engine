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

BATCH_SIZE = 1000

def build_index(
    dataset_dir: str | None = None,
    partial_dir: str | None = None,
    final_dir: str | None = None,
) -> tuple[int, int, float]:
    # build inverted index, returns (num_docs, num_unique_tokens, index_size_kb)
    dataset_dir = dataset_dir or doc_loading.DATASET_DIR
    partial_dir = partial_dir or doc_loading.PARTIAL_INDEX_DIR
    final_dir = final_dir or doc_loading.FINAL_INDEX_DIR

    doc_id_to_url: dict[int, str] = {}
    partial_paths: list[str] = []
    current_index = Index()
    doc_count = 0

    for doc_id, url, html in doc_loading.iter_documents(dataset_dir):
        doc_id_to_url[doc_id] = url
        doc_count += 1

        normal_text, important_text = extract_text(html or "")
        counts_normal, _ = tokenize(normal_text)
        counts_important, _ = tokenize(important_text)

        for token, tf in counts_normal.items():
            current_index.add_token(token, doc_id, tf, Importance.NORMAL)
        for token, tf in counts_important.items():
            current_index.add_token(token, doc_id, tf, Importance.BOLD_OR_HEADING)

        # offload to partial index every BATCH_SIZE documents
        if doc_count % BATCH_SIZE == 0 and current_index:
            part_path = os.path.join(partial_dir, f"partial_{len(partial_paths)}.json")
            write_partial_index(current_index, part_path)
            partial_paths.append(part_path)
            current_index = Index()

    # write remaining in-memory index as last partial if non-empty
    if current_index:
        part_path = os.path.join(partial_dir, f"partial_{len(partial_paths)}.json")
        write_partial_index(current_index, part_path)
        partial_paths.append(part_path)

    # merge all partial indexes into final index
    os.makedirs(final_dir, exist_ok=True)
    final_index_path = os.path.join(final_dir, "index.json")
    if not partial_paths:
        write_partial_index(Index(), final_index_path)
        num_unique_tokens = 0
    else:
        merged = merge_partial_indexes(partial_paths, final_index_path)
        num_unique_tokens = len(merged)

    # persist doc_id -> URL mapping for report and future search
    doc_mapping_path = os.path.join(final_dir, "doc_mapping.json")
    write_doc_mapping(doc_id_to_url, doc_mapping_path)

    # analytics: index size on disk (final index file + doc mapping, or just index per spec)
    index_size_bytes = os.path.getsize(final_index_path)
    index_size_kb = index_size_bytes / 1024.0

    return len(doc_id_to_url), num_unique_tokens, index_size_kb


def main() -> None:
    num_docs, num_tokens, size_kb = build_index()
    analytics = (
        f"Index analytics (for report):\n"
        f"  Number of indexed documents: {num_docs}\n"
        f"  Number of unique tokens:     {num_tokens}\n"
        f"  Total size of index on disk: {size_kb:.2f} KB\n"
    )
    print(analytics)
    report_path = os.path.join(doc_loading.FINAL_INDEX_DIR, "index_analytics.txt")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(analytics)


if __name__ == "__main__":
    main()