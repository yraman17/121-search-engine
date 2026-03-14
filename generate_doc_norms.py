"""
Generate document norms for cosine normalization.

This repo expects `index/final/doc_norms.jsonl` (despite the extension, the file is JSON).
If you copied an index from another machine and it's missing, run:

    python generate_doc_norms.py

It will read `index/final/final_index.jsonl` line-by-line and write a JSON dict:
    { "<doc_id>": <norm>, ... }
"""

import json
import math
from collections import defaultdict

from lib.globals import DOC_NORM_PATH, FINAL_INDEX_PATH


def _log_tf(tf: float) -> float:
    return 1 + math.log10(tf) if tf else 0.0


def main() -> None:
    doc_sq_norms: dict[int, float] = defaultdict(float)
    line_count = 0

    print(f"Reading index: {FINAL_INDEX_PATH}")
    with open(FINAL_INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # entry shape: {"token": str, "doc_postings": [{"doc_id": int, "positions": [[start, importance], ...]}, ...], "idf": float}
            for posting in entry.get("doc_postings", []):
                doc_id = posting.get("doc_id")
                if doc_id is None:
                    continue
                tf = 0.0
                for _, importance in posting.get("positions", []):
                    # match IndexEntry.get_tf() weighting: 1 + importance*0.5
                    tf += 1.0 + (float(importance) * 0.5)
                w = _log_tf(tf)
                doc_sq_norms[int(doc_id)] += w * w

            if line_count % 20000 == 0:
                print(f"  processed {line_count:,} tokens...")

    doc_norms = {str(doc_id): math.sqrt(v) for doc_id, v in doc_sq_norms.items()}
    print(f"Writing norms: {DOC_NORM_PATH} ({len(doc_norms):,} docs)")
    with open(DOC_NORM_PATH, "w", encoding="utf-8") as out:
        json.dump(doc_norms, out, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    main()

