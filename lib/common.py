import json
import os

from lib.globals import DOC_MAPPING_PATH, DOC_PAGERANK_PATH


def write_doc_mapping(doc_id_to_url: dict[int, str], path: str = DOC_MAPPING_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {str(k): v for k, v in doc_id_to_url.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def read_doc_mapping(path: str = DOC_MAPPING_PATH) -> dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def write_pagerank(rank_scores: dict[int, float], path: str = DOC_PAGERANK_PATH):
    #saves the pagerank scores to a file across runs so that we don't have to recompute them
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {str(k): round(v, 6) for k, v in rank_scores.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def read_pagerank(path: str = DOC_PAGERANK_PATH) -> dict[int, float]:
    #reads the pagerank scores from a file
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return {int(k): v for k, v in loaded.items()}
    except FileNotFoundError:
        return {}
