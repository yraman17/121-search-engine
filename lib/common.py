import json
import os

from lib.globals import DOC_MAPPING_PATH


def write_doc_mapping(doc_id_to_url: dict[int, str], path: str = DOC_MAPPING_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {str(k): v for k, v in doc_id_to_url.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def read_doc_mapping(path: str = DOC_MAPPING_PATH) -> dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}
