import json
import os


def all_jsons_iter(dataset_dir):
    """Finds every json in the dataset folder"""
    all_paths = []
    for dirpath, _, filenames in os.walk(dataset_dir):
        all_paths.extend([os.path.join(dirpath, name) for name in filenames if name.endswith(".json")])
    all_paths.sort()
    yield from all_paths


def read_json_file(json_file):
    """Returns url and html from a single json file"""
    with open(json_file, "r", encoding="utf-8", errors="ignore") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return (None, None)

    url = data.get("url")
    html = data.get("content")
    if not url:
        return (None, None)
    url = url.split("#")[0]
    url = url.strip()
    return (url, html)


def iter_documents(dataset_dir):
    for path in all_jsons_iter(dataset_dir):
        url, html = read_json_file(path)
        if url is None:
            continue
        yield url, html
