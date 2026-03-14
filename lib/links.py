from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


def normalize_url(url: str) -> str:
    # Normalize URLs so that they can be added to the link graph
    if not url:
        return ""
    parts = urlparse(url)
    host = parts.netloc.lower()
    path = (parts.path or "/").rstrip("/") or "/"
    canon = f"{parts.scheme or 'https'}://{host}{path}"
    if parts.query:
        canon += "?" + parts.query
    return canon


def extract_outlinks(html: str, base_url: str, url_to_doc_id: dict[str, int]) -> list[int]:
    # Extract outlinks from HTML and return the doc_ids of the pages it links to
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    target_ids: set[int] = set()
    for anchor in soup.find_all("a", href=True):
        raw_href = str(anchor["href"]).strip()
        if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:")):
            continue
        try:
            resolved = urljoin(base_url, raw_href)
        except Exception:  # noqa: BLE001, S112
            continue
        resolved = resolved.split("#")[0].strip()
        canon = normalize_url(resolved)
        if canon and canon in url_to_doc_id:
            target_ids.add(url_to_doc_id[canon])
    return list(target_ids)
