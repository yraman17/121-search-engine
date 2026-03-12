import time

import app as st
from lib.common import read_doc_mapping
from search import query_parser

DOC_MAPPING = read_doc_mapping()


def main():
    st.title("Search Engine")
    query = st.text_input("Enter your search query:")
    if query:
        start = time.perf_counter()
        results = query_parser(query)
        elapsed = time.perf_counter() - start
        st.write(f"Search completed in {elapsed * 1000:.2f}ms")
        st.write(f"Results for: '{query}'")
        for doc_id, score in results:
            st.write(f"{DOC_MAPPING[doc_id]} (Score: {score:.4f})")


if __name__ == "__main__":
    main()
