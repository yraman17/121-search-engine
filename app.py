import time

import streamlit as st

from lib.common import read_doc_mapping
from search import INDEX_FILE, query_parser

DOC_MAPPING = read_doc_mapping()

st.set_page_config(page_title="ICS Search Engine", layout="wide")

# GUI using Streamlit (Alan)
def main():
    st.markdown("# ICS Search Engine")
    st.markdown("Search across UCI ICS web pages")

    query = st.text_input("Enter your search query:", placeholder="e.g. machine learning, cristina lopes, ICS 33")

    if query:
        start = time.perf_counter()
        results = query_parser(query)
        elapsed = time.perf_counter() - start

        col1, col2 = st.columns(2)
        col1.metric("Results", len(results))
        col2.metric("Response Time", f"{elapsed * 1000:.1f} ms")

        st.divider()

        if not results:
            st.info("No results found. Try a different query.")
        else:
            for rank, (doc_id, score) in enumerate(results, start=1):
                url = DOC_MAPPING.get(doc_id, "Unknown")
                st.markdown(
                    f"**{rank}.** [{url}]({url})  \n"
                    f"Score: `{score:.4f}`"
                )


if __name__ == "__main__":
    main()
    INDEX_FILE.close()
