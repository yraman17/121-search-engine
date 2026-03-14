import os

# Similarity vars
HAMMING_K = 3
NUM_BITS = 64
NUM_BLOCKS = HAMMING_K + 1
# Directories
DATASET_DIR = "developer"
PARTIAL_INDEX_DIR = os.path.join("index", "partials")
FINAL_INDEX_DIR = os.path.join("index", "final")
# File paths
TOKEN_INFO_PATH = os.path.join(FINAL_INDEX_DIR, "offsets.json")
FINAL_INDEX_PATH = os.path.join(FINAL_INDEX_DIR, "final_index.jsonl")
DOC_MAPPING_PATH = os.path.join(FINAL_INDEX_DIR, "doc_mapping.json")
DOC_NORM_PATH = os.path.join(FINAL_INDEX_DIR, "doc_norms.jsonl")
DOC_PAGERANK_PATH = os.path.join(FINAL_INDEX_DIR, "doc_pagerank.json")
IDF_PATH = os.path.join(FINAL_INDEX_DIR, "idf.json")
# Other constants
BATCH_SIZE = 10000
RETURN_SIZE = 100
MIN_IDF = 0.3
PROXIMITY_WEIGHT = 0.5
