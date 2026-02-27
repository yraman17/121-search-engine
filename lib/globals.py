import os

# Similarity vars
HAMMING_K = 3
NUM_BITS = 64
NUM_BLOCKS = HAMMING_K + 1
# Directories
DATASET_DIR = "developer"
PARTIAL_INDEX_DIR = os.path.join("index", "partials")
FINAL_INDEX_DIR = os.path.join("index", "final")
# Other constants
BATCH_SIZE = 5000
