
import faiss
import pickle
import numpy as np

# 1. Create a valid, empty FAISS index
dimension = 768  # A common dimension for embedding models
index = faiss.IndexFlatL2(dimension)
faiss.write_index(index, "faiss_index.index")
print("Created valid, empty faiss_index.index")

# 2. Create a valid, empty chunks pickle file
empty_list = []
with open("chunks.pkl", "wb") as f:
    pickle.dump(empty_list, f)
print("Created valid, empty chunks.pkl")
