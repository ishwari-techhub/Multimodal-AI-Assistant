import faiss
import numpy as np


def create_vector_store(embeddings):
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index

def search_vector_store(index, query_embedding, k=3):

    query_embedding = query_embedding.astype("float32")

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    return indices