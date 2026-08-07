from sentence_transformers import SentenceTransformer
import numpy as np

import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(texts):
    embeddings = model.encode(texts)

    return np.array(embeddings).astype("float32")