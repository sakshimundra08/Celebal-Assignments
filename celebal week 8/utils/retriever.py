from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a small, fast embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_faiss_index(corpus):
    embeddings = model.encode(corpus)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, corpus

def query_faiss(index, corpus, query, top_k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    results = [corpus[i] for i in I[0]]
    return results
