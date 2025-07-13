from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create index as embedded vectors
def create_faiss_index(corpus):  # Renamed for compatibility
    embeddings = model.encode(corpus)
    return embeddings, corpus

# Query top_k most similar documents
def query_faiss(embeddings, corpus, query, top_k=5):  # Renamed for compatibility
    query_vec = model.encode([query])
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    results = [corpus[i] for i in top_indices]
    return results
