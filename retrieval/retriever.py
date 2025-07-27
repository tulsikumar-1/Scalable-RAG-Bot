import numpy as np
class RetrievalQA:
    def __init__(self, embedder, faiss_index, db):
        self.embedder = embedder
        self.index = faiss_index
        self.db = db

    def build_index(self, chunks):
        self.index.reset()
        if not chunks:
            return
        embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
        ids = np.array([chunk["chunk_id"] for chunk in chunks], dtype=np.int64)
        self.index.add_vectors(embeddings, ids)

    def query(self, question: str, llm, top_k=5):
        query_emb = self.embedder.embed([question])[0]
        neighbors = self.index.search(query_emb, top_k=top_k)
        chunk_ids = [int(idx) for _, idx in neighbors if idx != -1]
        retrieved_chunks = self.db.get_chunks_by_ids(chunk_ids)
        print(retrieved_chunks)
        return llm.generate_answer(question, retrieved_chunks)