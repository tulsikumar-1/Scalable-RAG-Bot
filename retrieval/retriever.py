import numpy as np

class RetrievalQA:
    def __init__(self, embedder, faiss_index, db):
        self.embedder = embedder
        self.index = faiss_index
        self.db = db

    def build_index(self):
        """
        Build FAISS index from DB chunks and embeddings.
        """
        self.index.reset()
        chunks = self.db.get_all_chunks()
        if not chunks:
            return
        embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
        metadata = [(chunk["doc_id"], chunk["chunk_id"]) for chunk in chunks]
        self.index.add_vectors(embeddings, metadata)

    def query(self, question: str, llm):
        """
        Query LLM with retrieved chunks from FAISS index.
        """
        query_emb = self.embedder.embed([question])[0]
        neighbors = self.index.search(query_emb, top_k=5)

        retrieved_chunks = []
        session = self.db.Session()
        from storage.database import Chunk
        for _, (doc_id, chunk_id) in neighbors:
            chunk_obj = session.query(Chunk).filter(Chunk.id == chunk_id).first()
            if chunk_obj:
                retrieved_chunks.append({
                    "doc_name": chunk_obj.document.name,
                    "chunk_id": chunk_obj.id,
                    "text": chunk_obj.text
                })
        session.close()

        answer = llm.generate_answer(question, retrieved_chunks)
        return answer
