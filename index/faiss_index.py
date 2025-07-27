import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        # Use L2, and apply vector normalization for cosine similarity
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim)) 
        self.normalize = True  

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def add_vectors(self, vectors: np.ndarray, ids: np.ndarray):
        """
        Add vectors with corresponding chunk IDs.
        :param vectors: np.ndarray shape=(n, dim)
        :param ids: np.ndarray of chunk IDs (int64)
        """
        if self.normalize:
            vectors = self._normalize(vectors)
        self.index.add_with_ids(vectors.astype(np.float32), ids)

    def search(self, query_vector: np.ndarray, top_k=5):
        """
        Search for nearest neighbors using cosine similarity.
        :param query_vector: np.ndarray shape=(dim,)
        :param top_k: number of results
        :return: list of tuples (score, chunk_id)
        """
        query_vector = query_vector.reshape(1, -1)
        if self.normalize:
            query_vector = self._normalize(query_vector)
        query_vector = query_vector.astype(np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        return list(zip(distances[0], indices[0]))

    def reset(self):
        self.index.reset()

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
