import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.id_map = []  # maps index id to (doc_id, chunk_id)

    def add_vectors(self, vectors: np.ndarray, metadata: list[tuple]):
        """
        Add vectors and metadata to the index.
        :param vectors: np.ndarray shape=(n, dim)
        :param metadata: list of tuples (doc_id, chunk_id)
        """
        self.index.add(vectors)
        self.id_map.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k=5):
        """
        Search for nearest neighbors.
        :param query_vector: np.ndarray shape=(dim,)
        :param top_k: number of results
        :return: list of tuples (score, (doc_id, chunk_id))
        """
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((dist, self.id_map[idx]))
        return results

    def reset(self):
        self.index.reset()
        self.id_map.clear()
