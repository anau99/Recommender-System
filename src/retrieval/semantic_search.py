import faiss
import numpy as np
import os

class SemanticSearcher:
    def __init__(self, index_path):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Faiss index not found: {index_path}")
        self.index = faiss.read_index(index_path)
        
    def search(self, query_vector, top_k=20):
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        distances, indices = self.index.search(query_vector.astype('float32'), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            results.append({
                'id': int(idx), # Index trong file npy/csv
                'semantic_score': float(dist)
            })
        return results