import numpy as np
import pickle
import os

class ClusterAssigner:
    def __init__(self, pca_path, kmeans_path):

        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA Model not found at: {pca_path}")
        if not os.path.exists(kmeans_path):
            raise FileNotFoundError(f"KMeans Model not found at: {kmeans_path}")
            
        # Load PCA
        with open(pca_path, 'rb') as f:
            self.pca = pickle.load(f)
            
        # Load KMeans
        with open(kmeans_path, 'rb') as f:
            self.kmeans = pickle.load(f)
            
        print(f"   - Cluster Assigner Loaded (PCA + KMeans)")

    def get_cluster(self, embedding_vector):
        if len(embedding_vector.shape) == 1:
            embedding_vector = embedding_vector.reshape(1, -1)
            
        pca_vector = self.pca.transform(embedding_vector)
        cluster_id = self.kmeans.predict(pca_vector)[0]
        
        return int(cluster_id)