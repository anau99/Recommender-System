import time
import numpy as np

def compare_methods(jd_vectors, ground_truths):
   
    results = {
        "faiss": {"time": [], "precision": []},
        "cluster": {"time": [], "precision": []}
    }
    
    print("Comparing Faiss vs Clustering...")
    
    for i, vec in enumerate(jd_vectors):
        # 1. Test Faiss (Brute-force or IVF)
        start = time.time()
        # faiss_res = search_faiss(vec, top_k=10) # CALL REAL FUNCTION
        faiss_time = time.time() - start
        
        # 2. Test Cluster (Predict cluster -> Search in cluster)
        start = time.time()
        # cluster_res = search_cluster(vec, top_k=10) # CALL REAL FUNCTION
        cluster_time = time.time() - start
        
        # Record Time
        results["faiss"]["time"].append(faiss_time)
        results["cluster"]["time"].append(cluster_time)
        
    
    avg_time_faiss = np.mean(results["faiss"]["time"])
    avg_time_cluster = np.mean(results["cluster"]["time"])
    
    print(f"Avg Time Faiss: {avg_time_faiss:.4f}s")
    print(f"Avg Time Cluster: {avg_time_cluster:.4f}s")
    
    improvement = (avg_time_faiss - avg_time_cluster) / avg_time_faiss * 100
    print(f"Speed Improvement: {improvement:.2f}%")