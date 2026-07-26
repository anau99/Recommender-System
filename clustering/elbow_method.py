import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time

# Import GPU libraries
try:
    import cupy as cp
    from cuml.cluster import KMeans
    from cuml.metrics.cluster import silhouette_score
    GPU_AVAILABLE = True
    print(">>> RUNNING ON GPU (RAPIDS CUML)")
except ImportError:
    print("WARNING: cuML not found. Please install RAPIDS to run on GPU.")
    print("Falling back to CPU (sklearn)...")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as cp # Fallback alias for compatibility
    GPU_AVAILABLE = False

def calculate_metrics_gpu(data, k_range):

    wcss = []
    sil_scores = []
    
    if GPU_AVAILABLE and not isinstance(data, cp.ndarray):
        data_gpu = cp.asarray(data)
    else:
        data_gpu = data

    print(f"Calculating metrics for K in range {list(k_range)}...")
    
    for k in k_range:
        start_time = time.time()

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
        kmeans.fit(data_gpu)
        wcss.append(kmeans.inertia_)
        
        if GPU_AVAILABLE:
            labels = kmeans.labels
            score = silhouette_score(data_gpu, labels)
        else:
            labels = kmeans.labels_
            score = silhouette_score(data_gpu, labels)
            
        sil_scores.append(score)
        
        elapsed = time.time() - start_time
        print(f" - K={k}: SSE={kmeans.inertia_:.2f} | Sil={score:.4f} ({elapsed:.2f}s)")
        
    return wcss, sil_scores

def find_optimal_k_geometric(k_range, wcss):
    p1 = np.array([k_range[0], wcss[0]])
    p2 = np.array([k_range[-1], wcss[-1]])
    
    max_dist = 0
    best_k = k_range[0]
    
    for i, k in enumerate(k_range):
        p0 = np.array([k, wcss[i]])
        numerator = np.abs(np.cross(p2-p1, p1-p0))
        denominator = np.linalg.norm(p2-p1)
        dist = numerator / denominator
        
        if dist > max_dist:
            max_dist = dist
            best_k = k
    return best_k

def save_logs(k_range, wcss, sil_scores, best_k_elbow, best_k_sil, output_img_path):
    log_path = output_img_path.replace('.png', '.txt')
    
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("==========================================\n")
            f.write("      K-MEANS ANALYSIS REPORT (GPU)       \n")
            f.write("==========================================\n\n")
            f.write(f"Best K (Elbow Method):      {best_k_elbow}\n")
            f.write(f"Best K (Max Silhouette):    {best_k_sil}\n\n")
            f.write(f"{'K':<5} | {'SSE (Inertia)':<15} | {'Silhouette':<12} | {'Note'}\n")
            f.write("-" * 55 + "\n")
            
            for k, sse, sil in zip(k_range, wcss, sil_scores):
                note = ""
                if k == best_k_elbow: note += "[ELBOW] "
                if k == best_k_sil:   note += "[MAX SIL]"
                
                f.write(f"{k:<5} | {sse:<15.2f} | {sil:<12.4f} | {note}\n")
                
        print(f"Log saved to: {log_path}")
    except Exception as e:
        print(f"Error saving log: {e}")

def draw_plots(k_range, wcss, sil_scores, best_k_elbow, best_k_sil, output_path):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, wcss, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    
    if best_k_elbow in k_range:
        idx = list(k_range).index(best_k_elbow)
        plt.plot(best_k_elbow, wcss[idx], marker='o', markersize=12, color='red', 
                 label=f'Elbow Point (K={best_k_elbow})', zorder=5)
        
    plt.title('Elbow Method (Inertia)', fontsize=14)
    plt.xlabel('Number of clusters (k)', fontsize=12)
    plt.ylabel('SSE (Inertia)', fontsize=12)
    plt.xticks(list(k_range))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(k_range, sil_scores, marker='s', linestyle='-', color='green', linewidth=2)
    if best_k_sil in k_range:
        idx = list(k_range).index(best_k_sil)
        plt.plot(best_k_sil, sil_scores[idx], marker='*', markersize=15, color='orange',
                 label=f'Max Silhouette (K={best_k_sil})', zorder=5)
        
    plt.title('Silhouette Score (Higher is Better)', fontsize=14)
    plt.xlabel('Number of clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.xticks(list(k_range))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to: {output_path}")
    plt.close()

def run_analysis(emb_path, out_img_path, k_min=2, k_max=20):
    print(f"Loading data from {emb_path}...")
    if not os.path.exists(emb_path):
        print("Error: Embedding file not found.")
        return
    embeddings = np.load(emb_path)
    
    if embeddings.shape[0] > 20000:
        print("Data large, sampling 20k rows for analysis...")
        idx = np.random.choice(embeddings.shape[0], 20000, replace=False)
        data = embeddings[idx]
    else:
        data = embeddings
    K_range = range(k_min, k_max + 1)
    wcss, sil_scores = calculate_metrics_gpu(data, K_range)
    
    best_k_elbow = find_optimal_k_geometric(K_range, wcss)

    sil_list = [float(s) for s in sil_scores]
    best_k_sil = K_range[np.argmax(sil_list)]
    
    print(f"\n>>> SUGGESTION:")
    print(f"   Best K (Elbow Geometric): {best_k_elbow}")
    print(f"   Best K (Max Silhouette):  {best_k_sil}")
    
    draw_plots(K_range, wcss, sil_list, best_k_elbow, best_k_sil, out_img_path)
    save_logs(K_range, wcss, sil_list, best_k_elbow, best_k_sil, out_img_path)

if __name__ == "__main__":
    BASE_DIR = "/Users/nhitruong/Documents/data_mining_project"

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default=f"{BASE_DIR}/outputs/embeddings/course_emb_pca.npy")
    parser.add_argument("--out_img", default=f"{BASE_DIR}/outputs/images/elbow_analysis_gpu.png")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=20)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.emb):
        print(f"PCA file not found, checking raw embeddings...")
        args.emb = f"{BASE_DIR}/outputs/embeddings/course_emb.npy"
        
    run_analysis(args.emb, args.out_img, args.k_min, args.k_max)