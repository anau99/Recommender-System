import numpy as np
import pickle
import argparse
import os
import pandas as pd
from sklearn.cluster import KMeans

def train_final_model(emb_path, meta_path, model_out_path, centroids_out_path, map_out_path, k):
    print(f"Loading embeddings from {emb_path}...")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"File not found: {emb_path}")

    embeddings = np.load(emb_path)
    
    print(f"Training Final KMeans Model with K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    kmeans.fit(embeddings)
    
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    with open(model_out_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"Saved model to: {model_out_path}")

    np.save(centroids_out_path, kmeans.cluster_centers_)
    print(f"Saved centroids to: {centroids_out_path}")

    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        
        if len(df) == len(embeddings):
            df['cluster_id'] = kmeans.labels_
            
            id_col = 'id'
            if 'id' not in df.columns:
                if 'emb_index' in df.columns:
                    id_col = 'emb_index'
                elif 'page_url' in df.columns:
                    id_col = 'page_url'
                else:
                    df['id'] = df.index
                    id_col = 'id'
        
            print(f"Saving mapping using ID column: '{id_col}'")
            df[[id_col, 'cluster_id']].to_csv(map_out_path, index=False)
            print(f"Saved cluster mapping to: {map_out_path}")
        else:
            print(f"Warning: Meta rows ({len(df)}) != Embedding rows ({len(embeddings)}). Skipping mapping file.")
    else:
        print(f"Meta file not found at {meta_path}, only model saved.")

if __name__ == "__main__":
    BASE_DIR = "/Users/nhitruong/Documents/data_mining_project"
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default=f"{BASE_DIR}/outputs/embeddings/course_emb_pca.npy")
    parser.add_argument("--meta", default=f"{BASE_DIR}/outputs/embeddings/course_meta.csv")
    parser.add_argument("--model_out", default=f"{BASE_DIR}/outputs/models/kmeans_model.pkl")
    parser.add_argument("--centroids", default=f"{BASE_DIR}/outputs/models/cluster_centroids.npy")
    parser.add_argument("--map_out", default=f"{BASE_DIR}/outputs/models/course_cluster_map.csv")
    parser.add_argument("--k", type=int, required=True, help="Số K tốt nhất bạn CHỐT để train")

    args = parser.parse_args()
    if not os.path.exists(args.emb):
        print(f"PCA not found, using raw embeddings...")
        args.emb = f"{BASE_DIR}/outputs/embeddings/course_emb.npy"

    train_final_model(args.emb, args.meta, args.model_out, args.centroids, args.map_out, args.k)