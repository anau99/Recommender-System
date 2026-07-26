import numpy as np
import os
import pickle
import argparse
from sklearn.decomposition import PCA

def reduce_dimensions(input_path, output_path, model_path, variance=0.95):
    print(f"Loading embeddings from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    embeddings = np.load(input_path)
    print(f"Original shape: {embeddings.shape}")

    print(f"Fitting PCA (keeping {variance*100}% variance)...")
    pca = PCA(n_components=variance)
    pca_embeddings = pca.fit_transform(embeddings)

    print(f"Reduced shape: {pca_embeddings.shape}")
    print(f"Retained Variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    np.save(output_path, pca_embeddings)
    with open(model_path, 'wb') as f:
        pickle.dump(pca, f)

    print(f"Saved PCA embeddings to: {output_path}")
    print(f"Saved PCA model to: {model_path}")

if __name__ == "__main__":
    BASE_DIR = "/Users/nhitruong/Documents/data_mining_project"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=f"{BASE_DIR}/outputs/embeddings/course_emb.npy")
    parser.add_argument("--output", default=f"{BASE_DIR}/outputs/embeddings/course_emb_pca.npy")
    parser.add_argument("--model", default=f"{BASE_DIR}/outputs/models/pca_model.pkl")
    parser.add_argument("--variance", type=float, default=0.95)
    
    args = parser.parse_args()
    reduce_dimensions(args.input, args.output, args.model, args.variance)