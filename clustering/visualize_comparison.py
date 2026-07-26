import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import argparse
import os

def visualize_comparison(emb_path, meta_path, out_img_path, k):
    print("Loading data for visualization...")
    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        print("Error: Files not found.")
        return

    embeddings = np.load(emb_path)
    df = pd.read_csv(meta_path)
    cat_col = next((c for c in df.columns if c.lower() in ['category', 'topic', 'domain', 'level']), None)
    if not cat_col:
        print("Warning: No category column found in metadata. Cannot plot comparison.")
        return

    n_samples = min(len(embeddings), 3000)
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    
    subset_emb = embeddings[indices]
    subset_meta = df.iloc[indices].copy()

    print(f"Running KMeans (K={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    ai_labels = kmeans.fit_predict(subset_emb)

    print("Running T-SNE (reducing to 2D)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    tsne_res = tsne.fit_transform(subset_emb)

    subset_meta['x'] = tsne_res[:, 0]
    subset_meta['y'] = tsne_res[:, 1]
    subset_meta['AI_Cluster'] = ai_labels
    subset_meta['Original_Category'] = subset_meta[cat_col].astype(str).apply(lambda x: x[:20])
    print("Plotting Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.scatterplot(data=subset_meta, x='x', y='y', hue='Original_Category', 
                    palette='tab20', s=40, alpha=0.6, ax=axes[0], legend='brief')
    axes[0].set_title(f"A. Manual Categories ({subset_meta['Original_Category'].nunique()} types)", fontsize=14, fontweight='bold')
    
    # Plot B: AI Clusters
    sns.scatterplot(data=subset_meta, x='x', y='y', hue='AI_Cluster', 
                    palette='tab20', s=40, alpha=0.6, ax=axes[1], legend='full')
    axes[1].set_title(f"B. AI K-Means Clusters (K={k})", fontsize=14, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    plt.savefig(out_img_path)
    print(f"Saved comparison image to: {out_img_path}")

if __name__ == "__main__":
    BASE_DIR = "/Users/nhitruong/Documents/data_mining_project"
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default=f"{BASE_DIR}/outputs/embeddings/course_emb_pca.npy")
    parser.add_argument("--meta", default=f"{BASE_DIR}/outputs/embeddings/course_meta.csv")
    parser.add_argument("--out_img", default=f"{BASE_DIR}/outputs/images/category_vs_kmeans.png")
    parser.add_argument("--k", type=int, default=20, help="Số K tối ưu bạn chọn")
    
    args = parser.parse_args()
    visualize_comparison(args.emb, args.meta, args.out_img, args.k)