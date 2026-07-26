import os
import argparse
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def auto_detect_text_column(df, user_col=None):
    if user_col and user_col in df.columns:
        return user_col

    candidates = ["full_content_clean", "jd_text", "description", "text"]
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError("Không tìm thấy cột chứa nội dung JD! Vui lòng cung cấp --text_col.")


def build_text_field(df, text_col):
    title = df["title"].fillna("").astype(str) if "title" in df.columns else ""
    company = df["company"].fillna("").astype(str) if "company" in df.columns else ""
    content = df[text_col].fillna("").astype(str)

    full_text = (title + " " + company + " " + content).str.strip()
    return full_text.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to jds.csv")
    parser.add_argument("--text_col", default=None, help="Tên cột chứa JD (optional)")
    parser.add_argument("--pca_path", required=True, help="pca_model.pkl")
    parser.add_argument("--centroid_path", required=True, help="centroids_kX.npy")
    parser.add_argument("--scaler_path", required=False, help="scaler_for_selection.pkl")
    parser.add_argument("--model_name", default=None, help="SBERT model nếu muốn override")
    parser.add_argument("--out_dir", default="outputs/embeddings")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # ---------------------------------------------------------
    # Load JD CSV
    # ---------------------------------------------------------
    df = pd.read_csv(args.csv)
    text_col = auto_detect_text_column(df, args.text_col)
    print(f"Using JD text column: {text_col}")

    texts = build_text_field(df, text_col)
    print(f"Loaded {len(texts)} JD entries.")

    # ---------------------------------------------------------
    # model SBERT 
    # ---------------------------------------------------------
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

    model_name = args.model_name if args.model_name else DEFAULT_MODEL
    print(f"Using SBERT model: {model_name}")

    model = SentenceTransformer(model_name)

    # ---------------------------------------------------------
    # Encode JD text
    # ---------------------------------------------------------
    print("Encoding JD embeddings...")
    emb = model.encode(texts, normalize_embeddings=True)
    print("Raw JD embedding shape:", emb.shape)

    # ---------------------------------------------------------
    # PCA (must match courses pipeline)
    # ---------------------------------------------------------
    pca = pickle.load(open(args.pca_path, "rb"))


    if emb.shape[1] != pca.n_features_in_:
        raise ValueError(
            f"\n❌ Sai model SBERT!\n"
            f"JD embedding có {emb.shape[1]} chiều nhưng PCA yêu cầu {pca.n_features_in_} chiều.\n"
            f"→ Bạn phải dùng đúng model SBERT đã dùng cho courses.\n"
        )

    emb_pca = pca.transform(emb)
    print("After PCA:", emb_pca.shape)

    # ---------------------------------------------------------
    # Scale (if used during clustering)
    # ---------------------------------------------------------
    if args.scaler_path and os.path.exists(args.scaler_path):
        scaler = pickle.load(open(args.scaler_path, "rb"))
        emb_scaled = scaler.transform(emb_pca)
        print("Scaler applied.")
    else:
        emb_scaled = emb_pca
        print("No scaler used.")

    # ---------------------------------------------------------
    # Load centroids + predict cluster
    # ---------------------------------------------------------
    centroids = np.load(args.centroid_path)
    sims = cosine_similarity(emb_scaled, centroids)
    cluster_ids = sims.argmax(axis=1)
    scores = sims.max(axis=1)

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------
    np.save(os.path.join(args.out_dir, "jd_emb.npy"), emb_scaled)
    print("Saved jd_emb.npy")

    df_out = df.copy()
    df_out["cluster"] = cluster_ids
    df_out["score"] = scores
    df_out.to_csv(os.path.join(args.out_dir, "jd_cluster_map.csv"), index=False)
    print("Saved jd_cluster_map.csv")

    print("\n🎉 DONE! JD embeddings + cluster assignment generated without errors!")


if __name__ == "__main__":
    main()
