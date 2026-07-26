from typing import Tuple, Union, Optional
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(os.getcwd()) 
from src.embedding.embedding_model import get_default_model

def _ensure_out_dirs(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

def _load_courses_arg(courses_arg: Union[pd.DataFrame, str]) -> pd.DataFrame:
    if isinstance(courses_arg, pd.DataFrame):
        return courses_arg.copy()
    if isinstance(courses_arg, str):
        if not os.path.exists(courses_arg):
            raise FileNotFoundError(f"File not found: {courses_arg}")
        return pd.read_csv(courses_arg)
    raise ValueError("Input must be DataFrame or path string")

def build_embeddings(
    courses: Union[pd.DataFrame, str],
    out_dir: str = "data/outputs/embeddings",
    model_name: str = "all-mpnet-base-v2",
    device: Optional[str] = None,
    batch_size: int = 64,
) -> Tuple[str, str]:
    
    _ensure_out_dirs(out_dir)
    df = _load_courses_arg(courses)
    
    print(f"Original Data Rows: {len(df)}")
    

    df.columns = [c.lower().strip() for c in df.columns]

    priority_cols = [
        'course_name', 
        'topic', 
        'course_summary', 
        'what_you_learn_clean', 
        'skills_gain_clean'
    ]
    
    cols_to_use = [c for c in priority_cols if c in df.columns]
    
    print(f"Constructing embedding text from columns: {cols_to_use}")

    def combine_text(row):
        parts = []
        for col in cols_to_use:
            val = str(row[col])
    
            if val and val.lower() != 'nan' and val.strip() != '':
                parts.append(val.strip())
        return ". ".join(parts)


    df['clean_text'] = df.apply(combine_text, axis=1)

  
    empty_mask = df["clean_text"].str.len() < 2 
    empty_count = empty_mask.sum()
    
    if empty_count > 0:
        print(f"Warning: Found {empty_count} rows with empty text. Filling with 'Course Info Unavailable'.")
        df.loc[empty_mask, "clean_text"] = "Course Info Unavailable"
    
    df = df.reset_index(drop=True)
    texts = df["clean_text"].tolist()

    print(f"Start embedding ALL {len(texts)} rows...")
    model = get_default_model(model_name=model_name, device=device)

    embeddings = model.encode_batch(texts, batch_size=batch_size)
    emb_path = os.path.join(out_dir, "course_emb.npy")
    meta_path = os.path.join(out_dir, "course_meta.csv")

    np.save(emb_path, embeddings)

    meta_cols_candidate = [
        'page_url', 'course_name', 'topic', 'rating', 'level', 
        'duration_hours', 'skills_gain_clean', 'clean_text'
    ]
    keep_cols = [c for c in meta_cols_candidate if c in df.columns]
    
    df_meta = df[keep_cols].copy()
    
    if 'course_name' in df_meta.columns:
        df_meta = df_meta.rename(columns={'course_name': 'course_name'})
    
    df_meta["emb_index"] = df_meta.index
    df_meta.to_csv(meta_path, index=False)
    
    return emb_path, meta_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--courses", default="data/courses.csv")
    parser.add_argument("--out_dir", default="outputs/embeddings")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    emb, meta = build_embeddings(
        courses=args.courses,
        out_dir=args.out_dir,
        batch_size=args.batch_size
    )
    meta_df = pd.read_csv(meta)
    print(f"DONE! Processed {len(meta_df)} rows.")
    print(f"- Embeddings saved to: {emb}")
    print(f"- Metadata saved to:   {meta}")