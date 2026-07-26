import pandas as pd
import numpy as np
import pickle
import os
import argparse
from rank_bm25 import BM25Okapi
import re

class BM25Searcher:
    def __init__(self, meta_path=None, model_path=None):
        self.bm25 = None
        self.corpus_indices = []
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        elif meta_path and os.path.exists(meta_path):
            self.train(meta_path)
        else:
            print("BM25 not initialized. Train or load a model.")

    def preprocess(self, text):

        return str(text).lower().split()

    def train(self, meta_path):
        print("Training BM25 model...")
        df = pd.read_csv(meta_path)
        
        if 'clean_text' in df.columns:
            corpus = df['clean_text'].fillna("").tolist()
        else:
            corpus = df.iloc[:, -1].astype(str).tolist() # Cột cuối cùng
            
        tokenized_corpus = [self.preprocess(doc) for doc in corpus]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_indices = df.index.tolist() 
        print("BM25 trained.")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.bm25, 'indices': self.corpus_indices}, f)
        print(f"Saved BM25 model to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['model']
            self.corpus_indices = data['indices']

    def search(self, query, top_k=20):
        tokenized_query = self.preprocess(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            results.append({
                'id': self.corpus_indices[idx],
                'bm25_score': float(scores[idx])
            })
        return results

if __name__ == "__main__":
    BASE_DIR = "/Users/nhitruong/Documents/data_mining_project"
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=f"{BASE_DIR}/outputs/embeddings/course_meta.csv")
    parser.add_argument("--save_path", default=f"{BASE_DIR}/outputs/models/bm25_model.pkl")
    
    args = parser.parse_args()
    
    searcher = BM25Searcher(meta_path=args.meta)
    searcher.save(args.save_path)