import numpy as np

def precision_at_k(recommended_ids, true_ids, k=5):
    """
    Tính Precision@K: Trong K gợi ý, có bao nhiêu cái đúng?
    """
    if not true_ids: return 0.0
    
    # Lấy top K
    top_k_recs = recommended_ids[:k]
    
    # Đếm số lượng đúng (nằm trong tập true_ids)
    hits = len(set(top_k_recs) & set(true_ids))
    
    return hits / k

def ndcg_at_k(recommended_ids, true_ids, k=5):
    """
    Tính NDCG@K: Đánh giá cả thứ hạng (Gợi ý đúng nằm trên cùng thì điểm cao hơn)
    """
    if not true_ids: return 0.0
    
    dcg = 0.0
    idcg = 0.0
    
    # Tính DCG
    for i, doc_id in enumerate(recommended_ids[:k]):
        if doc_id in true_ids:
            # Công thức: rel / log2(i + 2)
            dcg += 1.0 / np.log2(i + 2)
            
    # Tính IDCG (Ideal DCG - Giả sử tất cả item đúng đều nằm trên cùng)
    num_relevant = min(len(true_ids), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

# Test
if __name__ == "__main__":
    preds = [101, 102, 103, 104, 105] # ID máy gợi ý
    truth = [103, 101, 999]           # ID đúng thực tế
    
    print(f"P@5: {precision_at_k(preds, truth, k=5)}")
    print(f"NDCG@5: {ndcg_at_k(preds, truth, k=5)}")