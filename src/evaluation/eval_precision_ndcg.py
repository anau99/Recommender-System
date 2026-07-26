import numpy as np

def dcg_at_k(r, k):
    """Discounted Cumulative Gain"""
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    """Normalized Discounted Cumulative Gain"""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def calculate_metrics(predictions, ground_truths, k=5):
    """
    Input:
        - predictions: List[List[item_id]] (Dự đoán)
        - ground_truths: List[List[item_id]] (Thực tế đúng)
        - k: Top K
    Output: Dictionary các chỉ số
    """
    precisions = []
    ndcgs = []
    
    for pred, true_list in zip(predictions, ground_truths):
        # Tạo mảng relevance (1 nếu đúng, 0 nếu sai) cho NDCG
        relevance_vector = [1 if item in true_list else 0 for item in pred]
        
        # 1. Calculate Precision@K
        # Số lượng item đúng trong top K / K
        correct_count = sum(relevance_vector[:k])
        precisions.append(correct_count / k)
        
        # 2. Calculate NDCG@K
        ndcgs.append(ndcg_at_k(relevance_vector, k))
        
    return {
        f"precision@{k}": np.mean(precisions),
        f"ndcg@{k}": np.mean(ndcgs)
    }