class CandidateUnion:
    def __init__(self, semantic_searcher, bm25_searcher):
        self.semantic = semantic_searcher
        self.bm25 = bm25_searcher
        
    def get_candidates(self, query_text, query_vector, top_k_each=20):
        # 1. Semantic Search
        sem_res = self.semantic.search(query_vector, top_k=top_k_each)
        
        # 2. Keyword Search
        kw_res = self.bm25.search(query_text, top_k=top_k_each)
        
        # 3. Merge (Union)
        candidates = {}
        
        for item in sem_res:
            i = item['id']
            candidates[i] = item
            candidates[i]['bm25_score'] = 0.0 # Mặc định
            
        for item in kw_res:
            i = item['id']
            if i in candidates:
                candidates[i]['bm25_score'] = item['bm25_score']
            else:
                candidates[i] = item
                candidates[i]['semantic_score'] = 0.0 # Mặc định
                
        return list(candidates.values())