import os
import sys
import yaml
import json
import itertools
import numpy as np
import time
from tqdm import tqdm

__)
src_dir = os.path.dirname(os.path.dirname(current_file)) # .../src
project_root = os.path.dirname(src_dir)                  # .../data_mining_project
if src_dir not in sys.path: sys.path.insert(0, src_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)


from src.main import CourseRecommenderSystem
from src.recommender.scoring import Scorer

def calculate_metrics(predictions, ground_truth, k=5):
    """
    Tính Precision@K: Tỷ lệ khóa học đúng trong top K gợi ý
    """
    if not ground_truth: return 0.0
    
    pred_ids = [p['id'] for p in predictions[:k]]
    
    hits = len(set(pred_ids).intersection(set(ground_truth)))
    
    return hits / k

class WeightSearcher:
    def __init__(self, config_path="configs/grid_search.yaml"):
        self.base_dir = project_root
        self.config_path = os.path.join(self.base_dir, config_path)
        
        # 1. Load Config
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found at {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 2. Load System 
        print(">>> Loading Recommender System Core...")
        self.system = CourseRecommenderSystem(base_dir=self.base_dir)
        
        # 3. Load Test Data (Ground Truth)
        test_path = os.path.join(self.base_dir, self.config['grid_search']['evaluation']['test_data_path'])
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found at {test_path}. Run create_dummy_testset.py first!")
            
        with open(test_path, 'r') as f:
            self.test_data = json.load(f)
        print(f" Loaded {len(self.test_data)} test cases.")

    def generate_grid(self):
        params = self.config['grid_search']['weights']
        keys = params.keys()
        values = params.values()

        grid = []
        for combo in itertools.product(*values):
            weight_dict = dict(zip(keys, combo))
            
            total_base = weight_dict['semantic'] + weight_dict['bm25']
            if 0.9 <= total_base <= 1.1:
                grid.append(weight_dict)
                
        print(f"Generated {len(grid)} weight combinations to test.")
        return grid

    def run(self):
        grid = self.generate_grid()
        top_k = self.config['grid_search']['evaluation']['top_k']
        
        best_score = -1.0
        best_config = None
        
        print("\n>>> STARTING GRID SEARCH...")
        start_time = time.time()
        
        for i, weights in enumerate(tqdm(grid, desc="Optimizing")):
            
            self.system.scorer = Scorer(weights=weights)
            
            avg_precision = 0.0
            
            for case in self.test_data:
                jd = case['jd_text']
                true_ids = case['correct_course_ids']
                
                result = self.system.recommend(jd, top_k=top_k)
                score = calculate_metrics(result['recommendations'], true_ids, k=top_k)
                avg_precision += score
            
            avg_precision /= len(self.test_data)
            if avg_precision > best_score:
                best_score = avg_precision
                best_config = weights
                tqdm.write(f"New Best Found! P@{top_k}={best_score:.4f} | Weights: {weights}")

        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("GRID SEARCH COMPLETED")
        print("="*60)
        print(f"Time Taken: {total_time:.2f}s")
        print(f"Best Precision@{top_k}: {best_score:.4f}")
        print("Best Configuration:")
        print(json.dumps(best_config, indent=4))
        
        out_path = os.path.join(self.base_dir, self.config['grid_search']['evaluation']['output_path'])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(best_config, f, indent=4)
        print(f"\nSaved best weights to: {out_path}")

if __name__ == "__main__":
    try:
        searcher = WeightSearcher()
        searcher.run()
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")