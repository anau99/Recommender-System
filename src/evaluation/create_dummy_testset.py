import pandas as pd
import json
import os
import random

def create_dummy_data(base_dir="/Users/nhitruong/Documents/data_mining_project/data_mining_project", num_samples=20):
    meta_path = f"{base_dir}/outputs/embeddings/course_meta.csv"
    output_path = f"{base_dir}/data/test_set_ground_truth.json"
    
    print(f"Loading metadata from {meta_path}...")
    if not os.path.exists(meta_path):
        print("Error: Meta file not found.")
        return

    df = pd.read_csv(meta_path)
    
    # Đảm bảo có cột ID
    if 'emb_index' not in df.columns:
        df['emb_index'] = df.index
        
    # Chọn cột text để làm JD giả
    # Ưu tiên 'clean_text', nếu không thì dùng 'description' hoặc 'course_name'
    text_col = 'clean_text' if 'clean_text' in df.columns else df.columns[-1]
    
    print(f"Generating {num_samples} synthetic test cases...")
    
    test_set = []
    
    # Lấy ngẫu nhiên num_samples khóa học
    sampled_df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    for _, row in sampled_df.iterrows():
        # Tạo JD giả: "I am looking for a course about [Tên khóa học]. [Mô tả...]"
        fake_jd = f"I want to learn {row.get('course_name', '')}. {str(row[text_col])[:500]}"
        
        test_case = {
            "jd_text": fake_jd,
            # Lưu ý: Kết quả đúng chính là khóa học này
            "correct_course_ids": [int(row['emb_index'])] 
        }
        test_set.append(test_case)
        
    # Lưu file JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Created dummy ground truth at: {output_path}")
    print("Example entry:")
    print(json.dumps(test_set[0], indent=2))

if __name__ == "__main__":
    create_dummy_data()