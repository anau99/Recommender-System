import pandas as pd
import google.generativeai as genai
import os
import sys
import time
import random
import json
from pathlib import Path
from dotenv import load_dotenv

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Thêm thư mục gốc vào sys.path để import được src.main
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load Env
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# --- IMPORT HỆ THỐNG GỢI Ý ---
try:
    from src.main import CourseRecommenderSystem
except ImportError:
    print("❌ Lỗi: Không tìm thấy src.main. Hãy chạy script này từ thư mục gốc dự án.")
    sys.exit(1)

# --- CẤU HÌNH GEMINI JUDGE ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ⚠️ SỬ DỤNG TÊN MODEL ĐÃ TEST THÀNH CÔNG
MODEL_NAME = 'models/gemini-2.5-flash' 

if not GEMINI_API_KEY:
    print("❌ Lỗi: Chưa cấu hình GEMINI_API_KEY trong file .env")
    sys.exit(1)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"❌ Config Error: {e}")
    sys.exit(1)

def ask_gemini_judge(jd_text, course_title, course_url):
    """
    Hàm đóng vai trò Giám khảo (Senior Technical Lead).
    Trả về: 1 (Relevant) hoặc 0 (Not Relevant)
    """
    prompt = f"""
    Bạn là một **Senior Technical Lead** (Trưởng nhóm kỹ thuật) giàu kinh nghiệm.
    Nhiệm vụ: Đánh giá xem nội dung khóa học có thực sự giúp ứng viên đáp ứng yêu cầu công việc (JD) hay không.

    --- JOB DESCRIPTION (JD) ---
    {jd_text[:1500]} ...

    --- KHÓA HỌC ĐƯỢC GỢI Ý ---
    Tên: {course_title}
    Link: {course_url}

    --- TIÊU CHÍ ĐÁNH GIÁ ---
    1. **Relevant (1)**: Khóa học dạy đúng kỹ năng/công nghệ/kiến thức mà JD yêu cầu (Ví dụ: JD cần 'React', khóa học dạy 'React Advanced').
    2. **Not Relevant (0)**: Khóa học quá cơ bản, không liên quan, hoặc sai lệch công nghệ (Ví dụ: JD cần 'Java', khóa học dạy 'JavaScript', hoặc JD cần 'Deep Learning' nhưng khóa học là 'Excel').

    YÊU CẦU: CHỈ TRẢ VỀ DUY NHẤT MỘT SỐ: 0 HOẶC 1. KHÔNG GIẢI THÍCH GÌ THÊM.
    """

    # Retry logic để tránh lỗi Rate Limit (429)
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            result = response.text.strip()
            
            # Xử lý kết quả trả về
            if '1' in result: return 1
            if '0' in result: return 0
            
            # Nếu model trả lời lan man, coi như không rõ ràng (0) hoặc thử lại
            return 0 
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                wait = 20 * (attempt + 1)
                print(f"      ⚠️ Gemini bận (Rate Limit). Đợi {wait}s...")
                time.sleep(wait)
            else:
                print(f"      ❌ Lỗi API Judge: {error_msg}")
                return 0
    return 0

def run_evaluation():
    print("="*60)
    print(">>> 🤖 BẮT ĐẦU ĐÁNH GIÁ HỆ THỐNG (ROLE: TECHNICAL LEAD)")
    print("="*60)

    # 1. Load Dữ liệu JD
    jd_path = PROJECT_ROOT / "data" / "jds.csv"
    if not jd_path.exists():
        print(f"❌ Không tìm thấy file {jd_path}")
        return

    df = pd.read_csv(jd_path)
    
    # Tìm cột text JD
    text_col = next((c for c in ['description', 'jd', 'Job Description', 'content'] if c in df.columns), None)
    if not text_col:
        text_col = df.columns[-1] # Fallback
    
    # Lấy ngẫu nhiên 20 JD (Sample) để đánh giá
    SAMPLE_SIZE = 20
    if len(df) > SAMPLE_SIZE:
        sample_jds = df.sample(n=SAMPLE_SIZE, random_state=42)[text_col].tolist()
    else:
        sample_jds = df[text_col].tolist()

    # 2. Khởi tạo Hệ thống
    print("   -> Loading Recommender System...")
    # Khởi tạo hệ thống (có thể mất vài giây load model)
    recsys = CourseRecommenderSystem(base_dir=str(PROJECT_ROOT))
    
    total_score = 0
    total_items = 0
    results_log = []

    print(f"\n   -> Đang đánh giá {len(sample_jds)} JD mẫu...")

    # 3. Vòng lặp Đánh giá từng JD
    for i, jd in enumerate(sample_jds):
        # Bỏ qua JD quá ngắn/lỗi
        if not isinstance(jd, str) or len(jd) < 50: continue

        print(f"\n[{i+1}/{len(sample_jds)}] Evaluating JD...")
        
        # A. Lấy gợi ý từ hệ thống (Top 5)
        try:
            rec_output = recsys.recommend(jd, top_k=5)
            courses = rec_output['recommendations']
        except Exception as e:
            print(f"      ❌ Lỗi hệ thống recommend: {e}")
            continue

        if not courses:
            print("      ⚠️ Không có gợi ý nào.")
            continue

        # B. Chấm điểm từng khóa học
        jd_relevant_count = 0
        for course in courses:
            # Gọi Gemini Judge
            score = ask_gemini_judge(jd, course['title'], course.get('url', ''))
            jd_relevant_count += score
            
            # Log chi tiết để kiểm tra sau này
            results_log.append({
                "jd_id": i,
                "jd_snippet": jd[:50] + "...",
                "course_title": course['title'],
                "system_score": course['score'],
                "judge_score": score
            })
            
            # Nghỉ 2s giữa các lần gọi để tránh spam API
            time.sleep(2)

        # Tính Precision cho JD này (Số khóa đúng / Tổng số khóa gợi ý)
        p_at_k = jd_relevant_count / len(courses)
        total_score += p_at_k
        total_items += 1
        
        print(f"      👉 Precision@5: {p_at_k:.0%} ({jd_relevant_count}/{len(courses)} Relevant)")

    # 4. Kết quả chung cuộc
    if total_items > 0:
        final_precision = total_score / total_items
        print("\n" + "="*60)
        print(f"🏆 KẾT QUẢ ĐÁNH GIÁ (SAMPLE {total_items} JDs)")
        print("="*60)
        print(f"🎯 AVERAGE PRECISION (Technical Lead): {final_precision:.1%}")
        print("="*60)
        
        # Lưu file csv kết quả
        out_path = PROJECT_ROOT / "data" / "evaluation_results_tech_lead.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(results_log).to_csv(out_path, index=False)
        print(f"📝 Chi tiết đã lưu tại: {out_path}")
    else:
        print("❌ Không đánh giá được JD nào.")

if __name__ == "__main__":
    run_evaluation()