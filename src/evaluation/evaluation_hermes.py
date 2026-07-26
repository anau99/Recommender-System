#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation_hermes_full.py
-------------------------
Đánh giá hệ thống recommend bằng "Hermes-2-Pro-Mistral-7B" (HuggingFace Serverless).
- Yêu cầu: thêm HF_TOKEN vào file .env (HF_TOKEN=hf_xxx...)
- Chạy trên Mac M1/M2/M4 / Linux / Windows (không cần GPU).
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import requests

# -----------------------
# Cấu hình đường dẫn dự án
# -----------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent  # chỉnh nếu bạn đặt file ở nơi khác
sys.path.append(str(PROJECT_ROOT))

# Load .env nếu có
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# -----------------------
# Import hệ thống recommend của bạn
# -----------------------
try:
    from src.main import CourseRecommenderSystem
except Exception as e:
    print("❌ Lỗi import src.main.CourseRecommenderSystem:", e)
    print("   Hãy chạy script này từ thư mục gốc dự án hoặc kiểm tra đường dẫn.")
    sys.exit(1)

# -----------------------
# Cấu hình HuggingFace Hermes
# -----------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("❌ Lỗi: HF_TOKEN chưa được cấu hình. Thêm HF_TOKEN=hf_xxx... vào file .env hoặc export HF_TOKEN.")
    sys.exit(1)

HF_API_URL = "https://api-inference.huggingface.co/models/NousResearch/Hermes-2-Pro-Mistral-7B"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# -----------------------
# Tham số tần suất / backoff
# -----------------------
FIXED_SLEEP_BETWEEN_REQUESTS = 2.5   # nghỉ cố định giữa 2 request (giữa 2 lần gọi model)
MAX_RETRIES = 5                      # số lần thử lại khi gặp lỗi (429, timeout, v.v.)
BASE_BACKOFF_SECONDS = 3.0           # base backoff khi gặp 429 (tăng dần)

# -----------------------
# Hàm gọi Hermes (judge) với backoff + retry
# -----------------------
def ask_hf_hermes_judge(jd_text: str, course_title: str, course_url: str) -> int:
    """
    Gọi Hermes để chấm 0/1:
      - 1 = Relevant
      - 0 = Not relevant

    Trả về int 0 hoặc 1. Mọi lỗi/timeout/không parse => trả 0.
    """
    # Chuẩn hóa prompt (cắt bớt JD nếu quá dài để giảm token)
    jd_snippet = (jd_text or "")[:1600]

    prompt = f"""
Bạn là một Senior Technical Lead giàu kinh nghiệm.
Nhiệm vụ: đánh giá nếu khóa học có giúp ứng viên đáp ứng yêu cầu công việc (JD) hay không.

--- JOB DESCRIPTION ---
{jd_snippet}

--- KHÓA HỌC ---
{course_title}
{course_url}

TRẢ LẠI CHỈ MỘT SỐ: 1 nếu Relevant, 0 nếu Not Relevant.
KHÔNG GIẢI THÍCH GÌ THÊM.
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 8,
            "temperature": 0.0,
            # bạn có thể thêm "top_k" / "top_p" nếu cần
        }
    }

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            resp = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)

            # Nếu rate-limited, backoff logic
            if resp.status_code == 429:
                wait = BASE_BACKOFF_SECONDS * attempt + FIXED_SLEEP_BETWEEN_REQUESTS
                print(f"⚠️ HF 429 Rate limit. Backoff {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue

            # Nếu lỗi khác (5xx, 4xx) raise để catch dưới
            resp.raise_for_status()

            # Parse response
            data = resp.json()
            text = ""

            # Serverless HF thường trả list[{"generated_text": "..."}]
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                text = data[0]["generated_text"].strip()
            elif isinstance(data, dict) and "generated_text" in data:
                text = data["generated_text"].strip()
            else:
                # Fallback: stringify
                text = str(data).strip()

            # Tìm 1 hoặc 0 trong output (an toàn cho nhiều format)
            # Kiểm tra cụ thể: ưu tiên "1" nếu xuất hiện trước "0"?
            # Ở đây nếu có "1" thì return 1, else if có "0" then 0.
            if "1" in text and "0" not in text:
                return 1
            if "0" in text and "1" not in text:
                return 0
            # Nếu cả 1 và 0 đều xuất hiện (hiếm), chọn ký tự đầu tiên xuất hiện
            first_pos_1 = text.find("1") if "1" in text else -1
            first_pos_0 = text.find("0") if "0" in text else -1
            if first_pos_1 >= 0 and (first_pos_0 == -1 or first_pos_1 < first_pos_0):
                return 1
            if first_pos_0 >= 0:
                return 0
            # Nếu không parse được -> fallback 0
            return 0

        except requests.exceptions.ReadTimeout:
            wait = BASE_BACKOFF_SECONDS * attempt
            print(f"⏳ ReadTimeout, đợi {wait:.1f}s rồi thử lại (attempt {attempt})...")
            time.sleep(wait)
            continue
        except requests.exceptions.ConnectionError as e:
            wait = BASE_BACKOFF_SECONDS * attempt
            print(f"🔌 ConnectionError: {e}. Đợi {wait:.1f}s rồi thử lại (attempt {attempt})...")
            time.sleep(wait)
            continue
        except Exception as e:
            # Bất kỳ lỗi khác, in log và quay về 0 (an toàn)
            print(f"❌ Lỗi khi gọi HF: {e} (attempt {attempt}/{MAX_RETRIES})")
            # short sleep trước thử lại
            time.sleep(BASE_BACKOFF_SECONDS)
            continue

    # Sau MAX_RETRIES: trả về 0
    print("⚠️ Đã vượt quá số lần retry, trả về 0 (Not relevant) mặc định.")
    return 0

# -----------------------
# Hàm chính chạy đánh giá
# -----------------------
def run_evaluation(sample_size: int = 20, top_k: int = 5):
    print("=" * 80)
    print("🤖 BẮT ĐẦU ĐÁNH GIÁ HỆ THỐNG (Hermes-2-Pro-Mistral-7B)")
    print("=" * 80)

    # Load JD file
    jd_path = PROJECT_ROOT / "data" / "jds.csv"
    if not jd_path.exists():
        print(f"❌ Không tìm thấy file {jd_path}. Hãy đặt file jds.csv vào thư mục data/ của dự án.")
        return

    df_jd = pd.read_csv(jd_path)
    # chọn cột text JD: ưu tiên các tên phổ biến
    possible_cols = ['description', 'jd', 'content', 'Job Description', 'job_description']
    text_col = next((c for c in possible_cols if c in df_jd.columns), df_jd.columns[-1])
    all_jds = df_jd[text_col].dropna().astype(str).tolist()

    if len(all_jds) == 0:
        print("❌ Không có JD hợp lệ trong file.")
        return

    sample_size = min(sample_size, len(all_jds))
    sample_jds = random.sample(all_jds, sample_size)

    # Khởi tạo hệ thống recommend của bạn
    print("-> Loading CourseRecommenderSystem...")
    try:
        recsys = CourseRecommenderSystem(base_dir=str(PROJECT_ROOT))
    except Exception as e:
        print("❌ Lỗi khi khởi tạo CourseRecommenderSystem:", e)
        return

    results = []
    total_precision = 0.0
    evaluated_jds = 0

    print(f"\n-> Đang đánh giá {len(sample_jds)} JD (top_k={top_k})\n")

    for idx, jd in enumerate(sample_jds, start=1):
        jd_trimmed = jd.strip()
        if len(jd_trimmed) < 30:
            print(f"[{idx}] Bỏ qua JD quá ngắn.")
            continue

        print(f"[{idx}/{len(sample_jds)}] Đang đánh giá JD (length={len(jd_trimmed)} chars)...")

        # Gọi hệ thống recommend để lấy top_k
        try:
            rec_output = recsys.recommend(jd_trimmed, top_k=top_k)
            courses = rec_output.get("recommendations", []) if isinstance(rec_output, dict) else []
        except Exception as e:
            print(f"❌ Lỗi khi recommend: {e}")
            continue

        if not courses:
            print("⚠️ Không có đề xuất cho JD này.\n")
            continue

        # Chấm từng course bằng Hermes
        relevant_count = 0
        for i, course in enumerate(courses, start=1):
            title = course.get("title", "Unknown Title")
            url = course.get("url", "")

            # Gọi judge (với sleep giữa các request để tránh spam)
            score = ask_hf_hermes_judge(jd_trimmed, title, url)
            relevant_count += int(score)

            # Lưu log chi tiết
            results.append({
                "jd_index": idx,
                "jd_preview": jd_trimmed[:150].replace("\n", " "),
                "course_rank": i,
                "course_title": title,
                "course_url": url,
                "system_score": course.get("score", None),
                "judge_score": score
            })

            # nghỉ cố định giữa các request
            time.sleep(FIXED_SLEEP_BETWEEN_REQUESTS)

        # Tính precision@k cho JD này
        p_at_k = relevant_count / len(courses)
        total_precision += p_at_k
        evaluated_jds += 1

        print(f"   👉 Precision@{len(courses)} = {p_at_k:.0%} ({relevant_count}/{len(courses)})\n")

    # Tổng kết
    if evaluated_jds == 0:
        print("❌ Không có JD nào được đánh giá thành công.")
        return

    avg_precision = total_precision / evaluated_jds

    print("=" * 80)
    print("🏁 KẾT QUẢ CHUNG")
    print("=" * 80)
    print(f"🎯 Evaluated JDs: {evaluated_jds}")
    print(f"🎯 Average Precision@{top_k}: {avg_precision:.1%}")
    print("=" * 80)

    # Lưu kết quả chi tiết
    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evaluation_hf_hermes_full.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"📄 Đã lưu kết quả chi tiết tại: {out_file}")

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    # Bạn có thể sửa sample_size và top_k ở đây nếu cần
    run_evaluation(sample_size=20, top_k=5)
