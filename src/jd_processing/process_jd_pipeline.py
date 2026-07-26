# #!/usr/bin/env python3
# # coding: utf-8
# """
# process_jd_pipeline.py
# ----------------------
# Chạy pipeline xử lý JD đầy đủ (clean -> summarize -> skill extract -> experience map).
# Hỗ trợ 2 chế độ:
#  - use_llm = True: gọi OpenAI Chat API để tóm tắt và trích skill chính xác (cần OPENAI_API_KEY)
#  - use_llm = False: dùng local methods (SentenceTransformer summarizer + regex skill extractor)

# Output:
#  - out_dir/processed_jds.csv (DataFrame with columns: original_text, clean_text, summary, skills, domains, years, level, company/title if present)
#  - out_dir/processed_jds.jsonl (one json per line)

# Usage (ví dụ):
# python src/jd_processing/process_jd_pipeline.py \
#   --csv /workspace/nhitny/data_mining_project/data/jds.csv \
#   --text_col full_content_clean \
#   --out_dir outputs/embeddings \
#   --use_llm 1 \
#   --openai_model gpt-3.5-turbo

# Ghi chú:
# - Nếu bật --use_llm, cần cài package `openai` và export OPENAI_API_KEY.
# - Nếu không bật, sẽ dùng SentenceTransformer (cần cài sentence-transformers).
# """

# import os
# import argparse
# import json
# import time
# from typing import List, Dict, Any

# import pandas as pd
# from tqdm import tqdm

# # Import local jd processing modules (các file bạn đã có)
# # Cần đảm bảo thư mục chứa package src trong PYTHONPATH hoặc chạy từ project root.
# try:
#     from src.jd_processing.jd_cleaner import JDProcessorCleaner
#     from src.jd_processing.jd_summarizer import JDSummarizer
#     from src.jd_processing.skill_extractor import SkillExtractor
#     from src.jd_processing.experience_mapper import ExperienceMapper
# except Exception:
#     # support running when module path is direct relative
#     from jd_processing.jd_cleaner import JDProcessorCleaner
#     from jd_processing.jd_summarizer import JDSummarizer
#     from jd_processing.skill_extractor import SkillExtractor
#     from jd_processing.experience_mapper import ExperienceMapper

# # Optional OpenAI integration
# USE_OPENAI = False
# try:
#     import openai
#     USE_OPENAI = True
# except Exception:
#     USE_OPENAI = False

# # -------------------------
# # Helper: LLM wrappers
# # -------------------------
# def llm_summarize(text: str, model: str = "gpt-3.5-turbo", max_sentences: int = 5, api_key: str = None) -> str:
#     """
#     Gọi OpenAI ChatCompletion để tóm tắt JD: trả về tổng hợp ngắn gọn (max_sentences câu).
#     Nếu openai lib không sẵn hoặc api_key None, raise RuntimeError.
#     """
#     if not USE_OPENAI:
#         raise RuntimeError("openai package not available. Install openai or run with --use_llm 0.")
#     if api_key:
#         openai.api_key = api_key
#     if not openai.api_key:
#         raise RuntimeError("OPENAI_API_KEY not found. Set env var or pass api_key.")

#     system = "You are a concise assistant specialized in summarizing job descriptions into a short, informative summary."
#     user = (
#         "Tóm tắt văn bản sau thành tối đa {} câu, giữ lại các yêu cầu công việc (những nhiệm vụ chính, công nghệ chính, "
#         "và trình độ kinh nghiệm). Trả về văn bản tiếng Việt nếu đầu vào tiếng Việt, ngược lại trả về tiếng Anh.\n\n"
#         "JD:\n\n"
#     ).format(max_sentences) + text

#     # Use chat completion
#     try:
#         resp = openai.ChatCompletion.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": user}
#             ],
#             temperature=0.0,
#             max_tokens=512,
#         )
#         summary = resp["choices"][0]["message"]["content"].strip()
#         return summary
#     except Exception as e:
#         raise RuntimeError(f"LLM summarize failed: {e}")


# def llm_extract_skills(text: str, model: str = "gpt-3.5-turbo", api_key: str = None) -> Dict[str, Any]:
#     """
#     Gọi LLM để trích xuất skills và domains, trả về dict: {"skills": [...], "domains": [...]}.
#     Kết quả nên ở định dạng JSON.
#     """
#     if not USE_OPENAI:
#         raise RuntimeError("openai package not available.")
#     if api_key:
#         openai.api_key = api_key
#     if not openai.api_key:
#         raise RuntimeError("OPENAI_API_KEY not found. Set env var or pass api_key.")

#     prompt = (
#         "Bạn là một extractor: từ mô tả công việc dưới đây, hãy trả về JSON với 2 trường: "
#         "\"skills\" (mảng các kỹ năng kỹ thuật/ứng dụng, viết thường) và \"domains\" (mảng tên lĩnh vực/tên ngành). "
#         "Tránh trả lời text ngoài JSON. Nếu không tìm thấy, trả về mảng rỗng.\n\nJD:\n\n"
#     ) + text + "\n\nJSON:"

#     try:
#         resp = openai.ChatCompletion.create(
#             model=model,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.0,
#             max_tokens=512,
#         )
#         content = resp["choices"][0]["message"]["content"].strip()
#         # Try to parse JSON from response (robust)
#         try:
#             j = json.loads(content)
#             return j
#         except Exception:
#             # attempt to find JSON substring
#             import re
#             m = re.search(r"\{.*\}", content, flags=re.S)
#             if m:
#                 try:
#                     return json.loads(m.group(0))
#                 except:
#                     pass
#             # fallback: empty
#             return {"skills": [], "domains": []}
#     except Exception as e:
#         raise RuntimeError(f"LLM extract skills failed: {e}")


# # -------------------------
# # Main pipeline
# # -------------------------
# def process_single(
#     raw_text: str,
#     cleaner: JDProcessorCleaner,
#     summarizer_local: JDSummarizer,
#     extractor_local: SkillExtractor,
#     exp_mapper: ExperienceMapper,
#     use_llm: bool = False,
#     llm_model: str = "gpt-3.5-turbo",
#     llm_api_key: str = None,
#     max_summary_sentences: int = 5
# ) -> Dict[str, Any]:
#     """
#     Xử lý 1 JD: trả về dict với các trường.
#     """
#     out = {
#         "original_text": raw_text,
#         "clean_text": "",
#         "summary": "",
#         "skills": [],
#         "domains": [],
#         "years": None,
#         "level": None
#     }

#     # 1) Clean
#     clean = cleaner.clean(raw_text)
#     out["clean_text"] = clean

#     # 2) Summarize (LLM hoặc local)
#     if use_llm:
#         try:
#             summary = llm_summarize(clean, model=llm_model, max_sentences=max_summary_sentences, api_key=llm_api_key)
#         except Exception as e:
#             # fallback to local summarizer
#             summary = summarizer_local.summarize(clean, max_sentences=max_summary_sentences)
#     else:
#         summary = summarizer_local.summarize(clean, max_sentences=max_summary_sentences)
#     out["summary"] = summary

#     # 3) Skill extraction
#     if use_llm:
#         try:
#             res = llm_extract_skills(summary, model=llm_model, api_key=llm_api_key)
#             skills = res.get("skills", []) if isinstance(res, dict) else []
#             domains = res.get("domains", []) if isinstance(res, dict) else []
#         except Exception:
#             res_local = extractor_local.extract(summary)
#             skills = res_local.get("skills", [])
#             domains = res_local.get("domains", [])
#     else:
#         res_local = extractor_local.extract(summary)
#         skills = res_local.get("skills", [])
#         domains = res_local.get("domains", [])

#     out["skills"] = skills
#     out["domains"] = domains

#     # 4) Experience mapping
#     exp = exp_mapper.process(raw_text)
#     out["years"] = exp.get("years", 0)
#     out["level"] = exp.get("level", "")

#     return out

# def main():
#     parser = argparse.ArgumentParser(description="Process JD pipeline (clean -> summarize -> extract skills -> map experience)")
#     parser.add_argument("--csv", required=True, help="Path to jds.csv")
#     parser.add_argument("--text_col", default=None, help="Column name containing JD text (optional)")
#     parser.add_argument("--out_dir", default="outputs/embeddings", help="Output directory")
#     parser.add_argument("--use_llm", type=int, default=0, help="1 to use OpenAI LLM for summarize+extract. Requires OPENAI_API_KEY.")
#     parser.add_argument("--openai_model", default="gpt-3.5-turbo", help="OpenAI model to use if use_llm=1")
#     parser.add_argument("--batch_size", type=int, default=50, help="Batch size for encoding/LLM calls (if applicable)")
#     parser.add_argument("--max_summary_sentences", type=int, default=5)
#     args = parser.parse_args()

#     df = pd.read_csv(args.csv)
#     # auto-detect text column if not provided (same logic as encode_jds_csv)
#     text_col = args.text_col
#     if text_col is None:
#         for cand in ["full_content_clean", "jd_text", "description", "text"]:
#             if cand in df.columns:
#                 text_col = cand
#                 break
#         if text_col is None:
#             raise ValueError("Không tìm thấy cột text. Vui lòng cung cấp --text_col.")

#     os.makedirs(args.out_dir, exist_ok=True)

#     cleaner = JDProcessorCleaner()
#     summarizer_local = JDSummarizer()  # may download model if not present
#     extractor_local = SkillExtractor()
#     exp_mapper = ExperienceMapper()

#     use_llm = bool(args.use_llm)
#     llm_api_key = None
#     if use_llm:
#         # check openai lib + env var
#         if not USE_OPENAI:
#             raise RuntimeError("openai package not installed. Install it to use LLM mode.")
#         # prefer env var
#         if "OPENAI_API_KEY" in os.environ:
#             llm_api_key = os.environ["OPENAI_API_KEY"]
#         else:
#             raise RuntimeError("OPENAI_API_KEY not set in environment. Export it or set before running.")

#     results = []
#     err_count = 0

#     pbar = tqdm(range(0, len(df), args.batch_size), desc="Batches")
#     for i in pbar:
#         batch = df.iloc[i:i+args.batch_size]
#         for _, row in batch.iterrows():
#             raw = str(row.get(text_col, "") or "")
#             try:
#                 processed = process_single(
#                     raw_text=raw,
#                     cleaner=cleaner,
#                     summarizer_local=summarizer_local,
#                     extractor_local=extractor_local,
#                     exp_mapper=exp_mapper,
#                     use_llm=use_llm,
#                     llm_model=args.openai_model,
#                     llm_api_key=llm_api_key,
#                     max_summary_sentences=args.max_summary_sentences
#                 )
#                 # attach some meta if available
#                 if "title" in row:
#                     processed["title"] = row.get("title", "")
#                 if "company" in row:
#                     processed["company"] = row.get("company", "")
#                 results.append(processed)
#             except Exception as e:
#                 err_count += 1
#                 print(f"Error processing row idx {i}: {e}")
#                 results.append({
#                     "original_text": raw,
#                     "clean_text": "",
#                     "summary": "",
#                     "skills": [],
#                     "domains": [],
#                     "years": None,
#                     "level": None
#                 })

#     # Save CSV
#     df_out = pd.DataFrame(results)
#     csv_out = os.path.join(args.out_dir, "processed_jds.csv")
#     df_out.to_csv(csv_out, index=False, encoding="utf-8")
#     print("Saved:", csv_out)

#     # Save JSONL
#     jsonl_out = os.path.join(args.out_dir, "processed_jds.jsonl")
#     with open(jsonl_out, "w", encoding="utf-8") as f:
#         for rec in results:
#             f.write(json.dumps(rec, ensure_ascii=False) + "\n")
#     print("Saved:", jsonl_out)

#     print(f"Done. Processed {len(results)} records. Errors: {err_count}")


# if __name__ == "__main__":
#     main()
