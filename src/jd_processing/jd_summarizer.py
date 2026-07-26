# # # import os
# # # import torch
# # # from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# # # class JDSummarizer:
# # #     # Thêm tham số model_dir vào __init__
# # #     def __init__(self, model_dir="/Users/nhitruong/Documents/data_mining_project/models"):
# # #         print("   - Initializing Summarizer...")
# # #         self.model_name = "sshleifer/distilbart-cnn-12-6"
        
# # #         # Nếu truyền model_dir vào thì dùng, không thì dùng mặc định
# # #         self.cache_dir = model_dir
        
# # #         # Tạo thư mục nếu chưa có
# # #         if self.cache_dir:
# # #             os.makedirs(self.cache_dir, exist_ok=True)
        
# # #         device = 0 if torch.cuda.is_available() else -1
        
# # #         try:
# # #             # Tải model về thư mục chỉ định (nếu có cache_dir)
# # #             if self.cache_dir:
# # #                 tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
# # #                 model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, cache_dir=self.cache_dir)
# # #             else:
# # #                 tokenizer = AutoTokenizer.from_pretrained(self.model_name)
# # #                 model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
# # #             self.summarizer = pipeline(
# # #                 "summarization", 
# # #                 model=model, 
# # #                 tokenizer=tokenizer, 
# # #                 device=device
# # #             )
# # #         except Exception as e:
# # #             print(f"Warning: Could not load summarizer. {e}")
# # #             self.summarizer = None

# # #     def summarize(self, text, max_len=150):
# # #         """Tóm tắt JD dài thành đoạn ngắn gọn"""
# # #         if not self.summarizer or not text or len(text.split()) < 50:
# # #             return text[:300] + "..." if text else ""
            
# # #         try:
# # #             # Cắt ngắn input để tránh lỗi token limit của model
# # #             input_text = text[:3000]
# # #             summary = self.summarizer(
# # #                 input_text, 
# # #                 max_length=max_len, 
# # #                 min_length=30, 
# # #                 do_sample=False
# # #             )
# # #             return summary[0]['summary_text']
# # #         except Exception as e:
# # #             return text[:300] + "..."

# # # if __name__ == "__main__":
# # #     # Test nhanh
# # #     s = JDSummarizer()
# # #     text = "We are looking for a Senior Python Developer to join our engineering team and help us build functional software and web-based applications. Python Developer responsibilities include writing and testing code, debugging programs and integrating applications with third-party web services."
# # #     print("Summary:", s.summarize(text))

# # import os
# # from openai import OpenAI
# # from dotenv import load_dotenv

# # # Load biến môi trường từ file .env
# # load_dotenv()

# # class JDSummarizer:
# #     """
# #     Job Description Summarizer using GPT-3.5-Turbo.
# #     Tóm tắt JD thành đoạn ngắn (~150 tokens):
# #     - seniority level
# #     - required skills
# #     - responsibilities
# #     """

# #     def __init__(self, model_name="gpt-3.5-turbo", max_tokens=150):
# #         self.model_name = model_name
# #         self.max_tokens = max_tokens
        
# #         api_key = os.getenv("OPENAI_API_KEY")
# #         if not api_key:
# #             raise ValueError("❌ OPENAI_API_KEY is missing. Add it to your .env file.")

# #         self.client = OpenAI(api_key=api_key)
# #         print(f"   - OpenAI Summarizer initialized ({self.model_name}, ~{max_tokens} tokens)")

# #     def summarize(self, text):
# #         """Tóm tắt JD thành đoạn súc tích ~150 tokens."""

# #         if not text:
# #             return ""

# #         # JD quá ngắn thì không cần tóm tắt
# #         if len(text.split()) < 50:
# #             return text.strip()

# #         prompt = f"""
# # Summarize the following Job Description into ONE concise paragraph.
# # Hard limit: maximum {self.max_tokens} tokens.
# # Focus strictly on:
# # - seniority level
# # - core required skills
# # - key responsibilities

# # Do NOT include company info, culture, benefits, or noise.
# # Do NOT invent new details.

# # Job Description:
# # {text}
# # """

# #         try:
# #             response = self.client.chat.completions.create(
# #                 model=self.model_name,
# #                 messages=[{"role": "user", "content": prompt}],
# #                 temperature=0.0,
# #                 max_tokens=self.max_tokens * 2
# #             )

# #             summary = response.choices[0].message.content.strip()
# #             return summary

# #         except Exception as e:
# #             print(f"⚠️ OpenAI Summarization Error: {e}")
# #             return text[:300] + "..."  # fallback


# # # ----------------------------
# # # QUICK TEST
# # # ----------------------------
# # if __name__ == "__main__":
# #     jd = """
# #     We are seeking a Senior Data Scientist with 5+ years of experience
# #     in Python, SQL, ML, DL, and cloud. Responsibilities include building
# #     predictive models, leading DS projects, collaborating with engineering,
# #     and deploying ML pipelines. Experience with NLP or MLOps is a plus.
# #     """

# #     s = JDSummarizer()
# #     print("\nSUMMARY:\n", s.summarize(jd))

# # src/jd_processing/jd_summarizer.py
# import os
# from dotenv import load_dotenv
# import openai
# import time

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if OPENAI_API_KEY:
#     openai.api_key = OPENAI_API_KEY

# class JDSummarizer:
#     """
#     Summarizer dùng OpenAI GPT-3.5 (gpt-3.5-turbo).
#     __init__ chấp nhận model_dir để tương thích với main.py (không bắt buộc).
#     """

#     def __init__(self, model_dir=None, model_name="gpt-3.5-turbo", default_max_tokens=180):
#         # model_dir: chỉ để tương thích, không dùng trong implementation hiện tại
#         self.model_name = model_name
#         self.default_max_tokens = default_max_tokens

#     def summarize(self, text: str, max_tokens=None):
#         """
#         Trả về summary (string). Nếu OpenAI không cấu hình hoặc call lỗi -> fallback trả text rút gọn.
#         """
#         if not text:
#             return ""

#         max_tokens = max_tokens or self.default_max_tokens

#         prompt = (
#             "Bạn là trợ lý chuyên tóm tắt mô tả công việc. "
#             "Hãy tóm tắt nội dung JD dưới đây thành 3-5 câu ngắn gọn, "
#             "nêu rõ vai trò chính, kỹ năng cốt lõi, công nghệ quan trọng và "
#             "yêu cầu kinh nghiệm nếu có. Trả về văn bản ngắn gọn đủ để làm input cho embedding.\n\n"
#             f"JD:\n{text}"
#         )

#         # Nếu API key không có, trả fallback nhanh
#         if not OPENAI_API_KEY:
#             # fallback: cắt bớt text (để dùng làm embedding)
#             return text[:600]

#         # call OpenAI (wrapped try/except)
#         try:
#             resp = openai.ChatCompletion.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.2,
#                 max_tokens=max_tokens,
#             )
#             summary = resp["choices"][0]["message"]["content"].strip()
#             return summary
#         except Exception as e:
#             # log để debug
#             print(f"[WARN] JDSummarizer LLM call failed: {e}")
#             # fallback: safe substring
#             # Thêm sleep nhẹ để tránh bị rate-limit loop nếu này chạy trong batch
#             time.sleep(0.1)
#             return text[:600]
