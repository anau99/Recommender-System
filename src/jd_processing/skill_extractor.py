# # import pandas as pd
# # import os
# # import re

# # class SkillExtractor:
# #     def __init__(self, meta_path):
# #         self.skills_vocab = set()
        
# #         # Danh sách từ vô nghĩa CƠ BẢN + CÁC TỪ MÔ TẢ GÂY NHIỄU (Descriptors)
# #         self.stop_words = {
# #             'the', 'and', 'of', 'to', 'in', 'a', 'for', 'on', 'with', 'as', 'by', 'at', 'an', 'be', 
# #             'this', 'that', 'from', 'or', 'is', 'are', 'was', 'were', 'will', 'can', 'should',
# #             'based', 'using', 'learning', 'introduction', 'basics', 'fundamental', 'advanced',
# #             'course', 'project', 'capstone', 'specialization', 'professional', 'certificate',
# #             # -- BỔ SUNG LỚP TỪ VÔ NGHĨA CAO CẤP HƠN --
# #             'good', 'understanding', 'ability', 'read', 'understand', 'technical', 
# #             'documents', 'related', 'proficiency', 'experience', 'systems', 'planning', 
# #             'vision', 'processing', 'computer', 'deep', 'control', 'motion', 'navigation', 
# #             'detection', 'segmentation', 'tracking', 'pipeline', 'environments', 'hands',
# #             'years', 'require', 'bachelor', 'degree', 'master', 'phd'
# #         }
        
# #         # Danh sách các cụm kỹ năng quan trọng cần được bắt dính riêng
# #         self.hardcoded_phrases = [
# #             'machine learning', 'deep learning', 'computer vision', 'data analysis',
# #             'business analyst', 'financial analysis', 'agile scrum', 'object detection'
# #         ]
        
# #         if os.path.exists(meta_path):
# #             print(f"Loading skills from {meta_path}...")
# #             df = pd.read_csv(meta_path)
            
# #             vocab_source = []
# #             if 'skills_gain_clean' in df.columns:
# #                 vocab_source.extend(df['skills_gain_clean'].dropna().astype(str).tolist())
# #             if 'topic' in df.columns:
# #                 vocab_source.extend(df['topic'].dropna().astype(str).tolist())
                
# #             full_text = " ".join(vocab_source).lower()
            
# #             # Chỉ lấy các từ có độ dài > 2 và không nằm trong stop_words
# #             tokens = set(re.findall(r'[\w\+\#\.]+', full_text))
            
# #             # Lọc từ rác
# #             self.skills_vocab = {w for w in tokens if len(w) > 2 and w not in self.stop_words}
            
# #             # Bổ sung danh sách kỹ năng cứng (Hard skills) quan trọng
# #             hardcoded_skills = {
# #                 'ros', 'ros2', 'python', 'java', 'c++', 'c#', '.net', 'sql', 'mysql', 'postgresql', 
# #                 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'linux', 'git', 'tableau', 'power bi', 
# #                 'excel', 'communication', 'leadership', 'management', 'agile', 'scrum', 'marketing', 
# #                 'finance', 'html', 'css', 'javascript', 'react', 'angular', 'node', 'django', 'flask',
# #                 'tensorflow', 'pytorch', 'opencv', 'slam'
# #             }
# #             self.skills_vocab.update(hardcoded_skills)
            
# #             print(f"Skill Vocab Size (Cleaned): {len(self.skills_vocab)}")
# #         else:
# #             print(f"Warning: Meta file {meta_path} not found. Using empty vocab.")

# #     def extract(self, text):
# #         if not text: return []
        
# #         lower_text = text.lower()
        
# #         # 1. Tách từ trong JD (unigram)
# #         jd_tokens = set(re.findall(r'[\w\+\#\.]+', lower_text))
        
# #         # 2. Tìm điểm chung giữa token JD và vocab đã lọc
# #         found_tokens = list(jd_tokens.intersection(self.skills_vocab))
        
# #         # 3. Tìm các cụm từ (phrases/bigrams) quan trọng
# #         found_phrases = []
# #         for phrase in self.hardcoded_phrases:
# #             if phrase in lower_text:
# #                 found_phrases.append(phrase)
# #                 # Loại bỏ các từ đơn của cụm đó khỏi danh sách tokens để tránh trùng lặp/nhiễu
# #                 for token in phrase.split():
# #                     if token in found_tokens:
# #                         found_tokens.remove(token)
                        
# #         # 4. Loại bỏ các từ stop_word nếu lọt lưới (lọc lần cuối)
# #         clean_tokens = [w for w in found_tokens if w not in self.stop_words]
        
# #         # Kết hợp và trả về
# #         return clean_tokens + found_phrases

# # # Phần Test (giữ nguyên)
# # if __name__ == "__main__":
# #     BASE_DIR = "/Users/nhitruong/Documents/data_mining_project"
# #     meta_file = f"{BASE_DIR}/outputs/embeddings/course_meta.csv"
# #     extractor = SkillExtractor(meta_file)
    
# #     # Test case của bạn:
# #     jd_robotics = """
# #     Bachelor’s degree in Computer Science, Robotics, AI, Mechatronics, Electronics, or related fields.
# #     1–3+ years of hands-on experience with ROS / ROS2.
# #     Proficiency in Python and/or C++ in robotics environments.
# #     Experience in one or more of the following areas:
# #     - Image processing: OpenCV, image processing pipeline
# #     - Deep learning: TensorFlow / PyTorch
# #     - Computer vision: object detection, segmentation, tracking
# #     Good understanding of robotic systems: SLAM, navigation, motion planning, control.
# #     Ability to read and understand technical documents in English.
# #     """
    
# #     # Kết quả kỳ vọng sau khi lọc: ROS, Python, C++, Deep learning, Computer vision, TensorFlow, PyTorch, OpenCV, SLAM, etc.
# #     print(f"Extracted: {extractor.extract(jd_robotics)}")

# #     jd_agile = "looking for a python developer with sql and communication skills based on agile"
# #     print(f"Extracted: {extractor.extract(jd_agile)}")
# #     # Kỳ vọng: ['python', 'sql', 'communication', 'agile']

# import os
# import re
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# class SkillExtractor:
#     """
#     Skill extractor dùng GPT-3.5 để trích xuất skill theo JSON.
#     Có fallback regex nếu API lỗi.
#     """

#     def __init__(self, meta_path=None, model_name="gpt-3.5-turbo"):
#         self.model_name = model_name

#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY missing in .env")

#         self.client = OpenAI(api_key=api_key)

#         # fallback vocab (sử dụng khi API lỗi)
#         self.fallback_vocab = {
#             "python","java","c++","c#","sql","mysql","postgresql",
#             "tensorflow","pytorch","opencv","docker","kubernetes",
#             "aws","azure","gcp","react","node","linux","git",
#             "slam","ros","ros2","nlp","ml","dl"
#         }

#         print("   - SkillExtractor initialized using GPT-3.5")

#     # =========================================================
#     # 1. LLM EXTRACTION (ƯU TIÊN HÀNG ĐẦU)
#     # =========================================================
#     def extract_skills_llm(self, text):
#         prompt = f"""
# Extract ALL technical skills mentioned in the Job Description below.
# Return ONLY valid JSON array of strings, no explanation.

# Example output:
# ["python","sql","docker","react"]

# Job Description:
# {text}
# """

#         try:
#             resp = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0,
#                 max_tokens=200
#             )

#             raw = resp.choices[0].message.content.strip()

#             # ensure valid JSON list
#             if raw.startswith("{") or raw.startswith("skills"):
#                 raw = raw[raw.find("[") : raw.rfind("]") + 1]

#             import json
#             skills = json.loads(raw)

#             # chuẩn hóa
#             clean = sorted(set([s.strip().lower() for s in skills if len(s.strip()) > 1]))
#             return clean

#         except Exception as e:
#             print("LLM skill extraction failed:", e)
#             return None

#     # =========================================================
#     # 2. FALLBACK REGEX (KHI API LỖI)
#     # =========================================================
#     def extract_skills_regex(self, text):
#         text = text.lower()
#         tokens = set(re.findall(r"[\w\+\#\.]+", text))
#         hits = sorted(tokens.intersection(self.fallback_vocab))
#         return hits

#     # =========================================================
#     # 3. PUBLIC API: extract()
#     # =========================================================
#     def extract(self, text):
#         if not text:
#             return []

#         # Ưu tiên dùng LLM
#         skills = self.extract_skills_llm(text)
#         if skills:
#             return skills

#         # Nếu lỗi → fallback
#         return self.extract_skills_regex(text)


# # ===============================
# # TEST
# # ===============================
# if __name__ == "__main__":
#     jd_robotics = """
#     Bachelor’s degree in Computer Science, Robotics.
#     Experience with ROS, ROS2, Python, C++, OpenCV, SLAM, TensorFlow, PyTorch.
#     Knowledge of navigation, motion planning and computer vision.
#     """

#     extractor = SkillExtractor(meta_path=None)
#     print("Extracted skills:", extractor.extract(jd_robotics))
