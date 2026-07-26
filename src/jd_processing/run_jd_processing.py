# import sys
# import os

# # Đảm bảo Python nhìn thấy thư mục src
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# from src.jd_processing.jd_cleaner import JDCleaner
# from src.jd_processing.jd_summarizer import JDSummarizer
# from src.jd_processing.skill_extractor import SkillExtractor
# from src.jd_processing.experience_mapper import ExperienceMapper

# class JDProcessor:
#     def __init__(self):
#         print("⚙️ KHỞI TẠO HỆ THỐNG XỬ LÝ JD...")
#         self.cleaner = JDCleaner()
#         self.summarizer = JDSummarizer()
#         # Skill Extractor sẽ tự tải model vào /workspace/.../models nếu chưa có
#         self.extractor = SkillExtractor() 
#         self.mapper = ExperienceMapper()
#         print("✅ Hệ thống đã sẵn sàng!")

#     def process(self, raw_text: str):
#         """Pipeline chạy 4 bước tuần tự"""
#         # B1: Làm sạch
#         clean_text = self.cleaner.clean(raw_text)
        
#         # B2: Tóm tắt / Cắt gọn
#         summary_text = self.summarizer.summarize(clean_text)
        
#         # B3 & B4: Trích xuất thông tin (Song song)
#         skill_data = self.extractor.extract(summary_text)
#         exp_data = self.mapper.map_experience(summary_text)
        
#         # Tổng hợp kết quả
#         return {
#             "processed_text": summary_text,
#             "skills": skill_data['skills'],
#             "domains": skill_data['domains'],
#             "level": exp_data['level'],
#             "years_of_experience": exp_data['years']
#         }

# # --- PHẦN CHẠY THỬ (MAIN) ---
# if __name__ == "__main__":
#     # 1. Input giả lập
#     sample_jd = """
#     <html>
#     <h1>Tuyển Senior Python Developer (Lương cao)</h1>
#     <p>Yêu cầu: Có ít nhất 4 năm kinh nghiệm làm việc với Django, Flask.</p>
#     <p>Thành thạo SQL và AWS. Ưu tiên ứng viên biết Machine Learning.</p>
#     <p>Quyền lợi: Du lịch 2 lần/năm. Liên hệ: hr@congty.com</p>
#     </html>
#     """
    
#     # 2. Khởi tạo bộ xử lý
#     processor = JDProcessor()
    
#     # 3. Chạy xử lý
#     print("\n--- ⏳ ĐANG XỬ LÝ JD ĐẦU VÀO ---")
#     result = processor.process(sample_jd)
    
#     # 4. In kết quả
#     print("\n🎉 KẾT QUẢ CUỐI CÙNG:")
#     print(f"▶ Level: {result['level']} ({result['years_of_experience']} năm)")
#     print(f"▶ Domains: {result['domains']}")
#     print(f"▶ Skills: {result['skills']}")
#     print("-" * 50)