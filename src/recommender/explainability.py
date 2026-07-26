
import json

class Explainability:
    def generate_explanation(self, course_title, breakdown):
        """
        Nhận breakdown từ Scorer và tạo giải thích tự nhiên.
        """

        reasons = []

        # 1) Semantic Matching
        if breakdown.get("semantic_contrib", 0) > 0.5:
            reasons.append("Khóa học có mức độ tương đồng ngữ nghĩa cao với JD.")
        elif breakdown.get("semantic_contrib", 0) > 0.25:
            reasons.append("Khóa học phù hợp tương đối với nội dung JD.")

        # 2) Skill Matching
        skill_contrib = breakdown.get("skill_contrib", 0)
        matched_skills = breakdown.get("matched_skills", [])

        if skill_contrib > 0.05 and matched_skills:
            reasons.append(
                f"Khóa học bao phủ nhiều kỹ năng quan trọng mà JD yêu cầu: {', '.join(matched_skills[:5])}."
            )
        elif skill_contrib > 0:
            reasons.append("Khóa học có liên quan đến một số kỹ năng trong JD.")

        # 3) Cluster match
        if breakdown.get("cluster_match"):
            reasons.append("Khóa học thuộc cùng nhóm chủ đề (cluster) với JD.")

        # 4) Level match
        if breakdown.get("level_match"):
            reasons.append("Trình độ khóa học phù hợp với mức độ yêu cầu trong JD.")

        # 5) Business Rules
        rule_multiplier = breakdown.get("rule_multiplier", 1.0)
        if rule_multiplier > 1.05:
            reasons.append("Khóa học được ưu tiên theo các quy tắc nghiệp vụ (business rules).")
        elif rule_multiplier < 0.95:
            reasons.append("Khóa học bị giảm điểm do không khớp với một số yêu cầu quan trọng trong JD.")

        # 6) Nếu không có lý do nào → thêm fallback
        if not reasons:
            reasons.append("Khóa học được chọn dựa trên mức độ phù hợp tổng thể với JD.")

        explanation = {
            "course": course_title,
            "model_score": round(breakdown.get("model_score", 0), 4),
            "final_score": round(breakdown.get("final_score", 0), 4),
            "key_factors": reasons,
            "details": breakdown  
        }

        return json.dumps(explanation, ensure_ascii=False, indent=2)
