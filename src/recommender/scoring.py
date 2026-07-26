
import numpy as np
import re

class Scorer:
    """
    Scorer kết hợp:
    - semantic_score
    - bm25_score
    - cluster alignment
    - level alignment
    - skill matching

    Trả về:
    - final_score
    - breakdown (dùng cho Explainability + UI)
    """

    def __init__(self, weights=None):

        self.weights = weights or {
            "semantic": 0.6,
            "bm25": 0.4,
            "cluster_boost": 1.10,
            "level_boost": 1.05,
            "skill_bonus": 0.05
        }

    def _normalize_level(self, text):
        if not text: return ""
        t = str(text).lower()
        if "senior" in t or "expert" in t or t.startswith("sr"):
            return "advanced"
        if "mid" in t or "inter" in t:
            return "intermediate"
        if "junior" in t or "entry" in t or "beginner" in t:
            return "beginner"
        return t

    def _get_course_skill_set(self, candidate):
        for k in ("skills", "skills_list", "skills_gain_clean", "course_skills"):
            if candidate.get(k):
                txt = str(candidate[k]).lower()
                return set(re.findall(r"[a-zA-Z0-9\+\#\.]+", txt))

        # fallback: clean_text
        txt = (
            str(candidate.get("clean_text", "")).lower() + " " +
            str(candidate.get("course_name", "")).lower()
        )
        return set(re.findall(r"[a-zA-Z0-9\+\#\.]+", txt))

    # -----------------------------
    # MAIN FUNCTION
    # -----------------------------
    def calculate_score(self, candidate, user_profile, jd_cluster_id):
        """
        Return:
            model_score  (float)
            breakdown    (dict)
        """

        # ============================================
        # 1. SEMANTIC + BM25 COMBINATION
        # ============================================

        semantic = float(candidate.get("semantic_score", 0))
        bm25 = float(candidate.get("bm25_score", 0))

        w_sem = self.weights.get("semantic", 0.6)
        w_bm = self.weights.get("bm25", 0.4)
        norm = w_sem + w_bm

        w_sem /= norm
        w_bm /= norm

        model_score = semantic * w_sem + bm25 * w_bm

        # ============================================
        # 2. CLUSTER BOOST (multiplicative)
        # ============================================

        cluster_match = False
        if candidate.get("cluster_id") == jd_cluster_id:
            model_score *= self.weights.get("cluster_boost", 1.0)
            cluster_match = True

        # ============================================
        # 3. LEVEL BOOST
        # ============================================

        level_match = False
        user_lvl = self._normalize_level(user_profile.get("level"))
        course_lvl = self._normalize_level(candidate.get("level"))

        if user_lvl and course_lvl and user_lvl == course_lvl:
            model_score *= self.weights.get("level_boost", 1.0)
            level_match = True

        # ============================================
        # 4. SKILL MATCHING (ADD-ON, không nhân)
        # ============================================

        course_skill_set = self._get_course_skill_set(candidate)
        jd_skills = [s.lower() for s in (user_profile.get("skills") or [])]

        overlap = set(jd_skills).intersection(course_skill_set)
        skill_ratio = len(overlap) / max(1, len(jd_skills))

        skill_effect = skill_ratio * self.weights.get("skill_bonus", 0.05)

        # additive 
        model_score += skill_effect

        # ============================================
        # 5. BREAKDOWN
        # ============================================

        breakdown = {
            "model_score": round(float(model_score), 6),
            "semantic_contrib": semantic,
            "bm25_contrib": bm25,

            "cluster_match": cluster_match,
            "level_match": level_match,

            "skill_contrib": round(float(skill_effect), 6),
            "matched_skills": list(overlap),
            "skill_ratio": round(float(skill_ratio), 4),
        }

        return float(model_score), breakdown

