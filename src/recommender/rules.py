import re
def compute_experience_multiplier(user_years, jd_years):
   
    if user_years is None or jd_years is None:
        return 1.0

    try:
        u = float(user_years)
        j = float(jd_years)
    except Exception:

        return 1.0

    diff = u - j

    if diff >= 0:
        return 1.05

    if diff > -1:
        return 1.0

    if diff > -3:
        return 0.9

    return 0.7



class BusinessRules:
    def __init__(self):
        self.multipliers = {
    
            "missing_core_penalty": 0.5,
            "jd_adv_course_beginner_penalty": 0.7,
            "jd_beginner_course_adv_penalty": 0.5,
            "certificate_boost": 1.2
        }

    # -------------- helpers --------------
    def _simple_normalize(self, s):
        """
        Chuẩn hoá nhẹ một skill/chuỗi:
        - lowercase, strip
        - loại bớt ký tự punctuation thừa (nhưng giữ multi-word)
        """
        if not s:
            return ""
        t = str(s).strip().lower()
        t = re.sub(r"[()\[\]\{\}\.,;:/\\\*\"']", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _normalize_skills_from_llm(self, raw_skills):
        out = []
        if raw_skills is None:
            return out
        if isinstance(raw_skills, str):
            candidates = [s.strip() for s in re.split(r"[,\n;]+", raw_skills) if s.strip()]
        elif isinstance(raw_skills, (list, tuple)):
            candidates = list(raw_skills)
        else:
            candidates = [str(raw_skills)]

        for s in candidates:
            n = self._simple_normalize(s)
            if n and n not in out:
                out.append(n)
        return out

    def _normalize_candidate_skills(self, candidate_item):
        for k in ("skills", "course_skills", "skills_list", "skills_gain_clean"):
            v = candidate_item.get(k)
            if not v:
                continue
            if isinstance(v, (list, tuple)):
                out = []
                for s in v:
                    n = self._simple_normalize(s)
                    if n and n not in out:
                        out.append(n)
                return out
            if isinstance(v, str):
                parts = [s.strip() for s in re.split(r"[,\n;]+", v) if s.strip()]
                out = []
                for s in parts:
                    n = self._simple_normalize(s)
                    if n and n not in out:
                        out.append(n)
                return out
            try:
                txt = str(v)
                parts = [s.strip() for s in re.split(r"[,\n;]+", txt) if s.strip()]
                out = []
                for s in parts:
                    n = self._simple_normalize(s)
                    if n and n not in out:
                        out.append(n)
                return out
            except Exception:
                continue
        return []

    def _tokenize_course_text(self, candidate_item):
        txt = (str(candidate_item.get('course_name', '')) + " " + str(candidate_item.get('clean_text', ''))).lower()
        for k in ("skills", "course_skills", "skills_list", "skills_gain_clean"):
            if candidate_item.get(k):
                txt += " " + str(candidate_item.get(k))
        tokens = [t for t in re.split(r"[^a-z0-9\+\#]+", txt) if t]
        out = []
        for t in tokens:
            n = self._simple_normalize(t)
            if n and n not in out:
                out.append(n)
        return out
    def _detect_domain_from_skills(self, jd_skills):
  
        if not jd_skills:
            return "general"
        jd = set(s.lower() for s in jd_skills)
        DOMAIN_KEYWORDS = {
            "robotics": {"ros", "ros2", "slam", "opencv", "pytorch", "tensorflow", "navigation", "robot"},
            "data": {"machine learning", "deep learning", "sql", "pandas", "python", "nlp", "transformers"},
            "ba": {"business analyst", "bpmn", "requirements", "elicitation", "stakeholder", "agile", "scrum"},
            "devops": {"docker", "kubernetes", "ci", "cd", "jenkins", "terraform", "ansible", "mlops"},
            "software": {"java", "c++", "spring", "react", "api", "backend", "microservices"},
            "cloud": {"aws", "azure", "gcp", "cloud", "lambda"}
        }
        scores = {d: len(jd.intersection(kws)) for d, kws in DOMAIN_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "general"
        return best

    def apply_rules(self, user_profile, candidate_item):
     
        factor = 1.0

        raw_jd_skills = user_profile.get('skills') or []
        jd_skills = self._normalize_skills_from_llm(raw_jd_skills)

        jd_domain = user_profile.get('domain') or ""
        if not jd_domain or not isinstance(jd_domain, str) or jd_domain.strip() == "":
            jd_domain = self._detect_domain_from_skills(jd_skills)
        jd_domain = str(jd_domain).lower()

        # level
        jd_level_raw = user_profile.get('level') or ""
        jd_level = str(jd_level_raw).lower()
        course_level = str(candidate_item.get('level') or "").lower()
        course_skills = self._normalize_candidate_skills(candidate_item)
        if not course_skills:
            course_skills = self._tokenize_course_text(candidate_item)

        jd_core_set = set(jd_skills)
        course_set = set(course_skills)

        matched = list(sorted(jd_core_set.intersection(course_set)))
        # lưu matched skills vào candidate để explainability
        try:
            candidate_item["matched_skills"] = matched
        except Exception:
            pass

        if jd_core_set:
            if len(matched) == 0:
                factor *= float(self.multipliers.get("missing_core_penalty", 0.5))
            else:
                ratio = len(matched) / max(1.0, len(jd_core_set))
                if ratio < 0.2:
                    factor *= 0.9

        if "advanced" in jd_level and "beginner" in course_level:
            factor *= float(self.multipliers.get("jd_adv_course_beginner_penalty", 0.7))
        if "beginner" in jd_level and "advanced" in course_level:
            factor *= float(self.multipliers.get("jd_beginner_course_adv_penalty", 0.5))

        # Certificate preference
        user_title = str(user_profile.get('title', '')).lower()
        user_summary = str(user_profile.get('summary', '')).lower()
        if 'certificate' in user_title or 'certificate' in user_summary or user_profile.get('prefers_certificate'):
            course_text = (str(candidate_item.get('course_name', '')) + " " + str(candidate_item.get('clean_text', ''))).lower()
            if 'certificate' in course_text or 'certification' in course_text:
                factor *= float(self.multipliers.get("certificate_boost", 1.2))

        # Experience multiplier 
        user_years = user_profile.get("user_years")
        jd_years = user_profile.get("years_exp")
        exp_mult = compute_experience_multiplier(user_years, jd_years)
        factor *= exp_mult
        try:
            candidate_item["exp_multiplier"] = float(exp_mult)
        except Exception:
            candidate_item["exp_multiplier"] = exp_mult

        return float(factor)
