
import os
import sys
import json
import time
import argparse
import re
import logging

import os
import sys
import json
import time
import argparse

from dotenv import load_dotenv
load_dotenv()

# Khi chạy như script, đảm bảo src được trong sys.path
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Thiết lập logger đơn giản in ra terminal
logger = logging.getLogger("course_recommender")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)



def safe_import():
    global JDCleaner, ExperienceMapper
    global EmbeddingModel, ClusterAssigner
    global SemanticSearcher, BM25Searcher, CandidateUnion
    global Scorer, BusinessRules, Explainability

    try:
        from src.jd_processing.jd_cleaner import JDCleaner
        from src.jd_processing.experience_mapper import ExperienceMapper
        from src.embedding.embedding_model import EmbeddingModel
        from src.clustering.cluster_assigner import ClusterAssigner
        from src.retrieval.semantic_search import SemanticSearcher
        from src.retrieval.bm25_retriever import BM25Searcher
        from src.retrieval.candidate_union import CandidateUnion
        from src.recommender.scoring import Scorer
        from src.recommender.rules import BusinessRules
        from src.recommender.explainability import Explainability
    except ImportError:
        # Fallback tương đối (khi chạy file trực tiếp trong thư mục src)
        from jd_processing.jd_cleaner import JDCleaner
        from jd_processing.experience_mapper import ExperienceMapper
        from embedding.embedding_model import EmbeddingModel
        from clustering.cluster_assigner import ClusterAssigner
        from retrieval.semantic_search import SemanticSearcher
        from retrieval.bm25_retriever import BM25Searcher
        from retrieval.candidate_union import CandidateUnion
        from recommender.scoring import Scorer
        from recommender.rules import BusinessRules
        from recommender.explainability import Explainability



safe_import()


class CourseRecommenderSystem:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir if base_dir else project_root
        logger.info("=" * 60)
        logger.info(f"SYSTEM INITIALIZED (Root: {self.base_dir})")
        logger.info("=" * 60)

        self.paths = {
            "meta": f"{self.base_dir}/outputs/embeddings/course_meta.csv",
            "faiss": f"{self.base_dir}/outputs/indices/faiss_index.bin",
            "bm25": f"{self.base_dir}/outputs/models/bm25_model.pkl",
            "pca": f"{self.base_dir}/outputs/models/pca_model.pkl",
            "kmeans": f"{self.base_dir}/outputs/models/kmeans_model.pkl",
            "summ_model": f"{self.base_dir}/models",
            "best_weights": f"{self.base_dir}/outputs/models/best_weights.json"
        }

        if os.path.exists(self.paths["meta"]):
            import pandas as pd
            self.course_df = pd.read_csv(self.paths["meta"])
            if 'emb_index' not in self.course_df.columns:
                self.course_df['emb_index'] = self.course_df.index
            self.course_lookup = self.course_df.set_index('emb_index').to_dict('index')
            logger.info(f"Metadata loaded: {len(self.course_df)} courses")
        else:
            logger.warning("Meta file missing: %s", self.paths["meta"])
            self.course_df = None
            self.course_lookup = {}

        # 2. Load processors
        logger.info("[1/4] Loading Processors...")
        self.cleaner = JDCleaner()
        self.exp_mapper = ExperienceMapper()
      
        # 3. Load AI models
        logger.info("[2/4] Loading AI Models...")
        self.embedder = EmbeddingModel()
        self.cluster_assigner = ClusterAssigner(self.paths["pca"], self.paths["kmeans"])

        # 4. Load retrieval engines
        logger.info("[3/4] Loading Retrieval Engines...")
        self.semantic_searcher = SemanticSearcher(self.paths["faiss"])
        self.bm25_searcher = BM25Searcher(model_path=self.paths["bm25"])
        self.retriever = CandidateUnion(self.semantic_searcher, self.bm25_searcher)

        # 5. Load logic modules
        logger.info("[4/4] Loading Logic & Rules...")
        self.rules = BusinessRules()
        self.explainer = Explainability()

        # load optimized weights nếu có
        weights = None
        if os.path.exists(self.paths["best_weights"]):
            with open(self.paths["best_weights"], "r") as f:
                weights = json.load(f)
            logger.info("Loaded optimized weights from %s", self.paths["best_weights"])
        self.scorer = Scorer(weights=weights)

        logger.info("READY!\n")

    def recommend(self, raw_jd, user_years=None, top_k=10):
        """
        Quy trình recommend (phiên bản LLM extractor):
        - Clean JD
        - Gọi jd_quick_extractor.extract_jd_info một lần để lấy {summary, skills, domain}
        - Nếu extractor thất bại (ví dụ thiếu openai key), fallback nhẹ
        - Embed summary, assign cluster, retrieve, score, apply rules
        - Trả về kết quả giống format cũ
        """
        t0 = time.time()

        # B1. Clean JD 
        clean_text = self.cleaner.clean(raw_jd)

        # B2. Import extractor 
        extractor = None
        try:
            from src.jd_processing.jd_quick_extractor import extract_jd_info
            extractor = extract_jd_info
        except Exception:
            try:
                from jd_processing.jd_quick_extractor import extract_jd_info
                extractor = extract_jd_info
            except Exception as e:
                extractor = None
                logger.warning("Không import được jd_quick_extractor: %s", e)

     
        summary = ""
        skills = []
        domain = None
        llm_used = False

        if extractor:
            try:
          
                api_key = os.getenv("OPENAI_API_KEY")
              
                extractor_out = extractor(raw_jd, openai_api_key=api_key, model="gpt-3.5-turbo")
                if isinstance(extractor_out, dict):
                    summary = extractor_out.get("summary", "") or ""
                    skills = extractor_out.get("skills", []) or []
                    domain = extractor_out.get("domain", "") or None
                    llm_used = bool(extractor_out.get("llm_used", False))
                   
                    llm_raw = extractor_out.get("llm_raw", None)
                    if llm_raw:
                        logger.info("LLM extractor produced output (truncated): %s", (llm_raw[:1000] + "...") if len(llm_raw) > 1000 else llm_raw)
                    logger.info("Extractor returned summary (len=%d), skills=%d, domain=%s", len(summary), len(skills), domain)
                else:
                    logger.warning("Extractor trả về không phải dict, dùng fallback.")
            except Exception as e:
                logger.warning("Extractor LLM thất bại: %s. Sẽ dùng fallback nhẹ.", e)
                # tiếp tục để fallback bên dưới
                summary = ""
                skills = []
                domain = None
                llm_used = False

  
        if not summary:
        
            sentences = re.split(r'(?<=[\.\!\?])\s+', clean_text)
            if len(sentences) >= 2:
                summary = " ".join(sentences[:2]).strip()
            else:
                summary = clean_text[:300].strip()
            logger.info("Fallback summary used (len=%d).", len(summary))

        if not skills:
   
            text_l = clean_text.lower()
            common_tokens = [
                "python","java","c++","c#","sql","postgres","postgresql","redis","oracle","ms sql","mssql",
                "spring","spring boot","docker","kubernetes","aws","azure","gcp","tensorflow","pytorch",
                "nlp","transformers","spark","hadoop","react","node","django","flask","rest","restful api",
                "graphql","mlops","ci/cd","jenkins","terraform","ros","opencv","slam"
            ]
            found = []
            for tok in sorted(common_tokens, key=lambda x: -len(x)):
                if tok in text_l and tok not in found:
                    found.append(tok)
                if len(found) >= 12:
                    break
            
            if not found:
                words = re.findall(r"[A-Za-z0-9\+\#\-]+(?:\s+[A-Za-z0-9\+\#\-]+){0,2}", clean_text)
                for w in words:
                    w_clean = w.strip().lower()
                    if len(w_clean) <= 2:
                        continue
                    if any(ch.isdigit() for ch in w_clean) or any(k in w_clean for k in ["api","sql","ml","c++","c#","aws","gcp","ros"]):
                        if w_clean not in found:
                            found.append(w_clean)
                    if len(found) >= 12:
                        break
            skills = found
            logger.info("Fallback skills detected: %s", skills[:10])


        if not domain:
            try:
                domain = self.rules._detect_domain_from_skills(skills)
            except Exception:
                domain = "general"

        # B4. Experience mapping 
        try:
            exp_info = self.exp_mapper.map_experience(clean_text)
        except Exception as e:
            logger.warning("ExperienceMapper failed: %s", e)
            exp_info = {"years": None, "level": None}

        # Title excerpt
        first_line = raw_jd.split("\n")[0] if raw_jd else ""
        title_excerpt = first_line[:80]

        # Build user profile
        user_profile = {
            "level": exp_info.get("level"),
            "years_exp": exp_info.get("years"),
            "skills": skills,
            "domain": domain,
            "title": title_excerpt,
            "summary": summary,
            "user_years": user_years
        }

        logger.info(">>> PROFILE: level=%s | domain=%s | skills=%s", user_profile['level'], domain, skills[:6])
        logger.info(">>> SUMMARY USED FOR EMBEDDING (preview): %s", (summary[:160] + "...") if len(summary) > 160 else summary)

        # B5. Embedding summary và assign cluster
        clean_summary = self.cleaner.clean(summary)
        jd_vector = self.embedder.get_embedding(clean_summary)
        jd_cluster_id = self.cluster_assigner.get_cluster(jd_vector.reshape(1, -1))
        logger.info(">>> CLUSTER: Assigned to Group #%s", jd_cluster_id)

        # B6. Retrieval
        candidates = self.retriever.get_candidates(query_text=clean_text, query_vector=jd_vector, top_k_each=50)

        # B7. Scoring + rules
        ranked_results = []
        for cand in candidates:
            c_id = cand.get("id")
            if c_id not in self.course_lookup:
                logger.warning("Candidate ID %s not found in course_lookup", c_id)
                continue

            info = self.course_lookup[c_id]
            full_cand = {**cand, **info}


            model_score, breakdown = self.scorer.calculate_score(full_cand, user_profile, jd_cluster_id)

            rule_multiplier = self.rules.apply_rules(user_profile, full_cand)
            if rule_multiplier != 1.0:
                logger.info("[RULE] Applied rule multiplier %s for course %s", rule_multiplier, c_id)

            breakdown["score_before_rules"] = model_score

            final_score = model_score * rule_multiplier

            breakdown["model_score"] = round(model_score, 6)
            breakdown["rule_multiplier"] = round(rule_multiplier, 6)
            breakdown["final_score"] = round(final_score, 6)

            explanation_json = self.explainer.generate_explanation(info.get('course_name'), breakdown)

            ranked_results.append({
                "id": c_id,
                "title": info.get('course_name'),
                "score": final_score,
                "level": info.get('level'),
                "url": info.get('page_url'),
                "breakdown": breakdown,
                "explanation": explanation_json
            })

        # Sort & top-k
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        final_recs = ranked_results[:top_k]

        # Trả về kết quả
        return {
            "time": f"{time.time() - t0:.2f}s",
            "summary": summary,
            "profile": user_profile,
            "recommendations": final_recs,
            "llm_used": llm_used
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", type=str, default="Data Scientist with Python and SQL skills")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    sysobj = CourseRecommenderSystem()
    res = sysobj.recommend(args.jd, top_k=args.top_k)

    print("=" * 60)
    print(f"JD SUMMARY: {res['summary']}")
    print("-" * 60)
    for i, r in enumerate(res['recommendations']):
        try:
            expl = json.loads(r['explanation'])
            reasons = ", ".join(expl.get('key_factors', []))
        except Exception:
            reasons = ""
        print(f"{i+1}. [{r.get('score', 0):.4f}] {r.get('title', 'N/A')} ({r.get('level','N/A')})")
        if reasons:
            print(f"   Lý do: {reasons}")
        print(f"   Link: {r.get('url')}\n")
