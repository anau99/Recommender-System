import os
import sys
import argparse
import json
import logging
import random
from typing import List, Dict, Any

import pandas as pd
from datetime import datetime

# Add src to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import backend
try:
    from main import CourseRecommenderSystem
    BACKEND_AVAILABLE = True
except Exception as e:
    CourseRecommenderSystem = None
    BACKEND_AVAILABLE = False
    IMPORT_ERROR = e

# Logging
logger = logging.getLogger("batch_recommender")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def find_columns(df: pd.DataFrame) -> Dict[str, str]:
    candidates = {c.lower(): c for c in df.columns}
    mapping = {"title": None, "full_content_clean": None, "company": None}

    for possible in ["title", "job_title", "position"]:
        if possible in candidates:
            mapping["title"] = candidates[possible]
            break
    if mapping["title"] is None:
        mapping["title"] = df.columns[0]

    for possible in ["full_content_clean", "description", "job_description", "content", "jd"]:
        if possible in candidates:
            mapping["full_content_clean"] = candidates[possible]
            break
    if mapping["full_content_clean"] is None:
        mapping["full_content_clean"] = df.columns[1] if len(df.columns) > 1 else mapping["title"]

    for possible in ["company", "employer", "organization", "company_name"]:
        if possible in candidates:
            mapping["company"] = candidates[possible]
            break

    return mapping


def sanitize_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        s = str(x)
    except Exception:
        s = json.dumps(x)
    return s.strip()


def top_titles_from_recs(recs: List[Dict], top_k: int) -> List[str]:
    out = []
    if not recs:
        return out
    for r in recs[:top_k]:
        title = ""
        if isinstance(r, dict):
            title = r.get("title") or r.get("course_name") or ""
        out.append(sanitize_text(title))
    return out


def run_batch(csv_path: str, n: int, top_k: int, out_path: str, user_years_arg: int, backend_base_dir: str):
    logger.info("CSV: %s", csv_path)
    df = pd.read_csv(csv_path).fillna("")
    cols = find_columns(df)
    logger.info("Mapping cột: %s", cols)

    n = min(n, len(df))

    # Khởi backend
    recsys = None
    if BACKEND_AVAILABLE:
        try:
            recsys = CourseRecommenderSystem(base_dir=backend_base_dir or PROJECT_ROOT)
            logger.info("Backend sẵn sàng.")
        except Exception as e:
            logger.error("Không khởi backend được: %s", e)
    else:
        logger.warning("Không thể import backend: %s", IMPORT_ERROR)

    sample_data = []

    for i in range(n):
        row = df.iloc[i]
        title = sanitize_text(row[cols["title"]])
        content = sanitize_text(row[cols["full_content_clean"]])
        company = sanitize_text(row[cols["company"]]) if cols["company"] else ""

        query_text = title + "\n" + content
        query_id = i + 1

        if user_years_arg is None:
            user_years = random.randint(0, 10)
        else:
            user_years = user_years_arg

        logger.info("JD %d: user_years=%d", query_id, user_years)

        rec_entry = {
            "query_id": query_id,
            "query": query_text,
            "company": company,
            "user_years": user_years
        }

        try:
            if recsys:
                res = recsys.recommend(query_text, user_years=user_years, top_k=top_k)
            else:
                res = None
        except Exception as e:
            logger.error("Lỗi recommend JD %d: %s", query_id, e)
            res = None

        if res and isinstance(res, dict):
            recs = res.get("recommendations", [])
            titles = top_titles_from_recs(recs, top_k)
            for t_i, t in enumerate(titles):
                rec_entry[f"result{t_i+1}"] = t
            rec_entry["_meta"] = {
                "summary": res.get("summary", ""),
                "profile": res.get("profile", {}),
                "time": res.get("time", "")
            }
        else:
            for k in range(top_k):
                rec_entry[f"result{k+1}"] = ""
            rec_entry["_meta"] = {"error": "no_result"}

        sample_data.append(rec_entry)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    logger.info("Đã lưu %d records vào %s", len(sample_data), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--n", default=10, type=int)
    parser.add_argument("--top-k", default=2, type=int)
    parser.add_argument("--out", default="outputs/batch_results.json")
    parser.add_argument("--user-years", type=int, default=None, help="Nếu không truyền → random")
    parser.add_argument("--base-dir", type=str, default=None)

    args = parser.parse_args()

    run_batch(
        csv_path=args.csv,
        n=args.n,
        top_k=args.top_k,
        out_path=args.out,
        user_years_arg=args.user_years,
        backend_base_dir=args.base_dir
    )
