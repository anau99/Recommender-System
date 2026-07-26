

import streamlit as st
import sys, os, time, json
import pandas as pd
import numpy as np

# -------------------------
# Project paths (update if needed)
# -------------------------
PROJECT_ROOT = "/Users/nhitruong/Documents/data_mining_project"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# -------------------------
# Try import backend class from your backend file
# -------------------------
try:
    from main import CourseRecommenderSystem
    BACKEND_OK = True
except Exception as e:
    CourseRecommenderSystem = None
    BACKEND_OK = False
    IMPORT_ERROR = e

# -------------------------
# Helpers: serialization safe guards
# -------------------------
def ensure_serializable(x):
    """Convert numpy types -> Python native, and recursively handle lists/dicts."""
    if isinstance(x, (np.generic,)):
        try:
            return x.item()
        except Exception:
            return float(x) if hasattr(x, "astype") else str(x)
    if isinstance(x, np.ndarray):
        try:
            return x.tolist()
        except Exception:
            try:
                return x.ravel().tolist()
            except Exception:
                return str(x)
    if isinstance(x, (list, tuple)):
        return [ensure_serializable(v) for v in x]
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            try:
                key = str(k)
            except Exception:
                key = json.dumps(k, default=str)
            out[key] = ensure_serializable(v)
        return out
    try:
        json.dumps(x)
        return x
    except Exception:
        try:
            return str(x)
        except Exception:
            return repr(x)

# -------------------------
# Small helper: smart title-casing keep common acronyms / libs pretty
# -------------------------
def smart_title(s: str) -> str:
    if not s:
        return s
    canonical = {
        "sql": "SQL",
        "nlp": "NLP",
        "ai": "AI",
        "cv": "CV",
        "gpu": "GPU",
        "cpu": "CPU",
        "m1": "M1",
        "m2": "M2",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "tf": "TensorFlow",
        "ros": "ROS",
        "ros2": "ROS2",
        "c++": "C++",
        "c#": "C#",
        ".net": ".NET",
        "ml": "ML",
    }
    parts = []
    for tok in str(s).split():
        key = tok.lower().strip(" ,.-()")
        if key in canonical:
            parts.append(canonical[key])
        else:
            if tok.isupper():
                parts.append(tok)
            else:
                parts.append(tok.capitalize())
    return " ".join(parts)

# Capitalize first character helper (keeps rest intact)
def cap_first(s: str) -> str:
    if not s:
        return s
    return s[0].upper() + s[1:]

# -------------------------
# Page config + CSS (use the original CSS you provided + small tweaks for skill chips)
# -------------------------
st.set_page_config(page_title="Course Recommendation", layout="wide")
st.markdown(
    """
<style>
.block-container {
  max-width: 1100px;
  margin-left: auto;
  margin-right: auto;
  padding-top: 32px;
  padding-bottom: 48px;
  display: flex;
  flex-direction: column;
  min-height: calc(100vh - 120px);
}
.block-container h1, .block-container h2, .block-container h3 {
  text-align: center;
}
.big-textarea .stTextArea>div>div>textarea { 
  min-height: 380px !important; 
  font-size: 15px; 
  padding: 16px;
  border-radius: 10px;
  box-shadow: 0 1px 3px rgba(12, 18, 36, 0.04);
}
.button-row {
  margin-top: auto;
  margin-bottom: 8px;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  padding-top: 18px;
}
.stButton {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
}
.stButton>button {
  padding: 14px 44px !important;
  font-size: 18px !important;
  font-weight: 800 !important;
  border-radius: 12px !important;
  background: linear-gradient(180deg, #0ea5e9, #0284c7) !important;
  color: white !important;
  border: none !important;
  box-shadow: 0 6px 18px rgba(2,6,23,0.25) !important;
}
.json-box pre { font-size: 13px; }
.course-card { 
  background: white;
  border-radius: 12px;
  box-shadow: 0 6px 20px rgba(2,6,23,0.04);
  padding: 12px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.course-card.top-card {
  min-height: 220px; /* ensure top 3 equal height */
}
.skill-chip {
  display:inline-block;
  background:#eef2ff;
  color:#3730A3;
  padding:6px 10px;
  border-radius:999px;
  font-weight:700;
  margin:4px 6px 4px 0;
  font-size:13px;
}
.level-badge {
  display:inline-block;
  padding:6px 12px;
  border-radius:999px;
  font-weight:700;
}
/* Extra selectors for download button variants */
.stDownloadButton > button,
.stDownloadButton>button,
.stDownloadButton > div > button,
div.stDownloadButton > button {
    padding: 14px 44px !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    border-radius: 12px !important;
    background: linear-gradient(180deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 6px 18px rgba(2,6,23,0.25) !important;
}

/* center download container */
.download-container {
    display:flex;
    justify-content:center;
    width:100%;
    margin-top:14px;
    margin-bottom:8px;
}

/* small responsive tweaks for cards */
@media (max-width: 900px) {
  .course-card { padding: 10px; }
  .stButton>button, .stDownloadButton>button { padding: 10px 30px !important; font-size:16px !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar controls
# -------------------------
# with st.sidebar:
#     st.markdown("### ⚙️ Cấu hình hệ thống")
#     st.write("")
#     # Years of experience input provided by the user and passed to recommend() as user_year
#     yoe = st.slider("Kinh nghiệm (năm)", 0, 30, 2)
#     # show_raw default is True as you requested
#     show_raw = st.checkbox("Hiện JSON kết quả (raw)", value=True)
#     st.markdown("---")
with st.sidebar:
    st.write("Nhóm 8")
    st.markdown("---")

# Top K fixed
TOP_K = 10

# -------------------------
# Main UI
# -------------------------
st.markdown("## 🎓 Gợi ý khoá học")
jd_text = st.text_area(
    "📋 Dán JD vào đây:",
    height=380,
    key="jd",
    placeholder="Dán toàn bộ mô tả công việc ở đây...",
    label_visibility="visible",
)
# st.markdown("### ⚙️ Cấu hình thêm")
yoe = st.slider("Kinh nghiệm (năm)", 0, 30, 2)

show_raw = True   # luôn bật nhưng không hiển thị checkbox



# Center Recommend button
st.markdown('<div style="display:flex; justify-content:center; width:100%; margin-top:18px;">', unsafe_allow_html=True)
col_center = st.columns([3, 2, 3])[1]
with col_center:
    recommend_btn = st.button("🚀 Recommend")
st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# -------------------------
# render_inferred_profile (improved: capitalized summary + skill chips with color)
# -------------------------
def render_inferred_profile(profile: dict):
    import html as _html

    skills = profile.get("skills") or profile.get("known_skills") or []
    domain = (profile.get("domain") or "N/A").upper()


    st.markdown('<h4 style="text-align:left;">👤 Inferred profile</h4>', unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 3])

    with col_a:
        st.markdown(f"**Domain:**  `{domain}`")

    with col_b:
        st.markdown("**Top skills:**")
        if isinstance(skills, (list, tuple)) and len(skills) > 0:
            short_skills = [str(s).strip() for s in skills[:12] if s is not None and str(s).strip() != ""]
            safe_skills = [_html.escape(s) for s in short_skills]
            chip_html = ['<div style="margin-top:6px;">']
            for i, s in enumerate(safe_skills):
                chip_html.append(f'<span class="skill-chip">{s}</span>')
            chip_html.append("</div>")
            st.markdown("".join(chip_html), unsafe_allow_html=True)
        else:
            st.write(skills)

# -------------------------
# Fallback demo data (kept original)
# -------------------------
DEMO_COURSES = [
    {"id": "c1", "course_name": "Python for Data Analysis", "skills": ["python", "pandas", "numpy"], "level": "Beginner", "page_url": "#"},
    {"id": "c2", "course_name": "Machine Learning Foundations", "skills": ["machine learning", "sklearn"], "level": "Intermediate", "page_url": "#"},
    {"id": "c3", "course_name": "Deep Learning with PyTorch", "skills": ["deep learning", "pytorch"], "level": "Advanced", "page_url": "#"},
    {"id": "c4", "course_name": "SQL for Analysts", "skills": ["sql", "database"], "level": "Beginner", "page_url": "#"},
    {"id": "c5", "course_name": "MLOps Practical", "skills": ["mlops", "docker"], "level": "Advanced", "page_url": "#"},
    {"id": "c6", "course_name": "NLP with Transformers", "skills": ["nlp", "transformers"], "level": "Advanced", "page_url": "#"},
]

def fallback_recommend(jd, years, top_k):
    words = [w.strip().lower().strip(".,()") for w in jd.split() if len(w) > 2]
    keywords = list(dict.fromkeys(words))[:8]
    rows = []
    for c in DEMO_COURSES:
        overlap = sum(1 for k in keywords if any(k in s for s in c["skills"]))
        score = 0.7 * (overlap / max(1, len(c["skills"]))) + 0.3 * (min(1, years / 3))
        rows.append({"data": c, "final_score": score, "breakdown": {"overlap": overlap}})
    rows = sorted(rows, key=lambda x: x["final_score"], reverse=True)
    return {"summary": f"Keywords: {', '.join(keywords)}", "profile": {"skills": keywords, "domain": "N/A"}, "recommendations": rows[:top_k]}

# -------------------------
# Run recommendation
# -------------------------
if recommend_btn:
    if not jd_text or not jd_text.strip():
        st.warning("Vui lòng nhập nội dung JD trước khi Recommend.")
    else:
        with st.spinner("⏳ đang xử lý..."):
            time.sleep(0.15)
            if BACKEND_OK:
                try:
                    recsys = CourseRecommenderSystem(base_dir=PROJECT_ROOT)
                    try:
                        # primary: pass user_years (matches main.recommend signature)
                        result = recsys.recommend(jd_text, user_years=yoe, top_k=TOP_K)
                    except TypeError:
                        # older backends might expect `years` or `user_year` instead
                        try:
                            result = recsys.recommend(jd_text, years=yoe, top_k=TOP_K)
                        except TypeError:
                            try:
                                result = recsys.recommend(jd_text, user_year=yoe, top_k=TOP_K)
                            except TypeError:
                                # final fallback: call without year parameter
                                result = recsys.recommend(jd_text, top_k=TOP_K)
                except Exception as e:
                    # keep messages simple and safe (avoid multi-line f-string issue)
                    st.error(f"⚠️ Backend lỗi khi recommend(): {e}\nSử dụng fallback demo.")
                    result = fallback_recommend(jd_text, yoe, TOP_K)
            else:
                result = fallback_recommend(jd_text, yoe, TOP_K)

        # show summary (left aligned) — capitalize first letter
        st.markdown("<h3 style='text-align:left;'>🔎 JD Summary</h3>", unsafe_allow_html=True)
        raw_summary = result.get("summary", "(no summary)")
        try:
            display_summary = cap_first(str(raw_summary))
        except Exception:
            display_summary = raw_summary
        st.write(display_summary)

        # render profile (skills shown as colored chips)
        profile = result.get("profile", {"skills": []})
        render_inferred_profile(profile)

        # recommendations grid (re-introduce level badge)
        st.markdown("---")
        st.markdown(f"### Top {TOP_K} Recommendations")
        recs = result.get("recommendations", []) or []

        # Normalize candidate items: backend may return objects in different shapes
        normalized = []
        for r in recs:
            # If fallback returns dict with 'data' key
            if isinstance(r, dict) and "data" in r:
                item = r["data"]
                score = r.get("final_score") or r.get("score") or r.get("score", 0.0)
                breakdown = r.get("breakdown", {})
            else:
                item = r
                score = r.get("final_score") or r.get("score") or 0.0
                breakdown = r.get("breakdown", {})
            normalized.append({"item": item, "score": float(score), "breakdown": breakdown})

        # Ensure sorted by score desc and show exactly TOP_K (if backend returned fewer, show what's available)
        normalized = sorted(normalized, key=lambda x: x["score"], reverse=True)[:TOP_K]

        rows_export = []
        if not normalized:
            st.info("Không có recommendation trả về.")
        else:
            per_row = 3
            rec_idx = 0
            for i in range(0, len(normalized), per_row):
                cols = st.columns(per_row)
                slice_ = normalized[i:i+per_row]
                for idx, rec in enumerate(slice_):
                    col = cols[idx]
                    info = rec["item"]
                    score = rec["score"]
                    breakdown = rec.get("breakdown", {})

                    # Normalize title & url
                    if isinstance(info, dict) and "course_name" in info:
                        raw_title = info.get("course_name", "")
                    else:
                        raw_title = info.get("title", "") if isinstance(info, dict) else str(info)
                    raw_title = str(raw_title).strip()
                    title = smart_title(raw_title)

                    # Determine level for badge
                    lvl = "N/A"
                    if isinstance(info, dict):
                        lvl = info.get("level") or info.get("difficulty") or info.get("seniority") or lvl
                    lvl = str(lvl).strip()

                    # choose badge colors by level
                    lvl_norm = lvl.lower()
                    if "begin" in lvl_norm or "junior" in lvl_norm:
                        lvl_bg, lvl_fg = "#ecfdf5", "#059669"   # green-ish
                    elif "inter" in lvl_norm or "mid" in lvl_norm:
                        lvl_bg, lvl_fg = "#fffbeb", "#92400e"   # yellow-ish
                    elif "adv" in lvl_norm or "senior" in lvl_norm:
                        lvl_bg, lvl_fg = "#eef2ff", "#3730a3"   # purple-ish
                    else:
                        lvl_bg, lvl_fg = "#f3f4f6", "#374151"   # gray

                    lvl_badge_html = f'<span class="level-badge" style="background:{lvl_bg};color:{lvl_fg};">{lvl}</span>'

                    url = ""
                    if isinstance(info, dict):
                        url = info.get("page_url") or info.get("url") or "#"
                    else:
                        url = "#"

                    # Build card HTML: ensure title appears before level badge and top 3 have equal min-height
                    top_class = " top-card" if rec_idx < 3 else ""
                    card_md = f"""
<div class="course-card" style="min-height:200px; display:flex; flex-direction:column; justify-content:space-between;">

  <!-- Title -->
  <div style="font-weight:800; font-size:16px; color:#0f1724; line-height:1.3; margin-bottom:8px;">
    {title}
  </div>

  <!-- Level (below title) -->
  <div style="margin-bottom:16px; text-align:left;">
    {lvl_badge_html}
  </div>

  <!-- Link + Score -->
  <div style="margin-top:auto; display:flex; justify-content:space-between; align-items:center;">
    <a href="{url}" target="_blank">🔗 Xem khoá học</a>
    <span style="font-size:13px; color:#4b5563;"></span>
  </div>

</div>
"""
                    with col:
                        st.markdown(card_md, unsafe_allow_html=True)
                        with st.expander("Chi tiết & breakdown"):
                            # show breakdown (dict) instead of raw string
                            try:
                                st.json(breakdown)
                            except Exception:
                                st.write(breakdown)

                    # append sanitized export row
                    safe_title = ensure_serializable(title)
                    try: safe_title = str(safe_title)
                    except Exception: safe_title = json.dumps(safe_title, ensure_ascii=False)
                    safe_url = ensure_serializable(url)
                    try: safe_url = str(safe_url)
                    except Exception: safe_url = json.dumps(safe_url, ensure_ascii=False)
                    safe_level = ensure_serializable(lvl)
                    try: safe_level = str(safe_level)
                    except Exception: safe_level = json.dumps(safe_level, ensure_ascii=False)

                    rows_export.append({"title": safe_title, "url": safe_url, "level": safe_level, "score": score})

                    rec_idx += 1

        # CSV export (sanitize then show centered download button)
        if rows_export:
            cleaned_rows = []
            problematic = []
            for ridx, rec in enumerate(rows_export):
                try:
                    if not isinstance(rec, dict):
                        rec = {"value": rec}
                    new_rec = {}
                    for k, v in rec.items():
                        try:
                            key = str(k)
                        except Exception:
                            key = json.dumps(k, default=str)
                        new_rec[key] = ensure_serializable(v)
                    cleaned_rows.append(new_rec)
                except Exception as e:
                    problematic.append({"index": ridx, "raw": str(rec)[:300], "error": str(e)})
            if problematic:
                st.warning(f"Có {len(problematic)} record không thể sanitize hoàn toàn — hiển thị 3 cái đầu.")
                for p in problematic[:3]:
                    st.write(p)

            try:
                df_export = pd.DataFrame(cleaned_rows)
            except Exception:
                df_export = pd.json_normalize(cleaned_rows)

            for c in df_export.columns:
                if df_export[c].dtype == object:
                    df_export[c] = df_export[c].apply(lambda v: ensure_serializable(v))

            csv = df_export.to_csv(index=False)
            csv_bytes = csv.encode("utf-8-sig")
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recommendations_{ts}.csv"

            left, mid, right = st.columns([3, 2, 3])
            with mid:
                st.download_button(
                    label="⬇️ Tải CSV kết quả",
                    data=csv_bytes,
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{ts}",
                )

        # show raw json (breakdown) at the end, after CSV button as requested
        if show_raw:
            st.markdown("---")
            st.markdown("### 🔧 Raw result (breakdown)")
            try:
                st.json(result)
            except Exception:
                st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")

# Footer / debug
st.markdown("---")
if not BACKEND_OK:
    with st.expander("⚠️ Backend không sẵn sàng — thông tin lỗi (nhấn để mở)"):
        st.write("Import error while importing `main.CourseRecommenderSystem`.")
        st.write("Error detail:")
        st.write(str(IMPORT_ERROR))
        st.write("")
        st.write("Nếu bạn muốn dùng backend thật, đảm bảo file `src/main.py` có class `CourseRecommenderSystem` với method `recommend(raw_jd, top_k=...)`.")
