# src/jd_processing/jd_quick_extractor.py
"""
Module duy nhất để trích xuất thông tin từ JD:
- Input: raw_jd (string)
- Output: dict {"summary": str, "skills": [str], "domain": str, "llm_raw": str, "llm_used": bool}

Thiết kế:
- Hỗ trợ cả openai phiên bản cũ (<1.0.0) và mới (>=1.0.0).
- Ghi log raw LLM output và parsed JSON để debug.
- Nếu không có openai hoặc thiếu key, hàm sẽ raise RuntimeError (caller có thể fallback).
"""

from typing import List, Dict, Optional
import os
import re
import json
import logging
import textwrap

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Kiểm tra openai lib (import động)
try:
    import openai  # type: ignore
    OPENAI_PRESENT = True
except Exception:
    OPENAI_PRESENT = False

# Danh sách domain hợp lệ
DOMAIN_CHOICES = ["robotics", "data", "ba", "devops", "software", "cloud", "general"]

# Prompt hệ thống (yêu cầu trả JSON)
LLM_SYSTEM_PROMPT = (
    "You are a concise assistant specialized in parsing job descriptions. "
    "Given a job description, return a compact JSON object with fields: "
    "\"summary\" (2-4 sentences), "
    "\"skills\" (an array of the most relevant skills/technologies, short lowercase strings), "
    "\"domain\" (one word from: robotics,data,ba,devops,software,cloud,general). "
    "ONLY return valid JSON and nothing else."
)

LLM_USER_TEMPLATE = (
    "Job description:\n\n\"\"\"{jd}\"\"\"\n\n"
    "Produce the JSON as requested. Keep summary short (<=2 sentences). "
    "Skills: up to 12 items, short lowercase tokens. Domain must be one of allowed domains. "
    "Return strictly a JSON object (no extra text)."
)


def _clean_json_blob_from_text(text: str) -> Optional[str]:
    """
    Thử trích JSON blob từ output LLM (loại bỏ ``` nếu có).
    """
    if not text or not isinstance(text, str):
        return None
    t = re.sub(r"```(?:json)?\s*", "", text)
    t = t.replace("```", "").strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return t[start:end+1]


def _parse_and_normalize(parsed_obj: Dict) -> Dict:
    """
    Chuẩn hóa trường summary, skills, domain.
    """
    summary = parsed_obj.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary) if summary is not None else ""
    summary = summary.strip()

    skills = parsed_obj.get("skills", []) or []
    if isinstance(skills, str):
        skills = [s.strip() for s in re.split(r"[,\n;]+", skills) if s.strip()]
    if not isinstance(skills, list):
        try:
            skills = list(skills)
        except Exception:
            skills = []
    skills = [s.strip().lower() for s in skills if isinstance(s, str) and s.strip()]

    # domain = parsed_obj.get("domain", "") or ""
    # if not isinstance(domain, str) or domain.strip() not in DOMAIN_CHOICES:
    #     domain = "general"
    # else:
    #     domain = domain.strip()
    domain = parsed_obj.get("domain", "")
    if not isinstance(domain, str):
        domain = str(domain)
    domain = domain.strip() or "general"


    return {"summary": summary, "skills": skills, "domain": domain}


def _call_openai_llm(jd_text: str, api_key: Optional[str] = None, model: str = "gpt-4o", timeout: int = 20) -> Dict:
    """
    Gọi OpenAI. Hỗ trợ:
      - openai.ChatCompletion.create(...) (phiên bản cũ)
      - OpenAI(...).chat.completions.create(...) (hoặc client.chat.completions.create) (phiên bản mới)
    Nếu cả hai cách đều fail -> raise RuntimeError với chi tiết lỗi.
    """
    if not OPENAI_PRESENT:
        raise RuntimeError("openai library is not installed. Install with `pip install openai`.")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Set OPENAI_API_KEY or pass openai_api_key param.")

    raw_text = None
    last_err_old = None
    last_err_new = None

    # Thử giao diện cũ
    try:
        openai.api_key = key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": LLM_USER_TEMPLATE.format(jd=jd_text)}
            ],
            temperature=0.0,
            max_tokens=None,
            timeout=timeout,
        )
        try:
            raw_text = resp.choices[0].message.content.strip()
        except Exception:
            raw_text = str(resp)
    except Exception as ex_old:
        last_err_old = ex_old
        # Thử giao diện mới openai>=1.0.0
        try:
            # import class OpenAI từ package (nếu có)
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=key)
            # Giao diện mới: client.chat.completions.create(...)
            # (có các biến thể; ta thử .chat.completions.create)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": LLM_USER_TEMPLATE.format(jd=jd_text)}
                    ],
                    temperature=0.0,
                    max_tokens=None,
                )
            except Exception as ex_api_shape:
                # Nếu tên API khác (thay đổi giữa bản minor), thử client.chat.create(...) (fallback)
                try:
                    resp = client.chat.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": LLM_SYSTEM_PROMPT},
                            {"role": "user", "content": LLM_USER_TEMPLATE.format(jd=jd_text)}
                        ],
                        temperature=0.0,
                        max_tokens=512,
                    )
                except Exception as ex2:
                    raise ex2
            # cố lấy content
            try:
                raw_text = resp.choices[0].message.content.strip()
            except Exception:
                try:
                    raw_text = resp.choices[0]["message"]["content"].strip()
                except Exception:
                    raw_text = str(resp)
        except Exception as ex_new:
            last_err_new = ex_new

    if raw_text is None:
        raise RuntimeError(f"OpenAI API call failed. Old_err: {last_err_old}; New_err: {last_err_new}")

    # Log raw output (rút gọn nếu quá dài)
    logger.info("LLM RAW OUTPUT BEGIN")
    # display = raw_text if len(raw_text) <= 2000 else (raw_text[:2000] + " ...(truncated)")
    display = raw_text
    logger.info("\n" + display)
    logger.info("LLM RAW OUTPUT END")

    # Thử parse JSON
    json_blob = _clean_json_blob_from_text(raw_text)
    to_parse = json_blob if json_blob is not None else raw_text
    try:
        parsed = json.loads(to_parse)
    except Exception as e:
        logger.error("Failed to parse JSON from LLM output. Raw follows:")
        logger.error(raw_text)
        raise RuntimeError(f"Failed to parse JSON from LLM output: {e}")

    norm = _parse_and_normalize(parsed)
    return {
        "summary": norm["summary"],
        "skills": norm["skills"],
        "domain": norm["domain"],
        "llm_raw": raw_text,
        "llm_used": True
    }


def extract_jd_info(raw_jd: str, openai_api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> Dict:
    """
    Gọi LLM DUY NHẤT để trích xuất summary, skills, domain.
    - Nếu openai lib hoặc key không sẵn, hàm sẽ raise RuntimeError để caller xử lý fallback.
    - Trả về dict có keys: summary, skills, domain, llm_raw, llm_used.
    """
    raw_jd = (raw_jd or "").strip()
    if not raw_jd:
        return {"summary": "", "skills": [], "domain": "general", "llm_raw": "", "llm_used": False}

    parsed = _call_openai_llm(raw_jd, api_key=openai_api_key, model=model)

    logger.info("LLM PARSED OUTPUT:")
    try:
        logger.info(json.dumps({"summary": parsed["summary"], "skills": parsed["skills"], "domain": parsed["domain"]}, ensure_ascii=False, indent=2))
    except Exception:
        logger.info(str(parsed))

    return parsed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_jd = (
        "Bachelor's degree in Computer Science/IT or related field. Well-communicate in English.\n"
        "Minimum 2 years of professional experience as a Java Developer.\n"
        "Knowledge of Spring / Spring Boot essential. Database technologies: Postgres and Redis desirable.\n"
        "Experience building ReSTful APIs essential – SOAP not required. Experience in Database: Oracle, MS SQL Server, PostgreSQL."
    )
    try:
        out = extract_jd_info(example_jd)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    except Exception as e:
        print("Error while extracting JD info:", e)
