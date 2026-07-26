# Course Recommender System

An AI-powered **course recommendation engine** that reads a raw **Job Description (JD)** and recommends the online courses (Coursera-style catalog) that best help a candidate close the gap between their profile and that job.

The system combines:
- **LLM-based JD understanding** (summary, skills, domain extraction via OpenAI)
- **Semantic search** (Sentence-Transformers embeddings + FAISS)
- **Keyword search** (BM25)
- **Unsupervised clustering** (PCA + KMeans) to group JDs/courses by topic
- **A weighted scoring model** with **business rules** (experience level, domain fit, certificate preference, etc.)
- **Automatic, human-readable explanations** for every recommendation

It ships with a CLI, a batch runner, a Streamlit web app, a grid-search weight optimizer, and several evaluation scripts (Precision@K, NDCG, LLM-as-a-judge).

---

## Table of Contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Data](#data)
- [Setup](#setup)
- [Building the pipeline from scratch](#building-the-pipeline-from-scratch)
- [Usage](#usage)
- [Weight optimization](#weight-optimization)
- [Evaluation](#evaluation)
- [Notes](#notes)

---

## How it works

```
Raw JD (text)
    │
    ▼
1. JDCleaner            → lowercased, cleaned text (for BM25 / logging)
    │
    ▼
2. jd_quick_extractor    → LLM (gpt-3.5-turbo) extracts {summary, skills, domain}
    (falls back to lightweight heuristics if the LLM/API key is unavailable)
    │
    ▼
3. ExperienceMapper      → detects years of experience & seniority level
    │
    ▼
4. EmbeddingModel         → embeds the JD summary (Sentence-Transformers, 768-dim)
    │
    ▼
5. ClusterAssigner        → PCA + KMeans assigns the JD to a topic cluster
    │
    ▼
6. CandidateUnion         → merges candidates from:
       - SemanticSearcher (FAISS, cosine/inner-product search)
       - BM25Searcher (keyword search)
    │
    ▼
7. Scorer                 → weighted score: semantic + bm25 + cluster boost
                              + level boost + skill bonus
    │
    ▼
8. BusinessRules           → multiplies the score based on:
       - missing core skills penalty
       - JD level vs. course level mismatch penalty
       - certificate preference boost
       - years-of-experience gap multiplier
    │
    ▼
9. Explainability          → generates a natural-language JSON explanation
    │
    ▼
Top-K ranked course recommendations
```

The end-to-end orchestration lives in **`src/main.py`** (`CourseRecommenderSystem.recommend()`).

---

## Project structure

```
Recommender-System/
├── clustering/                  # Standalone scripts for clustering research
│   ├── elbow_method.py          # Elbow / Silhouette analysis (GPU via RAPIDS cuML, CPU fallback)
│   ├── kmeans_trainer.py        # Trains and saves the final KMeans model
│   ├── pca_reducer.py           # Fits PCA for dimensionality reduction
│   ├── cluster_assigner.py      # Loads PCA + KMeans to assign a cluster to a new vector
│   └── visualize_comparison.py  # T-SNE plot: manual categories vs. AI clusters
│
├── configs/
│   ├── settings.py               # Loads .env, exposes BASE_DIR / DATA_DIR / OUTPUT_DIR / API keys
│   └── grid_search.yaml          # Weight ranges + evaluation config for grid search
│
├── data/
│   ├── jds.csv                   # ~1.8k job descriptions (title, full_content_clean, company)
│   ├── jds_knowledge_base.json   # Sample of manually/GPT-labeled JD extractions
│   └── test_set_ground_truth.json# Ground-truth course IDs per JD, used for Precision@K / NDCG
│
├── outputs/                      # Generated artifacts (embeddings, indices, trained models, logs)
│   ├── embeddings/                # course_emb.npy, course_emb_pca.npy, course_meta.csv (~10k courses)
│   ├── indices/                   # faiss_index.bin
│   ├── models/                    # bm25_model.pkl, pca_model.pkl, kmeans_model.pkl, best_weights.json ...
│   ├── clusters/                  # Elbow/silhouette results for different pipelines
│   └── images/                    # Generated charts
│
├── src/
│   ├── main.py                    # CourseRecommenderSystem — the current, active pipeline
│   ├── main_old.py                # Previous iteration, kept for reference (superseded by main.py)
│   ├── data_loader.py             # Simple CSV loader for courses/JDs
│   ├── streamlit_app.py           # Web UI for the recommender
│   ├── run_batch_recommend.py     # Batch-runs recommend() over a CSV of JDs → JSON results
│   ├── run.sh                     # Launches the Streamlit app (+ optional ngrok tunnel)
│   │
│   ├── jd_processing/
│   │   ├── jd_cleaner.py           # Text cleaning (lowercase, strip URLs/emails, keep Vietnamese chars)
│   │   ├── jd_quick_extractor.py   # LLM-based {summary, skills, domain} extractor (active)
│   │   ├── experience_mapper.py    # Regex-based years-of-experience & level detector (active)
│   │   ├── skill_extractor.py      # Legacy skill extractors (regex / LLM) — commented out, kept for reference
│   │   ├── jd_summarizer.py        # Legacy summarizers (BART / GPT-3.5) — commented out, kept for reference
│   │   ├── process_jd_pipeline.py  # Legacy standalone batch-processing pipeline — commented out
│   │   └── run_jd_processing.py    # Legacy demo entry point — commented out
│   │
│   ├── embedding/
│   │   ├── embedding_model.py       # SentenceTransformer wrapper (auto GPU/MPS/CPU detection)
│   │   ├── build_course_embeddings.py # Builds course_emb.npy + course_meta.csv from the raw course CSV
│   │   └── encode_jds_csv.py        # Embeds + PCA + cluster-assigns a JD CSV (standalone utility)
│   │
│   ├── retrieval/
│   │   ├── semantic_search.py       # FAISS vector search wrapper
│   │   ├── bm25_retriever.py        # BM25 keyword search (train/save/load/search)
│   │   ├── build_faiss_index.py     # Builds the FAISS index from course embeddings
│   │   └── candidate_union.py       # Merges semantic + BM25 candidates into one pool
│   │
│   ├── recommender/
│   │   ├── scoring.py                # Weighted scoring model (semantic/bm25/cluster/level/skill)
│   │   ├── rules.py                  # Business rules (LLM-first skill matching, level & experience penalties/boosts)
│   │   ├── explainability.py         # Turns a score breakdown into a natural-language explanation
│   │   └── weight_search.py          # Grid search over configs/grid_search.yaml → outputs/models/best_weights.json
│   │
│   └── evaluation/
│       ├── eval_metrics.py           # Precision@K
│       ├── eval_precision_ndcg.py    # DCG / NDCG@K
│       ├── compare_faiss_vs_cluster.py # Compares FAISS vs. cluster-based retrieval speed/precision
│       ├── create_dummy_testset.py   # Generates a synthetic ground-truth test set
│       ├── evaluate_with_gemini.py   # LLM-as-a-judge evaluation using Google Gemini
│       └── evaluation_hermes.py      # LLM-as-a-judge evaluation using Hermes-2-Pro (HF Inference API)
│
├── run_batch.sh                   # Shortcut: runs run_batch_recommend.py over data/jds.csv
```

---

## Data

This repository ships with:
- `data/jds.csv` — sample job descriptions used as recommendation queries and for evaluation.
- `data/jds_knowledge_base.json` / `data/test_set_ground_truth.json` — small labeled sets used by the evaluation scripts.
- `outputs/embeddings/course_meta.csv` — metadata for ~10,000 pre-processed courses (title, topic, level, skills, etc.).

Large binary artifacts (raw course dataset, precomputed embeddings, FAISS index, trained PCA/KMeans/BM25 models) are **not fully included in version control**. Download the full dataset / precomputed artifacts here:

**📦 Data download:** https://1drv.ms/f/c/7e84798955f2cae0/IgBJa1p5ehDCQK-SmMWlUaDnAXSdnOFKM723rKgWCJ38JbI?e=CkPc3o

After downloading, place the files under the matching folders (`data/`, `outputs/embeddings/`, `outputs/indices/`, `outputs/models/`) so the paths referenced in `src/main.py` resolve correctly.

---

## Setup

**Requirements:** Python 3.9+ (a CUDA GPU or Apple Silicon is optional but speeds up embedding/clustering).

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

2. Install the core dependencies:
   ```bash
   pip install pandas numpy scikit-learn sentence-transformers faiss-cpu \
               rank-bm25 python-dotenv streamlit tqdm matplotlib seaborn \
               pyyaml openai torch
   ```
   Optional extras used by specific scripts:
   ```bash
   pip install google-generativeai   # src/evaluation/evaluate_with_gemini.py
   pip install huggingface_hub       # src/evaluation/evaluation_hermes.py
   pip install cupy-cudaXX cuml      # GPU-accelerated clustering.elbow_method (RAPIDS)
   ```

3. Create a `.env` file in the project root with the API keys you plan to use:
   ```env
   OPENAI_API_KEY=sk-...     # required for JD extraction (jd_quick_extractor)
   GEMINI_API_KEY=...        # optional, for evaluate_with_gemini.py
   HF_TOKEN=hf_...           # optional, for evaluation_hermes.py
   ```
   If `OPENAI_API_KEY` is missing, `main.py` still works — it silently falls back to lightweight heuristic summarization/skill extraction.

---

## Building the pipeline from scratch

If you're starting from a raw course catalog CSV instead of the provided `outputs/` artifacts, rebuild them in this order:

```bash
# 1. Embed all courses (SBERT, 768-dim)
python src/embedding/build_course_embeddings.py \
    --courses data/courses.csv \
    --out_dir outputs/embeddings

# 2. (Optional) Reduce dimensionality with PCA, used for clustering
python clustering/pca_reducer.py \
    --input outputs/embeddings/course_emb.npy \
    --output outputs/embeddings/course_emb_pca.npy \
    --model outputs/models/pca_model.pkl

# 3. Find a good K (elbow / silhouette analysis)
python clustering/elbow_method.py --emb outputs/embeddings/course_emb_pca.npy

# 4. Train the final KMeans model with your chosen K
python clustering/kmeans_trainer.py --k 11

# 5. Build the FAISS index (uses the original 768-dim embeddings for best accuracy)
python src/retrieval/build_faiss_index.py \
    --emb outputs/embeddings/course_emb.npy \
    --index outputs/indices/faiss_index.bin

# 6. Train the BM25 keyword index
python src/retrieval/bm25_retriever.py \
    --meta outputs/embeddings/course_meta.csv \
    --save_path outputs/models/bm25_model.pkl
```

---

## Usage

### Command line (single JD)

```bash
python src/main.py --jd "Data Scientist with Python and SQL skills" --top_k 10
```

### Batch mode (a CSV of JDs → JSON results)

```bash
bash run_batch.sh
# or directly:
python src/run_batch_recommend.py \
    --csv data/jds.csv \
    --n 10 \
    --top-k 20 \
    --out outputs/batch_results.json
```

### Web app (Streamlit)

```bash
streamlit run src/streamlit_app.py
# or, with an ngrok public tunnel:
bash src/run.sh
```

---

## Weight optimization

`src/recommender/scoring.py` combines several signals (semantic similarity, BM25, cluster match, level match, skill overlap) using a weighted formula. `src/recommender/weight_search.py` performs a grid search over the ranges defined in `configs/grid_search.yaml`, evaluates each combination against `data/test_set_ground_truth.json` using Precision@K, and writes the best combination to `outputs/models/best_weights.json` (automatically picked up by `main.py` at startup).

```bash
python -m src.recommender.weight_search
```

---

## Evaluation

- **Precision@K / NDCG@K** — `src/evaluation/eval_metrics.py`, `src/evaluation/eval_precision_ndcg.py`
- **Retrieval comparison** (FAISS vs. cluster-based candidate generation) — `src/evaluation/compare_faiss_vs_cluster.py`
- **Synthetic ground truth generation** — `src/evaluation/create_dummy_testset.py`
- **LLM-as-a-judge** — `src/evaluation/evaluate_with_gemini.py` (Google Gemini) and `src/evaluation/evaluation_hermes.py` (Hermes-2-Pro via Hugging Face Inference API)

---

## Notes

- `src/main_old.py` is a previous version of the pipeline, kept for reference/history; **`src/main.py` is the actively used entry point**.
- Several legacy modules in `src/jd_processing/` (`skill_extractor.py`, `jd_summarizer.py`, `process_jd_pipeline.py`, `run_jd_processing.py`) are fully commented out — they document earlier approaches (regex-based extraction, local BART summarization) that were replaced by the single LLM call in `jd_quick_extractor.py`.
- Some scripts (`clustering/*.py`, `src/evaluation/create_dummy_testset.py`, `src/evaluation/evaluate_with_gemini.py`) contain a hard-coded `BASE_DIR`/`PROJECT_ROOT` from the original author's machine — update these paths (or pass the equivalent CLI arguments) before running them in a different environment.
- All source comments have been translated to English for readability; code logic is unchanged.