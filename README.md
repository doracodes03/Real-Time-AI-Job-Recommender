## Evaluation & Comparison

To quantitatively and qualitatively compare the old (TF-IDF+experience) and new (hybrid) models:

1. Ensure your job data is in `data/jobs.csv` with columns: `Job Description`, `Role`, `Experience`.
2. Run the evaluation script:
   ```python
   # In a Python shell or notebook:
   import pandas as pd
   from src.evaluate import run_evaluation
   jobs_df = pd.read_csv('data/jobs.csv')
   run_evaluation(jobs_df)
   ```

This will print average Precision@5 for both models, % improvement, and sample qualitative comparisons (especially for resumes with unseen skills).
# Hybrid Job Recommendation Engine

## Improvements (2026)

This system now supports a hybrid recommendation engine with:
- **Semantic Embeddings:** Uses Sentence-BERT for deep semantic matching (install `sentence-transformers`)
- **Skill Overlap:** Combines exact and semantic skill overlap (handles synonyms/unseen skills)
- **Hybrid Scoring:**
  - 0.5 * semantic similarity (SBERT)
  - 0.2 * TF-IDF similarity
  - 0.2 * skill overlap (dictionary + semantic)
  - 0.1 * experience score
- **Dynamic Skill Expansion:** Optionally expands skill dictionary using semantic similarity

## Setup


1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   pip install sentence-transformers
   ```

2. **Split your job dataset (if not already done):**
   ```powershell
   python split_job_descriptions.py
   ```
   This creates `data/jobs_train.csv` (80%) and `data/jobs_test.csv` (20%).

3. **Run training using jobs_train.csv:**
   ```powershell
   python train.py --input data/jobs_train.csv
   ```
   *(If your train.py does not support --input, update it to use data/jobs_train.csv by default.)*

4. **Run evaluation using jobs_test.csv:**
   ```powershell
   python -c "import pandas as pd; from src.evaluate import run_evaluation; jobs_df = pd.read_csv('data/jobs_test.csv'); run_evaluation(jobs_df)"
   ```
   This will print evaluation metrics for the hybrid and TF-IDF models on the held-out test set.

## Usage

- The main recommendation logic is in `src/recommend.py` and `src/vectorize.py`.
- To enable semantic skill expansion, set `semantic_expand=True` in `extract_entities` or use the default in `recommend_jobs`.
- The system will automatically use both exact and semantic skill overlap for scoring.

## Notes

- If you add new skills to the dictionary, they will be used for both exact and semantic matching.
- Unseen skills in resumes or jobs are handled via Sentence-BERT similarity.
- All new logic is modular and commented for clarity.
# AI-Powered Job Recommendation System

## Overview
A modern, hybrid job recommendation engine that combines semantic understanding with keyword precision. This system analyzes candidate resumes and matches them against a dataset of job descriptions using multiple machine learning strategies.

## Key Features
- **Semantic Search**: Uses Sentence-BERT (SBERT) to capture technical context (e.g., matching 'PyTorch' to 'Machine Learning').
- **Explicit Skill Overlap**: Calculates the precise match between resume skills and job requirements using a custom technical dictionary.
- **Explainable AI Integration**: Leveraging Google Gemini to provide human-readable reasoning for every recommendation.
- **Collaborative Behavior Tracking**: Tracks user saving and applying history to refine future matches.
- **Offline Training Pipeline**: Precomputes high-dimensional vectors to ensure sub-second inference in production.

## Technology Stack
- **Backend**: FastAPI (Python)
- **Machine Learning**: Sentence-Transformers, Scikit-Learn, Pandas, NLTK
- **LLM**: Google Generative AI (Gemini 2.5 Flash)
- **Frontend**: React, Tailwind CSS, Lucide Icons

## Getting Started

### Prerequisites
- Python 3.12+
- `GEMINI_API_KEY` (Add to `.env`)

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Initialize the dataset (First time only):
   ```bash
   python train.py
   ```
3. Start the API server:
   ```bash
   uvicorn api:app --reload
   ```
4. Run the frontend:
   ```bash
   cd frontend
   npm run dev
   ```

## Recommendation Logic
The system uses a weighted hybrid scoring formula:
- **50% Semantic similarity** (Sentence-BERT Embeddings)
- **20% TF-IDF similarity** (Term Frequency)
- **20% Skill overlap** (Dictionary-based extraction)
- **10% Experience Alignment** (Seniority match)

---
*Developed for providing transparent and intelligent job matching.*
