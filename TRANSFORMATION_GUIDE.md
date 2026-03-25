# Transformation Guide: Hybrid Semantic Reconciler

This document explains the technical refactoring performed to upgrade the job recommendation system from a simple TF-IDF engine to a sophisticated Hybrid Semantic Engine.

## Problems Addressed

### 1. Lack of Semantic Understanding
**Problem**: The previous TF-IDF system relied on exact keyword matches. It couldn't understand that "PyTorch" is semantically related to "Machine Learning" or "Deep Learning."
**Fix**: Integrated **Sentence-BERT (SBERT)** using the `all-MiniLM-L6-v2` model. This allows the system to compare the "meaning" of a resume against the job description, even if the exact keywords differ.

### 2. Ignoring Custom Dictionary in Scoring
**Problem**: Skills extracted via `src/preprocess.py` were not directly contributing to the final match score.
**Fix**: Added a **Skill Overlap Score**. We now explicitly calculate the percentage of required job skills (found in our dictionary) that appear in the candidate's resume. This ensures that having the "right" hard skills is heavily rewarded.

### 3. Neglecting Unseen Skills
**Problem**: Skills not in the `SKILLS_DICT` were completely invisible to the algorithm.
**Fix**: By shifting 50% of the weight to **Semantic Embeddings**, "unseen" skills now contribute through their vector representation. If a candidate has a new library or tool not yet in the dictionary, SBERT will still find jobs with similar technical contexts.

### 4. Poor Weighting & Experience Gaps
**Problem**: Scoring was simplistic and didn't allow for fine-tuning based on importance.
**Fix**: Implemented a **Weighted Hybrid Formula**:
- **50% Semantic similarity** (The "vibe" and context)
- **20% TF-IDF similarity** (The exact keywords)
- **20% Skill overlap** (The hard requirements)
- **10% Experience alignment** (The seniority)

## Key Technical Changes

### `src/recommend.py`
Refactored the `recommend_jobs` function to compute 4 separate scores and combine them. 

### `train.py`
Upgraded the offline pipeline to precompute 384-dimensional BERT embeddings for the entire job catalog, ensuring that the API remains fast (<1 second) during inference.

### `src/vectorize.py`
Implemented **Lazy Loading** for the BERT model to optimize memory usage in the FastAPI environment.
