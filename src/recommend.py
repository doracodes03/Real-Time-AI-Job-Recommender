import pandas as pd
import numpy as np

from src.vectorize import (
    get_similarity,
    get_bert_embeddings
)
from src.preprocess import (
    extract_entities,
    semantic_skill_overlap
)


# -------------------------------
# Experience Score
# -------------------------------
def calculate_experience_score(resume_exp, job_exp):
    try:
        resume_exp = float(resume_exp)
        job_exp = float(job_exp)

        if resume_exp >= job_exp:
            return 1.0
        return max(0.0, resume_exp / job_exp)

    except (ValueError, TypeError):
        return 0.0


# -------------------------------
# Skill Overlap (cleaned)
# -------------------------------
def calculate_skill_overlap(resume_skills, job_skills, threshold=0.75):
    if not job_skills:
        return 0.0

    resume_set = set(s.lower() for s in resume_skills)
    job_set = set(s.lower() for s in job_skills)

    exact = len(resume_set & job_set) / len(job_set)

    try:
        semantic = semantic_skill_overlap(resume_skills, job_skills, threshold)
        return max(exact, semantic)
    except:
        return exact


# -------------------------------
# Main Recommender
# -------------------------------
def recommend_jobs(
    resume_data,
    df,
    tfidf,
    job_vectors,
    bert_job_vectors=None,
    top_n=5,
    skill_threshold=0.75,
    skip_bert_embedding=False,
    skip_skill_extraction=False
):
    """
    Updated Hybrid Recommender:

    - 0.6 Embedding similarity (primary signal) - skip if skip_bert_embedding=True
    - 0.2 TF-IDF similarity (keyword precision)
    - 0.15 Skill overlap (structured signal)
    - 0.05 Experience (minor boost)
    
    Args:
        skip_bert_embedding: If True, skip resume embedding computation and use TF-IDF only
        skip_skill_extraction: If True, skip expensive skill extraction with BERT
    """

    resume_text = resume_data.get("text", "")

    # -------------------------------
    # 1. TF-IDF Similarity
    # -------------------------------
    resume_vec = tfidf.transform([resume_text])
    tfidf_scores = get_similarity(resume_vec, job_vectors)

    # -------------------------------
    # 2. Embedding Similarity (Fix 3)
    # -------------------------------
    if bert_job_vectors is not None and not skip_bert_embedding:
        # Only compute resume embeddings if needed (expensive on CPU)
        resume_emb = get_bert_embeddings([resume_text])[0]
        semantic_scores = get_similarity(resume_emb, bert_job_vectors)
    else:
        # fallback to TF-IDF (faster)
        semantic_scores = tfidf_scores

    # -------------------------------
    # 3. Skill Extraction (once) - SKIPPABLE for speed
    # -------------------------------
    if skip_skill_extraction:
        # ⚡ FAST PATH: Skip all skill extraction
        resumed_skills = []
        skill_scores = [0.0] * len(df)  # All zeros, rely on BERT + TF-IDF
    else:
        # Standard path: Extract skills
        resume_skills = resume_data.get("skills")

        if not resume_skills:
            resume_skills = extract_entities(
                resume_text,
                semantic_expand=True,
                threshold=skill_threshold
            ).get("skills", [])

        # -------------------------------
        # 4. Skill Scores (OPTIMIZED - skip if no skills)
        # -------------------------------
        skill_scores = []

        if resume_skills and len(resume_skills) > 0:
            # Only extract job skills if we have resume skills to compare
            for _, row in df.iterrows():
                job_text = row.get("Job Description", "")

                job_skills = extract_entities(
                    job_text,
                    semantic_expand=True,
                    threshold=skill_threshold
                ).get("skills", [])

                score = calculate_skill_overlap(
                    resume_skills,
                    job_skills,
                    threshold=skill_threshold
                )

                skill_scores.append(score)
        else:
            # If no skills provided, use job description match instead
            skill_scores = [0.0] * len(df)  # All zeros, rely on BERT + TF-IDF

    # -------------------------------
    # 5. Experience Scores
    # -------------------------------
    resume_exp = resume_data.get("experience", 0)

    exp_scores = [
        calculate_experience_score(resume_exp, job_exp)
        for job_exp in df["Experience"]
    ]

    # -------------------------------
    # Store intermediate scores
    # -------------------------------
    df["tfidf_score"] = tfidf_scores
    df["semantic_score"] = semantic_scores
    df["skill_score"] = skill_scores
    df["exp_score"] = exp_scores

    # -------------------------------
    # 6. Final Hybrid Score (Fix 4)
    # -------------------------------
    df["final_score"] = (
        (0.6 * df["semantic_score"]) +   # 🔥 dominant signal
        (0.2 * df["tfidf_score"]) +
        (0.15 * df["skill_score"]) +
        (0.05 * df["exp_score"])
    )

    # -------------------------------
    # Return Top N (with aggressive duplicate removal)
    # -------------------------------
    results = df.sort_values(by="final_score", ascending=False)
    
    # Try multiple duplicate-removal strategies
    initial_count = len(results)
    
    # Strategy 1: Remove by ID
    if 'id' in results.columns:
        results = results.drop_duplicates(subset=['id'], keep='first')
    
    # Strategy 2: Remove by Job Title + Company (in case IDs are not unique)
    if 'Job Title' in results.columns and 'Company' in results.columns:
        results = results.drop_duplicates(subset=['Job Title', 'Company'], keep='first')
    
    # Strategy 3: Remove by exact row content hash (catch sneaky duplicates)
    if len(results) > 1:
        results = results.drop_duplicates(subset=['Job Title', 'Company', 'Location'], keep='first')
    
    final_count = len(results)
    if initial_count != final_count:
        print(f"  Removed {initial_count - final_count} duplicates")
    
    return results.head(top_n)