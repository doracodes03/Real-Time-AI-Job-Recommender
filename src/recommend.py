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
def calculate_skill_overlap(resume_skills, job_skills, threshold=0.75, resume_embs=None):
    if not job_skills:
        return 0.0

    resume_set = set(s.lower() for s in resume_skills)
    job_set = set(s.lower() for s in job_skills)

    exact = len(resume_set & job_set) / len(job_set)

    try:
        semantic = semantic_skill_overlap(resume_skills, job_skills, threshold, resume_embs=resume_embs)
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
    """

    resume_text = resume_data.get("text", "")

    # =================================================================
    # STAGE 1: Fast Filtering (Vectorized)
    # =================================================================
    # 0.6 Semantic + 0.4 TF-IDF
    
    # 1. TF-IDF Similarity (always fast)
    resume_vec = tfidf.transform([resume_text])
    tfidf_scores = get_similarity(resume_vec, job_vectors)
    
    # 2. Embedding Similarity (Fast if vectors are pre-computed)
    if bert_job_vectors is not None and not skip_bert_embedding:
        resume_emb = get_bert_embeddings([resume_text])[0]
        semantic_scores = get_similarity(resume_emb, bert_job_vectors)
    else:
        semantic_scores = tfidf_scores

    df["tfidf_score"] = tfidf_scores
    df["semantic_score"] = semantic_scores
    
    # Initial Draft Score (no skills/exp yet)
    df["draft_score"] = (0.7 * df["semantic_score"]) + (0.3 * df["tfidf_score"])
    
    # =================================================================
    # STAGE 2: Detailed Re-ranking (Top 500)
    # =================================================================
    RE_RANK_N = min(500, len(df))
    top_candidates = df.nlargest(RE_RANK_N, "draft_score").copy()
    
    # Quick duplicate removal on top candidates to save time
    norm_temp = pd.DataFrame()
    norm_temp['title'] = top_candidates['Job Title'].fillna('').astype(str).str.strip().str.lower()
    comp_col = 'Company' if 'Company' in top_candidates.columns else 'Company Name' if 'Company Name' in top_candidates.columns else None
    if comp_col:
        norm_temp['comp'] = top_candidates[comp_col].fillna('').astype(str).str.strip().str.lower()
    
    subset = ['title']
    if 'comp' in norm_temp.columns: subset.append('comp')
    top_candidates = top_candidates.loc[norm_temp.drop_duplicates(subset=subset, keep='first').index]
    
    print(f"  ⚡ Re-ranking top {len(top_candidates)} unique candidates...")
    
    # 3. Skill Scores (BATCH OPTIMIZED)
    resume_skills = resume_data.get("skills")
    if not resume_skills and not skip_skill_extraction:
        resume_skills = extract_entities(resume_text, semantic_expand=True).get("skills", [])
    
    resume_skill_embs = None
    if resume_skills:
        resume_skill_embs = get_bert_embeddings(resume_skills)
    
    job_skills_col = 'skills' if 'skills' in df.columns else 'Skills' if 'Skills' in df.columns else None
    all_job_skill_lists = []
    unique_job_skills = set()
    
    for _, row in top_candidates.iterrows():
        if job_skills_col:
            skills = row[job_skills_col]
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(',')]
        else:
            skills = extract_entities(row.get("Job Description", ""), semantic_expand=True).get("skills", [])
        
        all_job_skill_lists.append(skills)
        for s in skills: unique_job_skills.add(s)
            
    unique_job_skills = sorted(list(unique_job_skills))
    if unique_job_skills and resume_skill_embs is not None:
        all_unique_embs = get_bert_embeddings(unique_job_skills)
        skill_to_emb = dict(zip(unique_job_skills, all_unique_embs))
    else:
        skill_to_emb = {}
        
    skill_scores = []
    for job_skills in all_job_skill_lists:
        if not resume_skills or not job_skills or resume_skill_embs is None:
            skill_scores.append(0.0)
            continue
        job_embs = np.array([skill_to_emb[s] for s in job_skills if s in skill_to_emb])
        if len(job_embs) == 0:
            skill_scores.append(0.0)
            continue
        sim_matrix = np.dot(resume_skill_embs, job_embs.T) / (
            np.linalg.norm(resume_skill_embs, axis=1, keepdims=True) * np.linalg.norm(job_embs, axis=1)
        )
        matches = (sim_matrix >= skill_threshold).sum()
        skill_scores.append(matches / len(job_skills))
        
    top_candidates["skill_score"] = skill_scores
    
    # 4. Experience Scores
    resume_exp = resume_data.get("experience", 0)
    top_candidates["exp_score"] = [
        calculate_experience_score(resume_exp, exp)
        for exp in top_candidates["Experience"]
    ]
    
    # =================================================================
    # STAGE 3: Final Hybrid Score
    # =================================================================
    top_candidates["final_score"] = (
        (0.60 * top_candidates["semantic_score"]) +
        (0.15 * top_candidates["tfidf_score"]) +
        (0.20 * top_candidates["skill_score"]) +
        (0.05 * top_candidates["exp_score"])
    )
    
    results = top_candidates.sort_values(by="final_score", ascending=False)
    
    # Final aggressive drop
    norm_final = pd.DataFrame()
    norm_final['title'] = results['Job Title'].fillna('').astype(str).str.strip().str.lower()
    if comp_col:
        norm_final['comp'] = results[comp_col].fillna('').astype(str).str.strip().str.lower()
    
    loc_col = 'Location' if 'Location' in results.columns else 'location' if 'location' in results.columns else None
    if loc_col:
        norm_final['loc'] = results[loc_col].fillna('').astype(str).str.strip().str.lower()

    subset = ['title']
    if 'comp' in norm_final.columns: subset.append('comp')
    if 'loc' in norm_final.columns: subset.append('loc')
    
    results = results.loc[norm_final.drop_duplicates(subset=subset, keep='first').index]
        
    return results.head(top_n)