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

    Full mode : 0.60 Semantic + 0.15 TF-IDF + 0.20 Skill + 0.05 Exp
    """

    resume_text = resume_data.get("text", "")

    # =================================================================
    # STAGE 1: Fast Filtering (Vectorized — raw cosine, no normalization)
    # =================================================================
    # Raw cosine similarity is already on [0,1] for non-negative TF-IDF
    # vectors. Min-max normalization collapsed all tied jobs to score=1.0,
    # which is WHY 20x the same job appeared. Use raw scores instead.

    # 1. TF-IDF Similarity
    skills_str = " ".join(resume_data.get("skills", []))
    enriched_text = resume_text + " " + skills_str if skills_str else resume_text
    resume_vec = tfidf.transform([enriched_text])
    tfidf_scores = get_similarity(resume_vec, job_vectors)          # raw [0,1]

    # 2. BERT Semantic Similarity (always computed — one forward pass ~50ms)
    if bert_job_vectors is not None:
        resume_emb = get_bert_embeddings([resume_text])[0]
        semantic_scores = get_similarity(resume_emb, bert_job_vectors)  # raw [0,1]
    else:
        semantic_scores = np.zeros_like(tfidf_scores)

    df["tfidf_score"]    = tfidf_scores
    df["semantic_score"] = semantic_scores

    print(f"  [DEBUG] TF-IDF  top-5 (raw): {sorted(tfidf_scores, reverse=True)[:5]}")
    print(f"  [DEBUG] Semantic top-5 (raw): {sorted(semantic_scores, reverse=True)[:5]}")

    # Draft score — BERT semantic is the primary tie-breaker
    if bert_job_vectors is not None:
        df["draft_score"] = (0.7 * df["semantic_score"]) + (0.3 * df["tfidf_score"])
    else:
        df["draft_score"] = df["tfidf_score"]

    # Jitter (0.01 scale) breaks remaining exact ties; seeded so same resume
    # always returns the same ordered list (deterministic results)
    np.random.seed(hash(resume_text) % (2**31))
    df["draft_score"] += np.random.uniform(0, 0.01, len(df))
    
    # =================================================================
    # STAGE 2: Detailed Re-ranking (Top 1000)
    # =================================================================
    RE_RANK_N = min(1000, len(df))        # larger pool → more diverse output
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
    
    print(f"  [RANK] Re-ranking top {len(top_candidates)} unique candidates...")
    
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
    # FIX 2: Different weight splits for fast vs full mode
    # =================================================================
    # STAGE 3: Final Hybrid Score
    # =================================================================
    top_candidates["final_score"] = (
        (0.60 * top_candidates["semantic_score"]) +
        (0.15 * top_candidates["tfidf_score"]) +
        (0.20 * top_candidates["skill_score"]) +
        (0.05 * top_candidates["exp_score"])
    )

    # Carry draft_score forward as a micro tiebreaker so when hundreds
    # of jobs share identical semantic/TF-IDF scores the jitter from
    # Stage 1 still propagates to the final ordering.
    top_candidates["final_score"] += 0.001 * top_candidates["draft_score"]

    print(f"  [DEBUG] Final top-5 scores: {top_candidates['final_score'].nlargest(5).tolist()}")

    results = top_candidates.sort_values(by="final_score", ascending=False)

    # ------------------------------------------------------------------
    # Diversity-aware dedup
    # Goal: up to top_n results with NO exact-duplicate listings AND
    #       no more than MAX_PER_TITLE results with the same job title.
    # This prevents "100x Sales Consultant" even when they all score identically.
    # ------------------------------------------------------------------
    MAX_PER_TITLE = 5   # max cards with the same title in the final list

    comp_col_final = (
        'Company' if 'Company' in results.columns
        else 'Company Name' if 'Company Name' in results.columns
        else None
    )
    loc_col_final = (
        'Location' if 'Location' in results.columns
        else 'location' if 'location' in results.columns
        else None
    )

    seen_listings = set()   # exact duplicate guard (title+company+location)
    title_counts  = {}      # per-title count for diversity cap
    diverse_idx   = []

    for idx, row in results.iterrows():
        title = str(row.get('Job Title', '')).strip().lower()
        comp  = str(row.get(comp_col_final, '') if comp_col_final else '').strip().lower()
        loc   = str(row.get(loc_col_final,  '') if loc_col_final  else '').strip().lower()

        listing_key = f"{title}|{comp}|{loc}"

        if listing_key in seen_listings:
            continue  # exact duplicate — skip

        if title_counts.get(title, 0) >= MAX_PER_TITLE:
            continue  # too many of this title already — skip

        seen_listings.add(listing_key)
        title_counts[title] = title_counts.get(title, 0) + 1
        diverse_idx.append(idx)

        if len(diverse_idx) >= top_n:
            break

    return results.loc[diverse_idx]
