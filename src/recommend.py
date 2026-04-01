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
    skip_skill_extraction=False,
    filters=None
):
    """
    Updated 3-Layer Hybrid Recommender:
    1. Hard Filtering (Location, Experience)
    2. Independent Candidate Expansion (Union Strategy of Top 300)
    3. MMR Diversity Ranking
    """
    resume_text = resume_data.get("text", "")
    
    if filters is None:
        filters = {}

    # =================================================================
    # LAYER 1: Hard Candidate Filtering (Sync Vector Matrix slices)
    # =================================================================
    if filters:
        mask = np.ones(len(df), dtype=bool)
        if "location" in filters and filters["location"]:
            loc_query = str(filters["location"]).lower()
            loc_col = 'Location' if 'Location' in df.columns else 'location' if 'location' in df.columns else None
            if loc_col:
                mask &= df[loc_col].fillna('').astype(str).str.lower().str.contains(loc_query)
        if "max_experience" in filters and filters["max_experience"] is not None:
            max_exp = float(filters["max_experience"])
            mask &= (pd.to_numeric(df["Experience"], errors='coerce').fillna(0) <= max_exp)
            
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return pd.DataFrame() # No jobs match filters
        
        # Slice everything synchronously
        df = df.iloc[valid_indices].reset_index(drop=True)
        job_vectors = job_vectors[valid_indices]
        if bert_job_vectors is not None:
            bert_job_vectors = bert_job_vectors[valid_indices]

    # =================================================================
    # STAGE 1: Fast Vector Similarity (Raw Cosine)
    # =================================================================
    skills_str = " ".join(resume_data.get("skills", []))
    enriched_text = resume_text + " " + skills_str if skills_str else resume_text
    resume_vec = tfidf.transform([enriched_text])
    tfidf_scores = get_similarity(resume_vec, job_vectors)          # raw [0,1]

    if bert_job_vectors is not None:
        resume_emb = get_bert_embeddings([resume_text])[0]
        semantic_scores = get_similarity(resume_emb, bert_job_vectors)  # raw [0,1]
    else:
        semantic_scores = np.zeros_like(tfidf_scores)

    df["tfidf_score"]    = tfidf_scores
    df["semantic_score"] = semantic_scores

    if bert_job_vectors is not None:
        df["draft_score"] = (0.7 * df["semantic_score"]) + (0.3 * df["tfidf_score"])
    else:
        df["draft_score"] = df["tfidf_score"]

    np.random.seed(hash(resume_text) % (2**31))
    df["draft_score"] += np.random.uniform(0, 0.01, len(df))

    # =================================================================
    # LAYER 2: Candidate Expansion (Independent Union Strategy)
    # =================================================================
    EXPAND_N = min(300, len(df))
    tfidf_top_idx = np.argsort(tfidf_scores)[::-1][:EXPAND_N]
    
    if bert_job_vectors is not None:
        bert_top_idx = np.argsort(semantic_scores)[::-1][:EXPAND_N]
        # Mathematical Union of the highest hits from both distinct systems
        expanded_idx = np.unique(np.concatenate([tfidf_top_idx, bert_top_idx]))
    else:
        expanded_idx = tfidf_top_idx
        
    top_candidates = df.iloc[expanded_idx].copy()
    print(f"  [RANK] Expanded Re-ranking pool: {len(top_candidates)} unique candidates from Union.")

    norm_temp = pd.DataFrame()
    norm_temp['title'] = top_candidates['Job Title'].fillna('').astype(str).str.strip().str.lower()
    comp_col = 'Company' if 'Company' in top_candidates.columns else 'Company Name' if 'Company Name' in top_candidates.columns else None
    if comp_col:
        norm_temp['comp'] = top_candidates[comp_col].fillna('').astype(str).str.strip().str.lower()
    
    subset = ['title']
    if 'comp' in norm_temp.columns: subset.append('comp')
    top_candidates = top_candidates.loc[norm_temp.drop_duplicates(subset=subset, keep='first').index]
    
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
    
    # Final Hybrid Score
    top_candidates["final_score"] = (
        (0.60 * top_candidates["semantic_score"]) +
        (0.15 * top_candidates["tfidf_score"]) +
        (0.20 * top_candidates["skill_score"]) +
        (0.05 * top_candidates["exp_score"])
    )
    top_candidates["final_score"] += 0.001 * top_candidates["draft_score"]

    results = top_candidates.sort_values(by="final_score", ascending=False)
    
    # Limit to top 100 strictly for the MMR computation payload
    results = results.head(100)

    # =================================================================
    # LAYER 3: Maximal Marginal Relevance (MMR) Diversity Ranker
    # =================================================================
    if len(results) <= top_n or bert_job_vectors is None:
        final_results = results.head(top_n)
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        # Extract embeddings for the 50 selected items
        # `results.index` matches the valid filtered `df` and `bert_job_vectors` index!
        candidate_embs = bert_job_vectors[results.index.values]
        
        # Calculate Item-to-Item Similarity penalty matrix
        sim_matrix = cosine_similarity(candidate_embs)
        
        selected_idx = []
        unselected_idx = list(range(len(results)))
        
        lambda_param = 0.70 # Balance 70% Context-Relevance, 30% Diversity
        
        # Always pick the absolute BEST item first
        selected_idx.append(unselected_idx.pop(0))
        
        while len(selected_idx) < top_n and unselected_idx:
            best_mmr_score = -float('inf')
            best_idx = -1
            
            for i in unselected_idx:
                relevance = results.iloc[i]["final_score"]
                # Penalty is max similarity to any ALREADY selected items
                penalty = np.max(sim_matrix[i, selected_idx])
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * penalty
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i
                    
            selected_idx.append(best_idx)
            unselected_idx.remove(best_idx)
            
        final_results = results.iloc[selected_idx]

    return final_results
