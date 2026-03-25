import pandas as pd
import numpy as np
from src.vectorize import get_similarity, get_bert_embeddings
from src.preprocess import extract_entities, semantic_skill_overlap

def calculate_experience_score(resume_exp, job_exp):
    """
    Score is 1.0 if resume experience is >= job experience.
    Provides a linear drop-off if you have less experience than required.
    """
    try:
        resume_exp = float(resume_exp)
        job_exp = float(job_exp)
        if resume_exp >= job_exp:
            return 1.0
        return max(0.0, resume_exp / job_exp)
    except (ValueError, TypeError):
        return 0.0

def calculate_skill_overlap(resume_skills, job_text, use_semantic=True, threshold=0.75):
    """
    Calculates overlap between resume skills and skills extracted from the job description.
    Uses both exact and semantic overlap (Sentence-BERT) if enabled.
    """
    job_entities = extract_entities(job_text)
    job_skills = job_entities.get("skills", [])
    if not job_skills:
        return 0.0
    # Exact overlap
    resume_skills_set = set([s.lower() for s in resume_skills])
    job_skills_set = set([s.lower() for s in job_skills])
    matches = job_skills_set.intersection(resume_skills_set)
    exact_score = len(matches) / len(job_skills_set)
    # Semantic overlap (for unseen/related skills)
    if use_semantic:
        try:
            sem_score = semantic_skill_overlap(resume_skills, job_skills, threshold=threshold)
            # Combine: max of exact/semantic (or average)
            return max(exact_score, sem_score)
        except Exception as e:
            # Fallback to exact if embedding fails
            return exact_score
    return exact_score

def recommend_jobs(resume_data, df, tfidf, job_vectors, bert_job_vectors=None, top_n=5, skill_threshold=0.75):
    """
    Hybrid recommendation engine with following weights:
    - 0.5 Semantic similarity (SBERT) - Captures deep meaning (e.g., 'pytorch' vs 'ML')
    - 0.2 TF-IDF similarity - Keyword precision
    - 0.2 Skill overlap - Explicit matching from custom dictionary
    - 0.1 Experience match - Seniority alignment
    """
    
    # 1. TF-IDF Match (0.2 weight) - Using precomputed sparse matrix
    resume_text = resume_data.get("text", "")
    resume_vec = tfidf.transform([resume_text])
    tfidf_scores = get_similarity(resume_vec, job_vectors)
    
    # 2. Semantic Match (0.5 weight) - SBERT handles "unseen" skills/synonyms
    if bert_job_vectors is not None:
        resume_embedding = get_bert_embeddings(resume_text)
        semantic_scores = get_similarity(resume_embedding, bert_job_vectors)
    else:
        # Fallback if BERT vectors not precomputed (not ideal)
        semantic_scores = tfidf_scores 

    # 3. Skill Overlap (0.2 weight) - Uses extract_entities from preprocess.py
    resume_skills = resume_data.get("skills", [])
    if not resume_skills:
        resume_skills = extract_entities(resume_text, semantic_expand=True, threshold=skill_threshold).get("skills", [])
    # Compute skill overlap (semantic-aware)
    skill_scores = []
    for _, row in df.iterrows():
        job_text = row.get('Job Description', '')
        score = calculate_skill_overlap(resume_skills, job_text, use_semantic=True, threshold=skill_threshold)
        skill_scores.append(score)
    
    # 4. Experience Match (0.1 weight) - Linear drop-off for gaps
    resume_exp = resume_data.get("experience", 0)
    exp_scores = [calculate_experience_score(resume_exp, job_exp) for job_exp in df['Experience']]
    
    # Store individual scores for transparency
    # Store individual scores for transparency
    df['tfidf_score'] = tfidf_scores
    df['semantic_score'] = semantic_scores
    df['skill_score'] = skill_scores
    df['exp_score'] = exp_scores
    # Apply the Hybrid Formula (modular, weighted)
    df['final_score'] = (
        (0.5 * df['semantic_score']) +
        (0.2 * df['tfidf_score']) +
        (0.2 * df['skill_score']) +
        (0.1 * df['exp_score'])
    )
    # Return top N results
    return df.sort_values(by='final_score', ascending=False).head(top_n)