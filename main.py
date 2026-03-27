# main.py

import pandas as pd
import numpy as np
import os
import glob
from src.preprocess import preprocess_text
from src.vectorize import get_tfidf_entities
from src.recommend import recommend_jobs
from src.llm_parser import parse_resume_with_llm
from src.llm_explainer import explain_match

import joblib

print("Starting Job Recommender Pipeline (Inference Mode)...")

# 1. Load context from artifacts
ARTIFACTS_DIR = 'artifacts'

# Smart BERT loading (handle both single file and chunks)
def load_bert_vectors_smart(artifacts_dir):
    bert_vectors_path = os.path.join(artifacts_dir, 'bert_job_vectors.pkl')
    
    if os.path.exists(bert_vectors_path):
        print("  Loading single BERT vectors file...")
        return joblib.load(bert_vectors_path)
    
    bert_chunks = sorted(glob.glob(os.path.join(artifacts_dir, 'bert_chunk_*.npy')))
    if bert_chunks:
        print(f"  Loading {len(bert_chunks)} BERT chunk files...")
        return np.vstack([np.load(f) for f in bert_chunks])
    
    return None

print("Loading artifacts...")
tfidf = joblib.load(os.path.join(ARTIFACTS_DIR, 'tfidf.pkl'))
job_vectors = joblib.load(os.path.join(ARTIFACTS_DIR, 'job_vectors.pkl'))
bert_job_vectors = load_bert_vectors_smart(ARTIFACTS_DIR)
df = pd.read_pickle(os.path.join(ARTIFACTS_DIR, 'jobs.pkl'))
print(f"✓ Loaded {len(df)} jobs from artifacts.")
if bert_job_vectors is not None:
    print(f"✓ Loaded BERT vectors: shape {bert_job_vectors.shape}")
else:
    print(f"⚠️  BERT vectors not available (will use TF-IDF only)")

# 4. Sample resume
resume_text = """
Python developer with 4 years of experience in machine learning,
data analysis, and SQL. I have built robust APIs using React and AWS.
"""

print("\n--- Phase 1: LLM Parsing Resume ---")
resume_data = parse_resume_with_llm(resume_text)
# Add raw text for TF-IDF Description math
resume_data["text"] = preprocess_text(resume_text)

print(f"Extracted Parse: {resume_data}")

# 5. Recommend (Inference Mode)
print("\n--- Phase 2: Algorithm Ranking (Hybrid) ---")
results = recommend_jobs(
    resume_data=resume_data,
    df=df,
    tfidf=tfidf,
    job_vectors=job_vectors,
    bert_job_vectors=bert_job_vectors,
    top_n=3
)

# Handle both column names
if 'Company Name' in results.columns:
    print(results[['Job Title', 'Company Name', 'final_score']].to_string(index=False))
elif 'Company' in results.columns:
    print(results[['Job Title', 'Company', 'final_score']].to_string(index=False))
else:
    print(results[['Job Title', 'final_score']].to_string(index=False))

# 6. Smart Explanations (Phase 3 LLM Explainer)
print("\n--- Phase 3: Smart Explanations ---")
for index, row in results.iterrows():
    company = row.get('Company Name') or row.get('Company', 'Unknown')
    print(f"\nEvaluating: {row['Job Title']} at {company}")
    
    explanation = explain_match(
        resume_data=resume_data,
        job_title=row['Job Title'],
        job_desc=row.get('Job Description', ''),
        job_skills=row.get('skills') or row.get('Skills', []),
        ranker_score=row['final_score']
    )
    
    print(f"  Score: {explanation.get('score', 0)}/100")
    print(f"  Matched Skills: {', '.join(explanation.get('matched_skills', []))}")
    print(f"  Missing Skills: {', '.join(explanation.get('missing_skills', []))}")
    print(f"  Reason: {explanation.get('reason', '')}")
    print(f"  Suggestions: {', '.join(explanation.get('suggestions', []))}")
    print("-" * 40)