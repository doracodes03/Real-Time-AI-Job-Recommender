import joblib
import os
import pandas as pd
from src.recommend import recommend_jobs

ARTIFACTS_DIR = 'artifacts'

if not os.path.exists(ARTIFACTS_DIR):
    print("Error: Artifacts directory not found. Please run train.py first.")
    exit(1)

print("Loading artifacts...")
tfidf = joblib.load(os.path.join(ARTIFACTS_DIR, 'tfidf.pkl'))
job_vectors = joblib.load(os.path.join(ARTIFACTS_DIR, 'job_vectors.pkl'))
df = pd.read_pickle(os.path.join(ARTIFACTS_DIR, 'jobs.pkl'))

# Optional BERT vectors
bert_path = os.path.join(ARTIFACTS_DIR, 'bert_job_vectors.pkl')
bert_job_vectors = joblib.load(bert_path) if os.path.exists(bert_path) else None

print(f"Loaded {len(df)} jobs.")
print(f"TF-IDF Vectors shape: {job_vectors.shape}")
if bert_job_vectors is not None:
    print(f"BERT Vectors shape: {bert_job_vectors.shape}")
else:
    print("Notice: BERT vectors not found. Using fallback scoring.")

resume_data = {
    "text": "python developer with 3 years of experience in machine learning and sql",
    "experience": 3,
    "skills": ["python", "machine learning", "sql"]
}

print("\n--- Running Hybrid Recommendation ---")
try:
    results = recommend_jobs(
        resume_data=resume_data, 
        df=df, 
        tfidf=tfidf, 
        job_vectors=job_vectors, 
        bert_job_vectors=bert_job_vectors,
        top_n=5
    )
    print("Recommendations successful!")
    # Show internal scores for verification
    cols_to_show = ['Job Title', 'final_score', 'semantic_score', 'tfidf_score', 'skill_score', 'exp_score']
    available_cols = [c for c in cols_to_show if c in results.columns]
    print(results[available_cols])
except Exception as e:
    print(f"Error during recommendation: {e}")
