"""
Evaluation framework for job recommender system:
- Compares old (TF-IDF+experience) vs new (hybrid) models
- Computes Precision@5, Recall@5, MRR
- Outputs quantitative and qualitative results
"""
import pandas as pd
from src.recommend import recommend_jobs
from src.vectorize import get_tfidf_entities, get_bert_embeddings
from src.preprocess import extract_entities

# 1. Evaluation Dataset (diverse, with unseen skills)
TEST_CASES = [
    {"resume": "python machine learning aws", "expected_roles": ["data scientist", "ml engineer"]},
    {"resume": "react javascript css tailwind", "expected_roles": ["frontend engineer", "frontend developer"]},
    {"resume": "java spring boot sql", "expected_roles": ["backend developer", "java developer"]},
    {"resume": "docker kubernetes ci/cd aws", "expected_roles": ["devops engineer"]},
    {"resume": "pytorch fastapi langchain", "expected_roles": ["ml engineer", "backend developer"]},
    {"resume": "sql r data analysis", "expected_roles": ["data scientist", "analyst"]},
    {"resume": "python flask fastapi", "expected_roles": ["backend developer", "python developer"]},
    {"resume": "typescript react nextjs", "expected_roles": ["frontend engineer"]},
    {"resume": "cloudformation terraform aws", "expected_roles": ["devops engineer"]},
    {"resume": "nlp huggingface pytorch", "expected_roles": ["ml engineer", "data scientist"]},
    {"resume": "excel tableau sql", "expected_roles": ["analyst"]},
    {"resume": "c++ embedded systems", "expected_roles": ["engineer"]},
    {"resume": "javascript nodejs express", "expected_roles": ["backend developer"]},
    {"resume": "react native mobile", "expected_roles": ["frontend engineer"]},
    {"resume": "python langchain llamaindex", "expected_roles": ["ml engineer", "data scientist"]},
]

# 2. Metrics

def precision_at_k(recommended, expected, k=5):
    recommended = recommended[:k]
    hits = sum(1 for r in recommended if r in expected)
    return hits / k

def recall_at_k(recommended, expected, k=5):
    recommended = recommended[:k]
    hits = sum(1 for r in recommended if r in expected)
    return hits / len(expected) if expected else 0.0

def mrr_at_k(recommended, expected, k=5):
    recommended = recommended[:k]
    for idx, r in enumerate(recommended):
        if r in expected:
            return 1.0 / (idx + 1)
    return 0.0

# 3. Model Wrappers

def old_recommender(resume_text, jobs_df, tfidf, job_vectors):
    """
    Only uses TF-IDF similarity + experience (no semantic, no skill overlap)
    """
    # Dummy resume_data with no skills
    resume_data = {"text": resume_text, "experience": 2, "skills": []}
    # Call recommend_jobs with only tfidf and experience (semantic/skill=0)
    # Simulate by setting bert_job_vectors=None and skill_scores=0
    result_df = recommend_jobs(
        resume_data, jobs_df, tfidf, job_vectors, bert_job_vectors=None, top_n=5, skill_threshold=1.0
    )
    # Return top job titles (use 'Job Title' instead of 'Role')
    return result_df["Job Title"].str.lower().tolist()

def new_recommender(resume_text, jobs_df, tfidf, job_vectors, bert_job_vectors):
    """
    Uses full hybrid model (semantic + tfidf + skill + experience)
    """
    resume_data = {"text": resume_text, "experience": 2}
    result_df = recommend_jobs(
        resume_data, jobs_df, tfidf, job_vectors, bert_job_vectors=bert_job_vectors, top_n=5
    )
    return result_df["Job Title"].str.lower().tolist()

# 4. Evaluation Functions

def evaluate_model(model_fn, test_cases, jobs_df, tfidf, job_vectors, bert_job_vectors=None):
    rows = []
    for case in test_cases:
        resume = case["resume"]
        expected = [r.lower() for r in case["expected_roles"]]
        recommended = model_fn(resume, jobs_df, tfidf, job_vectors, bert_job_vectors) if bert_job_vectors is not None else model_fn(resume, jobs_df, tfidf, job_vectors)
        prec = precision_at_k(recommended, expected, k=5)
        rec = recall_at_k(recommended, expected, k=5)
        mrr = mrr_at_k(recommended, expected, k=5)
        rows.append({
            "resume": resume,
            "expected_roles": expected,
            "recommended": recommended,
            "precision@5": prec,
            "recall@5": rec,
            "mrr@5": mrr
        })
    return pd.DataFrame(rows)

def compare_models(old_model_fn, new_model_fn, test_cases, jobs_df, tfidf, job_vectors, bert_job_vectors):
    results = []
    for case in test_cases:
        resume = case["resume"]
        expected = [r.lower() for r in case["expected_roles"]]
        old_rec = old_model_fn(resume, jobs_df, tfidf, job_vectors)
        new_rec = new_model_fn(resume, jobs_df, tfidf, job_vectors, bert_job_vectors)
        old_prec = precision_at_k(old_rec, expected, k=5)
        new_prec = precision_at_k(new_rec, expected, k=5)
        results.append({
            "resume": resume,
            "expected_roles": expected,
            "old_precision": old_prec,
            "new_precision": new_prec,
            "improvement": new_prec - old_prec,
            "old_rec": old_rec,
            "new_rec": new_rec
        })
    return pd.DataFrame(results)

# 5. Main Evaluation Pipeline


def run_evaluation(jobs_df=None):
    """
    Load precomputed artifacts (training already done).
    Evaluates recommender on test cases using saved vectorizers/embeddings.
    """
    import os
    import joblib
    
    # Load precomputed artifacts
    ARTIFACTS_DIR = 'artifacts'
    tfidf_path = os.path.join(ARTIFACTS_DIR, 'tfidf.pkl')
    vectors_path = os.path.join(ARTIFACTS_DIR, 'job_vectors.pkl')
    bert_vectors_path = os.path.join(ARTIFACTS_DIR, 'bert_job_vectors.pkl')
    jobs_path = os.path.join(ARTIFACTS_DIR, 'jobs.pkl')
    
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(f"Artifacts not found. Run train.py first. Missing: {tfidf_path}")
    
    print("Loading precomputed artifacts...")
    tfidf = joblib.load(tfidf_path)
    job_vectors = joblib.load(vectors_path)
    
    # Handle both single bert_job_vectors.pkl and chunked bert_chunk_*.npy files
    if os.path.exists(bert_vectors_path):
        bert_job_vectors = joblib.load(bert_vectors_path)
    else:
        import glob
        bert_chunks = sorted(glob.glob(os.path.join(ARTIFACTS_DIR, 'bert_chunk_*.npy')))
        if bert_chunks:
            print(f"Loading {len(bert_chunks)} BERT chunk files...")
            import numpy as np
            bert_job_vectors = np.vstack([np.load(f) for f in bert_chunks])
        else:
            raise FileNotFoundError(f"No BERT vectors found in artifacts")
    
    jobs_df = pd.read_pickle(jobs_path)
    print(f"Loaded {len(jobs_df)} jobs from artifacts")

    # Evaluate both models using precomputed vectors
    results_df = compare_models(
        old_recommender, new_recommender, TEST_CASES, jobs_df, tfidf, job_vectors, bert_job_vectors
    )

    # Output summary
    old_avg = results_df["old_precision"].mean()
    new_avg = results_df["new_precision"].mean()
    improvement = (new_avg - old_avg) / (old_avg + 1e-8) * 100
    print(f"Old Avg Precision@5: {old_avg:.2f}")
    print(f"New Avg Precision@5: {new_avg:.2f}")
    print(f"Improvement: {improvement:+.0f}%\n")

    # Show sample comparisons
    print(results_df[["resume", "expected_roles", "old_precision", "new_precision", "improvement"]].head(8))

    # Qualitative: show cases where new model is much better
    sig_cases = results_df.sort_values("improvement", ascending=False).head(5)
    print("\nQualitative Examples (Hybrid > TF-IDF):")
    for _, row in sig_cases.iterrows():
        if row["improvement"] > 0.2:
            print(f"\nResume: {row['resume']}")
            print(f"Expected: {row['expected_roles']}")
            print(f"Old: {row['old_rec']}")
            print(f"New: {row['new_rec']}")

    return results_df


if __name__ == "__main__":
    run_evaluation()
