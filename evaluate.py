import pandas as pd
import numpy as np
import joblib
import os
import glob
from src.recommend import recommend_jobs
from src.vectorize import get_bert_embeddings
from src.preprocess import extract_entities, preprocess_text

# =================================================================
# METRICS: Precision, Recall, and normalized Discounted Cumulative Gain
# =================================================================

def dcg_at_k(relevances, k):
    """Calculate Discounted Cumulative Gain at rank K."""
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.

def precision_at_k(relevances, k):
    """Percentage of top K items that are relevant."""
    relevances = relevances[:k]
    return np.sum(relevances) / k if k > 0 else 0.

def recall_at_k(relevances, total_relevant, k):
    """Percentage of total relevant items retrieved in top K."""
    relevances = relevances[:k]
    return np.sum(relevances) / total_relevant if total_relevant > 0 else 0.

# =================================================================
# SYNTHETIC TEST GENERATOR (The "Leave-One-Out" Ground Truth Method)
# =================================================================

def generate_synthetic_test_cases(jobs_df, sample_size=500):
    print(f"🧬 Generating {sample_size} synthetic test cases from the 100k+ dataset...")
    
    # 1. First, compute total counts of each exact job title acting as "Ground Truth Relevant count"
    title_counts = jobs_df['Job Title'].value_counts()
    
    # 2. Filter to jobs that have at least 1 actual match (prevents division by zero)
    valid_jobs = jobs_df[jobs_df['Job Title'].map(title_counts) >= 1]
    
    # 3. Random Sample our defined "N" queries
    sample_df = valid_jobs.sample(n=min(sample_size, len(valid_jobs)), random_state=42)
    
    test_cases = []
    for _, row in sample_df.iterrows():
        title = str(row['Job Title'])
        skills = str(row.get('skills', row.get('Skills', '')))
        desc = str(row.get('Job Description', ''))[:1000] # Truncated for speed 
        
        # Build synthetic resume purely from the job's underlying properties
        synthetic_resume = preprocess_text(f"{title} {skills} {desc}")
        
        test_cases.append({
            "resume_text": synthetic_resume,
            "target_title": title, # The exact job title we want matching (1 = relevant, 0 = missed)
            "total_relevant": title_counts[title] # Total count in DB
        })
        
    print(f"✅ Generated {len(test_cases)} target tracking cases.")
    return test_cases

# =================================================================
# RECOMMENDER WRAPPERS
# =================================================================

def old_recommender(resume_text, jobs_df, tfidf, job_vectors, k=10):
    resume_data = {"text": resume_text, "experience": 3, "skills": []} # No semantic fields provided
    result_df = recommend_jobs(resume_data, jobs_df.copy(), tfidf, job_vectors, bert_job_vectors=None, top_n=k)
    return result_df["Job Title"].tolist() if 'Job Title' in result_df.columns else []

def new_recommender(resume_text, jobs_df, tfidf, job_vectors, bert_job_vectors, k=10):
    resume_data = {"text": resume_text, "experience": 3}
    result_df = recommend_jobs(resume_data, jobs_df.copy(), tfidf, job_vectors, bert_job_vectors=bert_job_vectors, top_n=k)
    return result_df["Job Title"].tolist() if 'Job Title' in result_df.columns else []

# =================================================================
# EVALUATION HARNESS
# =================================================================

def compare_models(test_cases, jobs_df, tfidf, job_vectors, bert_job_vectors, k=10):
    results = []
    print(f"⏳ Running evaluations on {len(test_cases)} jobs vs 100k artifacts. This may take 1-3 minutes...")
    
    for idx, case in enumerate(test_cases):
        if idx > 0 and idx % 50 == 0:
            print(f"  ...processed {idx}/{len(test_cases)} cases")
            
        resume = case["resume_text"]
        target = case["target_title"].strip().lower()
        total_rel = case["total_relevant"]
        
        try:
            # 1. Fetch Top K
            old_rec = old_recommender(resume, jobs_df, tfidf, job_vectors, k)
            new_rec = new_recommender(resume, jobs_df, tfidf, job_vectors, bert_job_vectors, k)
            
            # 2. Map recommendations to Relevances (1 if matched title, 0 if not)
            old_relevances = [1 if t.strip().lower() == target else 0 for t in old_rec]
            new_relevances = [1 if t.strip().lower() == target else 0 for t in new_rec]
            
            # 3. Best possible DCG for Ideal NDCG baseline (maximum of K hits)
            ideal_relevances = [1] * min(k, total_rel)
            ideal_dcg = dcg_at_k(ideal_relevances, k)
            
            # 4. Compute metrics
            old_ndcg = dcg_at_k(old_relevances, k) / ideal_dcg if ideal_dcg > 0 else 0.
            new_ndcg = dcg_at_k(new_relevances, k) / ideal_dcg if ideal_dcg > 0 else 0.

            old_prec = precision_at_k(old_relevances, k)
            new_prec = precision_at_k(new_relevances, k)
            
            old_rec_score = recall_at_k(old_relevances, total_rel, k)
            new_rec_score = recall_at_k(new_relevances, total_rel, k)
            
            results.append({
                "target_title": target,
                "old_precision": old_prec, "new_precision": new_prec,
                "old_recall": old_rec_score, "new_recall": new_rec_score,
                "old_ndcg": old_ndcg, "new_ndcg": new_ndcg
            })
            
        except Exception as e:
            # Skip failures to not disrupt giant loop
            pass
            
    return pd.DataFrame(results)

def load_bert_vectors_smart(artifacts_dir):
    bert_path = os.path.join(artifacts_dir, 'bert_job_vectors.pkl')
    if os.path.exists(bert_path):
        return joblib.load(bert_path)
    # Check chunks incase Colab downloaded chunk format
    chunks = sorted(glob.glob(os.path.join(artifacts_dir, 'bert_chunk_*.npy')))
    if chunks:
        return np.vstack([np.load(f) for f in chunks])
    return None

def run_evaluation(sample_size=500):
    print("="*80)
    print("🚀 AUTOMATED SYNTHETIC EVALUATION FRAMEWORK (100K DATASET)")
    print("="*80)
    
    ARTIFACTS_DIR = 'artifacts'
    tfidf_path = os.path.join(ARTIFACTS_DIR, 'tfidf.pkl')
    jobs_path = os.path.join(ARTIFACTS_DIR, 'jobs.pkl')
    job_vecs_path = os.path.join(ARTIFACTS_DIR, 'job_vectors.pkl')
    
    if not all(os.path.exists(p) for p in [tfidf_path, jobs_path, job_vecs_path]):
        print("❌ Error: Missing artifacts. Make sure you trained the model first!")
        return
        
    print(f"📂 Loading 100k+ Pre-trained Artifacts...")
    jobs_df = pd.read_pickle(jobs_path)
    tfidf = joblib.load(tfidf_path)
    job_vectors = joblib.load(job_vecs_path)
    bert_job_vectors = load_bert_vectors_smart(ARTIFACTS_DIR)
    
    print(f"   ✓ Jobs index loaded: {len(jobs_df)}")
    print(f"   ✓ TF-IDF vectors loaded: {job_vectors.shape}")
    if bert_job_vectors is not None:
        print(f"   ✓ BERT vectors loaded: {bert_job_vectors.shape}")
    else:
        print("   ⚠️ No BERT vectors found. Hybrid semantic search might fail!")

    print("-" * 80)
        
    # Generate the Ground Truth
    test_cases = generate_synthetic_test_cases(jobs_df, sample_size=sample_size)
    
    # Perform Analysis
    results_df = compare_models(test_cases, jobs_df, tfidf, job_vectors, bert_job_vectors, k=10)
    
    if len(results_df) == 0:
        print("❌ Evaluation failed. No results returned.")
        return
        
    # Standardize Means
    old_p10 = results_df["old_precision"].mean()
    new_p10 = results_df["new_precision"].mean()
    old_r10 = results_df["old_recall"].mean()
    new_r10 = results_df["new_recall"].mean()
    old_ndcg = results_df["old_ndcg"].mean()
    new_ndcg = results_df["new_ndcg"].mean()
    
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS (K=10 Top Matches)")
    print("="*80)
    print(f"{'Metric':<20} | {'Old TF-IDF Engine':<18} | {'NEW HYBRID ENGINE':<18} | {'Improvement'}")
    print("-" * 80)
    print(f"{'Precision@10':<20} | {old_p10*100:6.2f}%            | {new_p10*100:6.2f}%            | {((new_p10 - old_p10)/(old_p10 + 1e-8))*100:+.1f}%")
    print(f"{'Recall@10':<20} | {old_r10*100:6.2f}%            | {new_r10*100:6.2f}%            | {((new_r10 - old_r10)/(old_r10 + 1e-8))*100:+.1f}%")
    print(f"{'NDCG@10':<20} | {old_ndcg*100:6.2f}%            | {new_ndcg*100:6.2f}%            | {((new_ndcg - old_ndcg)/(old_ndcg + 1e-8))*100:+.1f}%")
    print("=" * 80)
    
    # Save granular reporting 
    results_df.to_csv("artifacts/evaluation_metrics.csv", index=False)
    print("\n✅ Saved detailed case-by-case metrics matrix to 'artifacts/evaluation_metrics.csv'\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    run_evaluation(sample_size=500)
