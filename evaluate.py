"""
Evaluation framework for job recommender system (i5-optimized):
- Compares old (TF-IDF+experience) vs new (hybrid) models
- Computes Precision@5, Recall@5, MRR
- Memory-efficient evaluation on limited i5 systems
- Handles chunked BERT artifacts properly
"""
import pandas as pd
import os
import glob
import numpy as np
import joblib
from src.recommend import recommend_jobs
from src.vectorize import get_bert_embeddings
from src.preprocess import extract_entities

# ⚡ COMPREHENSIVE TEST SET: Better quality for real evaluation
TEST_CASES = [
    {"resume": "python machine learning data science", "expected_roles": ["data scientist", "ml engineer", "ml specialist"]},
    {"resume": "react javascript frontend html css", "expected_roles": ["frontend engineer", "frontend developer", "ui developer"]},
    {"resume": "java spring boot backend", "expected_roles": ["backend developer", "java developer", "senior developer"]},
    {"resume": "docker kubernetes devops aws", "expected_roles": ["devops engineer", "infrastructure engineer", "sre"]},
    {"resume": "pytorch fastapi machine learning", "expected_roles": ["ml engineer", "data scientist", "ai engineer"]},
    {"resume": "sql database analytics", "expected_roles": ["data scientist", "analyst", "sql developer"]},
    {"resume": "python django flask backend", "expected_roles": ["backend developer", "python developer", "full stack"]},
    {"resume": "typescript react nextjs frontend", "expected_roles": ["frontend engineer", "web developer", "react developer"]},
    {"resume": "aws cloud architecture devops", "expected_roles": ["devops engineer", "cloud architect", "aws engineer"]},
    {"resume": "nlp natural language processing", "expected_roles": ["ml engineer", "nlp specialist", "ai engineer"]},
]

# 2. Metrics

def precision_at_k(recommended, expected, k=5):
    recommended = recommended[:k]
    hits = sum(1 for r in recommended if r in expected)
    return hits / k if k > 0 else 0.0

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
    # Minimal resume data - just text and experience
    resume_data = {
        "text": resume_text, 
        "experience": 2, 
        "skills": []  # No skills provided to disable skill matching
    }
    # Call recommend_jobs with bert disabled
    result_df = recommend_jobs(
        resume_data, 
        jobs_df.copy(), 
        tfidf, 
        job_vectors, 
        bert_job_vectors=None,  # This makes semantic_scores = tfidf_scores
        top_n=5
    )
    # Return top job titles
    if 'Job Title' in result_df.columns:
        return result_df["Job Title"].str.lower().tolist()
    return []

def new_recommender(resume_text, jobs_df, tfidf, job_vectors, bert_job_vectors):
    """
    Uses full hybrid model (semantic + tfidf + skill + experience)
    """
    resume_data = {
        "text": resume_text, 
        "experience": 2
    }
    result_df = recommend_jobs(
        resume_data, 
        jobs_df.copy(), 
        tfidf, 
        job_vectors, 
        bert_job_vectors=bert_job_vectors, 
        top_n=5
    )
    if 'Job Title' in result_df.columns:
        return result_df["Job Title"].str.lower().tolist()
    return []

# 4. Evaluation Functions

def evaluate_model(model_fn, test_cases, jobs_df, tfidf, job_vectors, bert_job_vectors=None):
    rows = []
    for case in test_cases:
        resume = case["resume"]
        expected = [r.lower() for r in case["expected_roles"]]
        
        try:
            if bert_job_vectors is not None:
                recommended = model_fn(resume, jobs_df, tfidf, job_vectors, bert_job_vectors)
            else:
                recommended = model_fn(resume, jobs_df, tfidf, job_vectors)
            
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
        except Exception as e:
            print(f"  ⚠️  Error evaluating '{resume}': {str(e)}")
            rows.append({
                "resume": resume,
                "expected_roles": expected,
                "recommended": [],
                "precision@5": 0.0,
                "recall@5": 0.0,
                "mrr@5": 0.0
            })
    
    return pd.DataFrame(rows)

def compare_models(old_model_fn, new_model_fn, test_cases, jobs_df, tfidf, job_vectors, bert_job_vectors):
    results = []
    for case in test_cases:
        resume = case["resume"]
        expected = [r.lower() for r in case["expected_roles"]]
        
        try:
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
        except Exception as e:
            print(f"  ⚠️  Error comparing models for '{resume}': {str(e)}")
            results.append({
                "resume": resume,
                "expected_roles": expected,
                "old_precision": 0.0,
                "new_precision": 0.0,
                "improvement": 0.0,
                "old_rec": [],
                "new_rec": []
            })
    
    return pd.DataFrame(results)

# 5. Artifact Loading (Memory-Optimized)

def load_bert_vectors_smart(artifacts_dir):
    """
    Smart BERT loading:
    1. Try single stacked file first (bert_job_vectors.pkl)
    2. Fall back to chunked files if not found
    3. Return None if neither exists
    """
    bert_vectors_path = os.path.join(artifacts_dir, 'bert_job_vectors.pkl')
    
    # Try single file first (most memory efficient for evaluation)
    if os.path.exists(bert_vectors_path):
        print(f"✓ Loading single BERT vectors file: {bert_vectors_path}")
        return joblib.load(bert_vectors_path)
    
    # Fallback to chunks
    bert_chunks = sorted(glob.glob(os.path.join(artifacts_dir, 'bert_chunk_*.npy')))
    if bert_chunks:
        print(f"⚠️  Single BERT file not found. Loading {len(bert_chunks)} chunks...")
        try:
            bert_vectors = np.vstack([np.load(f) for f in bert_chunks])
            print(f"✓ Loaded BERT vectors from {len(bert_chunks)} chunks: shape {bert_vectors.shape}")
            return bert_vectors
        except Exception as e:
            print(f"✗ Failed to load BERT chunks: {e}")
            return None
    
    print("⚠️  No BERT vectors found (chunks or single file)")
    return None

# 6. Dynamic BERT Loading (for CSV subset evaluation)

def load_bert_vectors_smart_dynamic(jobs_df):
    """
    Generate BERT vectors on-the-fly for the given jobs_df.
    Memory-safe: uses batch processing.
    """
    print("   Computing BERT embeddings (batch processing)...")
    from src.preprocess import preprocess_text
    
    # Prepare combined text (same logic as train.py)
    def get_combined_text_for_jobs(df):
        df = df.copy()
        df['Job Title'] = df['Job Title'].fillna('')
        skill_col = 'skills' if 'skills' in df.columns else 'Skills'
        df[skill_col] = df[skill_col].fillna('')
        df['Job Description'] = df['Job Description'].fillna('')
        df['Job Description'] = df['Job Description'].str[:500]  # Truncate for memory
        
        return (
            df['Job Title'] + " " +
            df[skill_col] + " " +
            df['Job Description']
        ).apply(preprocess_text)
    
    combined_texts = get_combined_text_for_jobs(jobs_df).tolist()
    
    # Generate embeddings with batch processing
    print(f"   Processing {len(combined_texts)} jobs in batches of 32...")
    try:
        # get_bert_embeddings handles batching internally with batch_size parameter
        bert_vectors = get_bert_embeddings(combined_texts)
        print(f"   ✓ Generated embeddings: {bert_vectors.shape}")
        return bert_vectors
    except Exception as e:
        print(f"   ✗ BERT embedding failed: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return None


# 7. Main Evaluation Pipeline

def run_evaluation(jobs_df=None, use_csv_subset=True, subset_size=2000):
    """
    Evaluate recommender on real data subset from original CSV.
    This allows comparison of TF-IDF vs Hybrid on actual job data.
    
    Args:
        use_csv_subset: If True, loads from original CSV; if False, uses pre-artifacts
        subset_size: Number of jobs to evaluate on (i5-safe)
    """
    
    ARTIFACTS_DIR = 'artifacts'
    DATA_PATH = 'data/job_descriptions.csv'
    
    # Load precomputed artifacts (vectorizers)
    tfidf_path = os.path.join(ARTIFACTS_DIR, 'tfidf.pkl')
    
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(f"Run train.py first. Missing: {tfidf_path}")
    
    print("\n" + "=" * 70)
    if use_csv_subset:
        print("📊 EVALUATION: Real Dataset Subset (from job_descriptions.csv)")
    else:
        print("📊 EVALUATION: Pre-trained Artifacts")
    print("=" * 70)
    
    # Step 1: Load original CSV and sample
    if use_csv_subset:
        print(f"\n📂 Loading original dataset from {DATA_PATH}...")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Original dataset not found: {DATA_PATH}")
        
        all_jobs = pd.read_csv(DATA_PATH)
        print(f"   Total available: {len(all_jobs)} jobs")
        
        # Sample subset for i5-safety
        if len(all_jobs) > subset_size:
            print(f"   Sampling {subset_size} jobs for evaluation (i5-safe)...")
            jobs_df = all_jobs.sample(n=subset_size, random_state=42).reset_index(drop=True)
        else:
            jobs_df = all_jobs.reset_index(drop=True)
        
        print(f"   ✓ Using {len(jobs_df)} jobs for evaluation")
    else:
        # Fall back to pre-trained
        print(f"\n📂 Loading pre-trained artifacts...")
        jobs_path = os.path.join(ARTIFACTS_DIR, 'jobs.pkl')
        if not os.path.exists(jobs_path):
            raise FileNotFoundError(f"Jobs artifact not found: {jobs_path}")
        jobs_df = pd.read_pickle(jobs_path)
        print(f"   ✓ Loaded {len(jobs_df)} jobs from artifacts")
    
    # Step 2: Load pre-trained TF-IDF vectorizer
    print(f"\n🔧 Loading pre-trained TF-IDF vectorizer...")
    tfidf = joblib.load(tfidf_path)
    print(f"   ✓ Features: {len(tfidf.get_feature_names_out())}")
    
    # Step 3: Vectorize jobs using pre-trained vectorizer
    print(f"\n⚙️  Vectorizing jobs with TF-IDF...")
    
    def get_combined_text_for_jobs(df):
        """Combine job features for vectorization."""
        df = df.copy()
        df['Job Title'] = df['Job Title'].fillna('')
        skill_col = 'skills' if 'skills' in df.columns else 'Skills'
        df[skill_col] = df[skill_col].fillna('')
        df['Job Description'] = df['Job Description'].fillna('')
        df['Job Description'] = df['Job Description'].str[:500]  # Truncate for memory
        
        from src.preprocess import preprocess_text
        return (
            df['Job Title'] + " " +
            df[skill_col] + " " +
            df['Job Description']
        ).apply(preprocess_text)
    
    combined_texts = get_combined_text_for_jobs(jobs_df)
    job_vectors = tfidf.transform(combined_texts)
    print(f"   ✓ TF-IDF vectors: {job_vectors.shape}")
    
    # Step 4: Generate or load BERT vectors
    print(f"\n🧠 Loading BERT vectors...")
    if use_csv_subset:
        bert_job_vectors = load_bert_vectors_smart_dynamic(jobs_df)
    else:
        bert_job_vectors = load_bert_vectors_smart(ARTIFACTS_DIR)
    
    if bert_job_vectors is not None:
        print(f"   ✓ BERT vectors: {bert_job_vectors.shape}")
    else:
        print(f"   ⚠️  BERT vectors not available")
    
    # Step 5: Run comparison
    print(f"\n" + "=" * 70)
    print("🔬 Model Comparison: Old (TF-IDF only) vs New (Hybrid)")
    print("=" * 70 + "\n")
    
    results_df = compare_models(
        old_recommender, 
        new_recommender, 
        TEST_CASES, 
        jobs_df, 
        tfidf, 
        job_vectors, 
        bert_job_vectors
    )
    
    # Step 6: Output results
    if len(results_df) > 0:
        old_avg = results_df["old_precision"].mean()
        new_avg = results_df["new_precision"].mean()
        improvement_pct = (new_avg - old_avg) / (old_avg + 1e-8) * 100 if old_avg > 0 else 0
        
        print(f"📈 Performance Summary:")
        print(f"   ├─ Old Model (TF-IDF) Avg Precision@5:  {old_avg:.3f}")
        print(f"   ├─ New Model (Hybrid) Avg Precision@5:  {new_avg:.3f}")
        print(f"   └─ Relative Improvement:                {improvement_pct:+.1f}%\n")
        
        # Show detailed results
        print("📋 Detailed Results:")
        print("-" * 70)
        for idx, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"{idx}. Resume: '{row['resume']}'")
            print(f"   Old P@5: {row['old_precision']:.3f} | New P@5: {row['new_precision']:.3f} | Gain: {row['improvement']:+.3f}")
        
        # Show top improvements
        print(f"\n✨ Top 3 Improvements (Hybrid > TF-IDF):")
        print("-" * 70)
        top_cases = results_df.nlargest(3, "improvement")
        for idx, (_, row) in enumerate(top_cases.iterrows(), 1):
            if row["improvement"] >= 0:
                print(f"\n{idx}. Query: '{row['resume']}'")
                print(f"   Expected: {', '.join(row['expected_roles'])}")
                print(f"   Old (TF-IDF): {row['old_rec'][:3]}")
                print(f"   New (Hybrid): {row['new_rec'][:3]}")
                print(f"   Improvement: {row['improvement']:+.3f}")
    else:
        print("⚠️  No evaluation results. Check data and dependencies.")
    
    return results_df


if __name__ == "__main__":
    try:
        # Use CSV subset for better evaluation (shows real differences)
        run_evaluation(use_csv_subset=True, subset_size=2000)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
