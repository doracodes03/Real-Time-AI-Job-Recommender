from src.preprocess import preprocess_text
import joblib
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.recommend import recommend_jobs
from src.llm_parser import parse_resume_with_llm
from src.llm_explainer import explain_match
from src.collaborative import save_interaction, get_cf_recommendations, get_hybrid_recommendations, get_saved_jobs
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =====================================================================
# Smart Artifact Loading (i5-optimized with chunked BERT support)
# =====================================================================

def load_bert_vectors_smart(artifacts_dir):
    """
    Smart BERT loading:
    1. Try single stacked file first (bert_job_vectors.pkl)
    2. Fall back to chunked files if not found
    3. Return None if neither exists
    """
    bert_vectors_path = os.path.join(artifacts_dir, 'bert_job_vectors.pkl')
    
    # Try single file first (most memory efficient)
    if os.path.exists(bert_vectors_path):
        print(f"  ✓ Loading single BERT vectors file...")
        return joblib.load(bert_vectors_path)
    
    # Fallback to chunks
    bert_chunks = sorted(glob.glob(os.path.join(artifacts_dir, 'bert_chunk_*.npy')))
    if bert_chunks:
        print(f"  ⚠️  Single BERT file not found. Loading {len(bert_chunks)} chunks...")
        try:
            bert_vectors = np.vstack([np.load(f) for f in bert_chunks])
            print(f"  ✓ Loaded BERT vectors from {len(bert_chunks)} chunks: shape {bert_vectors.shape}")
            return bert_vectors
        except Exception as e:
            print(f"  ✗ Failed to load BERT chunks: {e}")
            return None
    
    print(f"  ⚠️  No BERT vectors found (chunks or single file)")
    return None

# Load context globally from precomputed artifacts
ARTIFACTS_DIR = 'artifacts'
tfidf_path = os.path.join(ARTIFACTS_DIR, 'tfidf.pkl')
vectors_path = os.path.join(ARTIFACTS_DIR, 'job_vectors.pkl')
jobs_path = os.path.join(ARTIFACTS_DIR, 'jobs.pkl')

from contextlib import asynccontextmanager

# Global state for artifacts
state = {
    "tfidf": None,
    "job_vectors": None,
    "bert_job_vectors": None,
    "df": pd.DataFrame()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load context from precomputed artifacts
    print("\n" + "="*60)
    print("🔧 BACKEND: Loading Precomputed Artifacts...")
    print("="*60 + "\n")
    
    if os.path.exists(tfidf_path) and os.path.exists(vectors_path) and os.path.exists(jobs_path):
        print("✓ Artifacts directory found. Loading models...")
        try:
            state["tfidf"] = joblib.load(tfidf_path)
            print(f"  ✓ TF-IDF vectorizer: {len(state['tfidf'].get_feature_names_out())} features")
            
            state["job_vectors"] = joblib.load(vectors_path)
            print(f"  ✓ TF-IDF job vectors: {state['job_vectors'].shape} (sparse matrix)")
            
            state["bert_job_vectors"] = load_bert_vectors_smart(ARTIFACTS_DIR)
            
            state["df"] = pd.read_pickle(jobs_path)
            print(f"  ✓ Jobs metadata: {len(state['df'])} records")
            
            print(f"\n✅ All artifacts loaded successfully!")
        except Exception as e:
            print(f"  ❌ Error loading artifacts: {e}")
    else:
        print("❌ WARNING: Artifacts not found. Falling back to empty/dummy data.")
        print(f"   Expected: {tfidf_path}, {vectors_path}, {jobs_path}")
        state["df"] = pd.DataFrame(columns=['id', 'Job Title', 'Company', 'Location', 'skills', 'Job Description', 'Experience'])

    print(f"\n🚀 Backend Ready!")
    print(f"   - API docs: http://localhost:8000/docs\n")
    yield
    # Clean up if needed
    print("Stopping backend...")

app = FastAPI(title="Job Recommender API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": state["tfidf"] is not None and state["job_vectors"] is not None,
        "jobs_count": len(state["df"]) if state["df"] is not None else 0,
        "bert_available": state["bert_job_vectors"] is not None
    }

@app.post("/recommend/fast")
def recommend_fast(resume_text: str = Form(...)):
    """
    ⚡ FASTEST endpoint - TF-IDF only, no BERT embedding computation.
    Skips expensive resume embedding to return in <1 second on i5.
    """
    if state["tfidf"] is None or state["job_vectors"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")
    
    # ✅ ULTRA-FAST: TF-IDF only (skip BERT embedding computation)
    resume_data = {
        "text": preprocess_text(resume_text),
        "experience": 2,
        "skills": []  # Skip skills for speed
    }
    
    results = recommend_jobs(
        resume_data=resume_data,
        df=state["df"].copy(),
        tfidf=state["tfidf"],
        job_vectors=state["job_vectors"],
        bert_job_vectors=state["bert_job_vectors"],
        top_n=20,  # Get more to filter duplicates
        skip_bert_embedding=True,  # ⚡ AVOID SLOW BERT EMBEDDING COMPUTATION
        skip_skill_extraction=True  # ⚡ SKIP EXPENSIVE SKILL EXTRACTION
    )
    
    # ✅ AGGRESSIVE DUPLICATE REMOVAL
    if 'id' in results.columns:
        results = results.drop_duplicates(subset=['id'], keep='first')
    if 'Job Title' in results.columns and 'Company' in results.columns:
        results = results.drop_duplicates(subset=['Job Title', 'Company'], keep='first')
    if 'Location' in results.columns:
        results = results.drop_duplicates(subset=['Job Title', 'Company', 'Location'], keep='first')
    
    # Trim to exactly 10 unique results
    results = results.head(10)
    
    # Map columns
    if 'skills' not in results.columns and 'Skills' in results.columns:
        results['skills'] = results['Skills']
    
    out_cols = ['id', 'Job Title', 'Company', 'Location', 'skills', 'Job Description', 'final_score']
    available_cols = [c for c in out_cols if c in results.columns]
    
    recommendations = results[available_cols].to_dict(orient="records")
    
    # Final check: ensure all recommended items are unique by ID
    seen_ids = set()
    unique_recs = []
    for rec in recommendations:
        rec_id = rec.get('id', str(rec.get('Job Title', '')))
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            unique_recs.append(rec)
    
    return {"recommendations": unique_recs[:10]}

class InteractionReq(BaseModel):
    user_id: str
    job_id: str
    interaction_type: str

@app.post("/interaction")
def log_interaction(req: InteractionReq):
    if req.interaction_type not in ["click", "save", "apply"]:
        raise HTTPException(status_code=400, detail="Invalid interaction type")
    save_interaction(req.user_id, req.job_id, req.interaction_type)
    return {"message": "Success"}

@app.post("/recommend/content")
def recommend_content(resume_text: str = Form(...)):
    """
    Fast recommendation endpoint - optimized for low latency.
    Skips slow LLM parsing for initial results.
    """
    if state["tfidf"] is None or state["job_vectors"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")
    
    # ✅ FAST PATH: Skip LLM parsing, use simple extraction
    resume_data = {
        "text": preprocess_text(resume_text),
        "experience": 2  # Default
    }
    
    # Try quick LLM parsing (with timeout) - not critical if it fails
    try:
        parsed = parse_resume_with_llm(resume_text)
        if parsed and 'experience' in parsed:
            resume_data['experience'] = parsed.get('experience', 2)
        if parsed and 'skills' in parsed:
            resume_data['skills'] = parsed.get('skills', [])
    except Exception as e:
        print(f"  ⚠️  LLM parsing skipped (timeout): {str(e)}")
        # Continue without LLM - use TF-IDF + BERT only
    
    # Get recommendations (fast path with BERT if available)
    print(f"  Recommending jobs for resume ({len(resume_text)} chars)...")
    results = recommend_jobs(
        resume_data=resume_data,
        df=state["df"].copy(),
        tfidf=state["tfidf"],
        job_vectors=state["job_vectors"],
        bert_job_vectors=state["bert_job_vectors"],
        top_n=20  # Get more to filter duplicates
    )
    
    # ✅ AGGRESSIVE DUPLICATE REMOVAL
    if 'id' in results.columns:
        results = results.drop_duplicates(subset=['id'], keep='first')
    if 'Job Title' in results.columns and 'Company' in results.columns:
        results = results.drop_duplicates(subset=['Job Title', 'Company'], keep='first')
    if 'Location' in results.columns:
        results = results.drop_duplicates(subset=['Job Title', 'Company', 'Location'], keep='first')
    
    results = results.head(10)
    
    # Map columns for frontend
    if 'skills' not in results.columns and 'Skills' in results.columns:
        results['skills'] = results['Skills']
    
    out_cols = ['id', 'Job Title', 'Company', 'Location', 'skills', 'Job Description', 'final_score']
    available_cols = [c for c in out_cols if c in results.columns]
    
    recommendations = results[available_cols].to_dict(orient="records")
    
    # Final check: ensure all recommended items are unique by ID
    seen_ids = set()
    unique_recs = []
    for rec in recommendations:
        rec_id = rec.get('id', str(rec.get('Job Title', '')))
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            unique_recs.append(rec)
    
    return {"recommendations": unique_recs[:10], "resume_data": resume_data}

@app.get("/recommend/collaborative/{user_id}")
def recommend_collaborative(user_id: str):
    cf_res = get_cf_recommendations(user_id, state["df"], top_n=10)
    if cf_res.empty:
        return {"recommendations": []}
    return {"recommendations": cf_res.to_dict(orient="records")}

@app.post("/recommend/hybrid/{user_id}")
def recommend_hybrid(user_id: str, resume_text: str = Form(...)):
    if state["tfidf"] is None or state["job_vectors"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")
    
    # 1. Content
    resume_data = parse_resume_with_llm(resume_text)
    resume_data["text"] = preprocess_text(resume_text)
    content_res = recommend_jobs(
        resume_data=resume_data,
        df=state["df"].copy(),
        tfidf=state["tfidf"],
        job_vectors=state["job_vectors"],
        bert_job_vectors=state["bert_job_vectors"],
        top_n=20
    )
    
    # 2. CF
    cf_res = get_cf_recommendations(user_id, state["df"], top_n=20)
    
    # 3. Hybridize
    hybrid_res = get_hybrid_recommendations(user_id, content_res, cf_res, state["df"])
    
    # ✅ AGGRESSIVE DUPLICATE REMOVAL
    if 'id' in hybrid_res.columns:
        hybrid_res = hybrid_res.drop_duplicates(subset=['id'], keep='first')
    if 'Job Title' in hybrid_res.columns and 'Company' in hybrid_res.columns:
        hybrid_res = hybrid_res.drop_duplicates(subset=['Job Title', 'Company'], keep='first')
    if 'Location' in hybrid_res.columns:
        hybrid_res = hybrid_res.drop_duplicates(subset=['Job Title', 'Company', 'Location'], keep='first')
    
    hybrid_res = hybrid_res.head(10)
    
    # Map columns for frontend
    if 'skills' not in hybrid_res.columns and 'Skills' in hybrid_res.columns:
        hybrid_res['skills'] = hybrid_res['Skills']
    
    out_cols = ['id', 'Job Title', 'Company', 'Location', 'skills', 'Job Description', 'hybrid_score']
    available_cols = [c for c in out_cols if c in hybrid_res.columns]
    
    recommendations = hybrid_res[available_cols].to_dict(orient="records")
    
    # Final check: ensure all recommended items are unique by ID
    seen_ids = set()
    unique_recs = []
    for rec in recommendations:
        rec_id = rec.get('id', str(rec.get('Job Title', '')))
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            unique_recs.append(rec)
        
    return {"recommendations": unique_recs[:10]}

@app.post("/recommend/explain")
def explain_job(job_id: str = Form(...), resume_text: str = Form(...)):
    # 1. Find the job in our global dataframe
    job_row = state["df"][state["df"]['id'] == str(job_id)]
    if job_row.empty:
        raise HTTPException(status_code=404, detail="Job not found in current dataset")
    
    job = job_row.iloc[0]
    
    # 2. Parse resume to get structured data
    try:
        resume_data = parse_resume_with_llm(resume_text)
    except Exception as e:
        print(f"[Error] Failed to parse resume: {e}")
        resume_data = {"skills": [], "roles": [], "experience": 0, "education": "Unknown"}

    # 3. Generate explanation
    # Handle both 'skills' and 'Skills' column names (data uses lowercase 'skills')
    job_skills = job.get('skills') or job.get('Skills') or "Not specified"
    job_title = job.get('Job Title') or "Unknown Title"
    job_desc = job.get('Job Description') or "No description available"
    
    try:
        explanation = explain_match(
            resume_data=resume_data,
            job_title=job_title,
            job_desc=job_desc,
            job_skills=job_skills,
            ranker_score=0.85
        )
        return explanation
    except Exception as e:
        print(f"[Error] Failed to explain match: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/saved/{user_id}")
def view_saved_jobs(user_id: str):
    saved_res = get_saved_jobs(user_id, state["df"])
    if saved_res.empty:
        return {"recommendations": []}
    
    # Map columns for frontend
    if 'skills' not in saved_res.columns and 'Skills' in saved_res.columns:
        saved_res['skills'] = saved_res['Skills']
        
    return {"recommendations": saved_res.to_dict(orient="records")}

@app.post("/recommend/realtime/{user_id}")
def recommend_realtime(user_id: str, query: str = Form(...), location: str = Form(...)):
    """
    Real-time job search by keyword and location.
    Returns matching jobs sorted by relevance.
    """
    if state["tfidf"] is None or state["job_vectors"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")
    
    # Filter by location if available
    filtered_df = state["df"].copy()
    if location and location.lower() != "any":
        location_col = 'Location' if 'Location' in state["df"].columns else 'location'
        if location_col in state["df"].columns:
            filtered_df = filtered_df[
                filtered_df[location_col].str.contains(location, case=False, na=False)
            ]
    
    if filtered_df.empty:
        return {"recommendations": []}
    
    # Search by query (use TF-IDF for fast keyword matching)
    query_vec = state["tfidf"].transform([query])
    search_vectors = state["job_vectors"][:len(filtered_df)]
    
    similarities = cosine_similarity(query_vec, search_vectors)[0]
    
    filtered_df['search_score'] = similarities
    results = filtered_df.nlargest(20, 'search_score')  # Get more to filter duplicates
    
    # ✅ AGGRESSIVE DUPLICATE REMOVAL
    if 'id' in results.columns:
        results = results.drop_duplicates(subset=['id'], keep='first')
    if 'Job Title' in results.columns and 'Company' in results.columns:
        results = results.drop_duplicates(subset=['Job Title', 'Company'], keep='first')
    if 'Location' in results.columns:
        results = results.drop_duplicates(subset=['Job Title', 'Company', 'Location'], keep='first')
    
    results = results.head(10)
    
    # Map columns for frontend
    if 'skills' not in results.columns and 'Skills' in results.columns:
        results['skills'] = results['Skills']
    
    out_cols = ['id', 'Job Title', 'Company', 'Location', 'skills', 'Job Description', 'search_score']
    available_cols = [c for c in out_cols if c in results.columns]
    
    recommendations = results[available_cols].to_dict(orient="records")
    
    # Final check: ensure all recommended items are unique by ID
    seen_ids = set()
    unique_recs = []
    for rec in recommendations:
        rec_id = rec.get('id', str(rec.get('Job Title', '')))
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            unique_recs.append(rec)
    
    return {"recommendations": unique_recs[:10]}
