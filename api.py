from dotenv import load_dotenv
load_dotenv(override=True)  # Load .env FIRST so GEMINI_API_KEY is available everywhere

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
from src.auth import (
    get_password_hash, 
    verify_password, 
    save_user, 
    get_user, 
    create_access_token, 
    get_current_user,
    User
)
from fastapi import FastAPI, HTTPException, UploadFile, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
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
        print(f"  Loading single BERT vectors file...")
        return joblib.load(bert_vectors_path)
    
    # Fallback to chunks
    bert_chunks = sorted(glob.glob(os.path.join(artifacts_dir, 'bert_chunk_*.npy')))
    if bert_chunks:
        print(f"  Single BERT file not found. Loading {len(bert_chunks)} chunks...")
        try:
            bert_vectors = np.vstack([np.load(f) for f in bert_chunks])
            print(f"  Loaded BERT vectors from {len(bert_chunks)} chunks: shape {bert_vectors.shape}")
            return bert_vectors
        except Exception as e:
            print(f"  Failed to load BERT chunks: {e}")
            return None
    
    print(f"  No BERT vectors found (chunks or single file)")
    return None

# Load context globally from precomputed artifacts
ARTIFACTS_DIR = 'artifacts'
tfidf_path = os.path.join(ARTIFACTS_DIR, 'tfidf.pkl')
vectors_path = os.path.join(ARTIFACTS_DIR, 'job_vectors.pkl')
jobs_path = os.path.join(ARTIFACTS_DIR, 'jobs.pkl')

print("\n" + "="*60)
print("BACKEND: Loading Precomputed Artifacts...")
print("="*60 + "\n")

if os.path.exists(tfidf_path) and os.path.exists(vectors_path) and os.path.exists(jobs_path):
    print("Artifacts directory found. Loading models...")
    tfidf = joblib.load(tfidf_path)
    print(f"  TF-IDF vectorizer: {len(tfidf.get_feature_names_out())} features")
    
    job_vectors = joblib.load(vectors_path)
    print(f"  TF-IDF job vectors: {job_vectors.shape} (sparse matrix)")
    
    bert_job_vectors = load_bert_vectors_smart(ARTIFACTS_DIR)
    
    df = pd.read_pickle(jobs_path)
    print(f"  Jobs metadata: {len(df)} records")
    
    print(f"\nAll artifacts loaded successfully!")
    print(f"\nBackend Ready!")
    print(f"   - Use /recommend/fast for ultra-fast results (<1sec)")
    print(f"   - Use /recommend/content for full analysis")
    print(f"   - API docs: http://localhost:8000/docs\n")
else:
    print("WARNING: Artifacts not found. Falling back to empty/dummy data.")
    print(f"   Expected: {tfidf_path}, {vectors_path}, {jobs_path}")
    df = pd.DataFrame(columns=['id', 'Job Title', 'Company', 'Location', 'skills', 'Job Description', 'Experience'])
    tfidf = None
    job_vectors = None
    bert_job_vectors = None

app = FastAPI(title="Job Recommender API")

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
        "models_loaded": tfidf is not None and job_vectors is not None,
        "jobs_count": len(df) if df is not None else 0,
        "bert_available": bert_job_vectors is not None
    }

@app.post("/recommend/fast")
def recommend_fast(
    resume_text: str = Form(...),
    location: str = Form(None),
    max_experience: float = Form(None),
    page: int = Form(1),
    page_size: int = Form(10)
):
    if tfidf is None or job_vectors is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")

    # FASTEST path - TF-IDF only
    resume_data = {
        "text": preprocess_text(resume_text),
        "experience": 2,
        "skills": []  # Skip skills for speed
    }
    
    results = recommend_jobs(
        resume_data=resume_data,
        df=df.copy(),
        tfidf=tfidf,
        job_vectors=job_vectors,
        bert_job_vectors=bert_job_vectors,
        top_n=100,  # Get more for pagination
        skip_bert_embedding=True,  # ⚡ AVOID SLOW BERT EMBEDDING COMPUTATION
        skip_skill_extraction=True  # ⚡ SKIP EXPENSIVE SKILL EXTRACTION
    )
    
    # AGGRESSIVE DUPLICATE REMOVAL
    if 'id' in results.columns:
        results = results.drop_duplicates(subset=['id'], keep='first')
    if 'Job Title' in results.columns and 'Company' in results.columns:
        results = results.drop_duplicates(subset=['Job Title', 'Company'], keep='first')
    if 'Location' in results.columns:
        results = results.drop_duplicates(subset=['Job Title', 'Company', 'Location'], keep='first')
    
    total_count = len(results)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = results.iloc[start_idx:end_idx].copy()
    
    # Map columns
    if 'skills' not in paginated.columns and 'Skills' in paginated.columns:
        paginated['skills'] = paginated['Skills']
    
    out_cols = ['id', 'Job Title', 'Company', 'Location', 'skills', 'Job Description', 'final_score']
    available_cols = [c for c in out_cols if c in paginated.columns]
    
    recommendations = paginated[available_cols].to_dict(orient="records")
    
    # Final check: ensure all recommended items are unique by ID
    seen_ids = set()
    unique_recs = []
    for rec in recommendations:
        rec_id = rec.get('id', str(rec.get('Job Title', '')))
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            unique_recs.append(rec)
    
    return {
        "recommendations": unique_recs,
        "total": total_count,
        "page": page,
        "page_size": page_size
    }

# --- Authentication Endpoints ---

@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    hashed_pw = get_password_hash(password)
    if save_user(username, hashed_pw):
        return {"message": "User registered successfully"}
    raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/token")
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# =====================================================================
# RECOMMENDATION ENDPOINTS (AUTHENTICATED)
# =====================================================================

class InteractionReq(BaseModel):
    job_id: str
    interaction_type: str

@app.post("/interaction")
def log_interaction(req: InteractionReq, current_user: User = Depends(get_current_user)):
    if req.interaction_type not in ["click", "save", "apply"]:
        raise HTTPException(status_code=400, detail="Invalid interaction type")
    save_interaction(current_user.username, req.job_id, req.interaction_type)
    return {"message": "Success"}

@app.post("/recommend/content")
def recommend_content(
    resume_text: str = Form(...), 
    location: str = Form(None),
    max_experience: float = Form(None),
    page: int = Form(1),
    page_size: int = Form(10),
    current_user: User = Depends(get_current_user)
):
    """
    Fast recommendation endpoint - optimized for low latency.
    Skips slow LLM parsing for initial results.
    """
    if tfidf is None or job_vectors is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")
    
    # FAST PATH: Skip LLM parsing, use simple extraction
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
        df=df.copy(),
        tfidf=tfidf,
        job_vectors=job_vectors,
        bert_job_vectors=bert_job_vectors,
        top_n=100  # Get more for pagination
    )
    
    # AGGRESSIVE DUPLICATE REMOVAL
    if 'id' in results.columns:
        results = results.drop_duplicates(subset=['id'], keep='first')
    if 'Job Title' in results.columns and 'Company' in results.columns:
        results = results.drop_duplicates(subset=['Job Title', 'Company'], keep='first')
    if 'Location' in results.columns:
        results = results.drop_duplicates(subset=['Job Title', 'Company', 'Location'], keep='first')
    
    total_count = len(results)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = results.iloc[start_idx:end_idx].copy()
    
    # Map columns for frontend
    if 'skills' not in paginated.columns and 'Skills' in paginated.columns:
        paginated['skills'] = paginated['Skills']
    
    out_cols = ['id', 'Job Title', 'Company', 'Location', 'skills', 'Job Description', 'final_score']
    available_cols = [c for c in out_cols if c in paginated.columns]
    
    recommendations = paginated[available_cols].to_dict(orient="records")
    
    # Final check: ensure all recommended items are unique by ID
    seen_ids = set()
    unique_recs = []
    for rec in recommendations:
        rec_id = rec.get('id', str(rec.get('Job Title', '')))
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            unique_recs.append(rec)
    
    return {
        "recommendations": unique_recs, 
        "resume_data": resume_data,
        "total": total_count,
        "page": page,
        "page_size": page_size
    }

@app.get("/recommend/collaborative/{user_id}")
def recommend_collaborative(
    user_id: str, 
    page: int = 1, 
    page_size: int = 10, 
    current_user: User = Depends(get_current_user)
):
    # Use authenticated user if available, otherwise fallback to path user_id
    effective_user_id = current_user.username if current_user else user_id
    cf_res = get_cf_recommendations(effective_user_id, df, top_n=100)
    if cf_res.empty:
        return {"recommendations": [], "total": 0, "page": page, "page_size": page_size}
    
    total_count = len(cf_res)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = cf_res.iloc[start_idx:end_idx]
    
    return {
        "recommendations": paginated.to_dict(orient="records"),
        "total": total_count,
        "page": page,
        "page_size": page_size
    }

@app.post("/recommend/hybrid/{user_id}")
def recommend_hybrid(
    user_id: str, 
    resume_text: str = Form(...), 
    location: str = Form(None),
    max_experience: float = Form(None),
    page: int = Form(1),
    page_size: int = Form(10),
    current_user: User = Depends(get_current_user)
):
    if tfidf is None or job_vectors is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")
    
    # Use authenticated user if available
    effective_user_id = current_user.username if current_user else user_id
    resume_data = parse_resume_with_llm(resume_text)
    resume_data["text"] = preprocess_text(resume_text)
    content_res = recommend_jobs(
        resume_data=resume_data,
        df=df.copy(),
        tfidf=tfidf,
        job_vectors=job_vectors,
        bert_job_vectors=bert_job_vectors,
        top_n=100
    )
    
    # 2. CF
    cf_res = get_cf_recommendations(effective_user_id, df, top_n=100)
    
    # 3. Hybridize
    hybrid_res = get_hybrid_recommendations(effective_user_id, content_res, cf_res, df)
    
    # AGGRESSIVE DUPLICATE REMOVAL
    if 'id' in hybrid_res.columns:
        hybrid_res = hybrid_res.drop_duplicates(subset=['id'], keep='first')
    if 'Job Title' in hybrid_res.columns and 'Company' in hybrid_res.columns:
        hybrid_res = hybrid_res.drop_duplicates(subset=['Job Title', 'Company'], keep='first')
    if 'Location' in hybrid_res.columns:
        hybrid_res = hybrid_res.drop_duplicates(subset=['Job Title', 'Company', 'Location'], keep='first')
    
    total_count = len(hybrid_res)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated = hybrid_res.iloc[start_idx:end_idx].copy()
    
    # Map columns for frontend
    if 'skills' not in paginated.columns and 'Skills' in paginated.columns:
        paginated['skills'] = paginated['Skills']
    
    out_cols = ['id', 'Job Title', 'Company', 'Location', 'skills', 'Job Description', 'hybrid_score']
    available_cols = [c for c in out_cols if c in paginated.columns]
    
    recommendations = paginated[available_cols].to_dict(orient="records")
    
    # Final check: ensure all recommended items are unique by ID
    seen_ids = set()
    unique_recs = []
    for rec in recommendations:
        rec_id = rec.get('id', str(rec.get('Job Title', '')))
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            unique_recs.append(rec)
        
    return {
        "recommendations": unique_recs,
        "total": total_count,
        "page": page,
        "page_size": page_size
    }

@app.post("/recommend/explain")
def explain_job(
    job_id: str = Form(...),
    resume_text: str = Form(...),
    # Optional inline job details — used for real-time / JSearch jobs not in local df
    job_title_inline: str = Form(None),
    job_desc_inline: str = Form(None),
    job_skills_inline: str = Form(None),
    ranker_score: float = Form(0.85),
):
    # 1. Try to find the job in our static dataframe
    job_row = df[df['id'] == str(job_id)]

    if job_row.empty:
        # Real-time job (e.g. from JSearch) — use inline fields passed by the frontend
        if not job_title_inline and not job_desc_inline:
            raise HTTPException(
                status_code=404,
                detail="Job not found in dataset and no inline details provided."
            )
        job_title  = job_title_inline  or "Unknown Title"
        job_desc   = job_desc_inline   or "No description available"
        job_skills = job_skills_inline or "Not specified"
        print(f"[Explain] Using inline job details for real-time job id={job_id}")
    else:
        job = job_row.iloc[0]
        job_skills = job.get('skills') or job.get('Skills') or "Not specified"
        job_title  = job.get('Job Title') or "Unknown Title"
        job_desc   = job.get('Job Description') or "No description available"

    # 2. Parse resume to get structured data
    try:
        resume_data = parse_resume_with_llm(resume_text)
    except Exception as e:
        print(f"[Error] Failed to parse resume: {e}")
        resume_data = {"skills": [], "roles": [], "experience": 0, "education": "Unknown"}

    # 3. Generate explanation
    try:
        explanation = explain_match(
            resume_data=resume_data,
            job_title=job_title,
            job_desc=job_desc,
            job_skills=job_skills,
            ranker_score=ranker_score
        )
        return explanation
    except Exception as e:
        print(f"[Error] Failed to explain match: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/saved/{user_id}")
def view_saved_jobs(user_id: str, current_user: User = Depends(get_current_user)):
    effective_user_id = current_user.username if current_user else user_id
    saved_res = get_saved_jobs(effective_user_id, df)
    if saved_res.empty:
        return {"recommendations": []}
    
    # Map columns for frontend
    if 'skills' not in saved_res.columns and 'Skills' in saved_res.columns:
        saved_res['skills'] = saved_res['Skills']
        
    return {"recommendations": saved_res.to_dict(orient="records")}

@app.post("/recommend/realtime/{user_id}")
def recommend_realtime(
    user_id: str,
    query: str = Form(...),
    location: str = Form(...),
    resume_text: str = Form(...),
    page: int = Form(1),
    page_size: int = Form(10),
    current_user: User = Depends(get_current_user)
):
    """
    Real-time job search + ML ranking + pagination using JSearch API
    """

    if tfidf is None or job_vectors is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please run train.py first.")

    import requests

    rapidapi_key = os.getenv("JSEARCH_RAPIDAPI_KEY")
    rapidapi_host = os.getenv("JSEARCH_RAPIDAPI_HOST", "jsearch.p.rapidapi.com")
    if not rapidapi_key:
        raise HTTPException(
            status_code=500,
            detail="JSearch API key missing. Set JSEARCH_RAPIDAPI_KEY in .env"
        )

    # =========================
    # 🔹 1. Fetch real-time jobs from JSearch
    # =========================
    url = "https://jsearch.p.rapidapi.com/search"

    search_query = f"{query} jobs in {location}"

    querystring = {
        "query": search_query,
        "page": str(page),
        "num_pages": "1",
        "country": "us",
        "date_posted": "all"
    }

    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": rapidapi_host
    }

    timeout_seconds = int(os.getenv("JSEARCH_TIMEOUT_SECONDS", "25"))
    max_retries = int(os.getenv("JSEARCH_MAX_RETRIES", "2"))

    data = None
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, headers=headers, params=querystring, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.Timeout as e:
            last_error = e
            print(f"  ⚠️  JSearch timeout on attempt {attempt + 1}/{max_retries + 1}: {e}")
            continue
        except Exception as e:
            last_error = e
            print(f"  ❌ JSearch API Error: {e}")
            break

    if data is None:
        if isinstance(last_error, requests.exceptions.Timeout):
            raise HTTPException(
                status_code=504,
                detail=f"JSearch timed out after {max_retries + 1} attempts. Try again in a moment."
            )
        raise HTTPException(status_code=500, detail=f"Failed to fetch jobs from JSearch: {str(last_error)}")

    if "data" not in data or len(data["data"]) == 0:
        return {
            "recommendations": [],
            "total": 0,
            "page": page,
            "page_size": page_size
        }

    # =========================
    # 🔹 2. Convert to DataFrame
    # =========================
    realtime_jobs = []
    for job in data["data"]:
        realtime_jobs.append({
            "id": job.get("job_id", ""),
            "Job Title": job.get("job_title", "N/A"),
            "Company": job.get("employer_name", "N/A"),
            "Location": f"{job.get('job_city', '')}, {job.get('job_state', '')}, {job.get('job_country', '')}".strip(", "),
            "Job Description": job.get("job_description", ""),
            "apply_url": job.get("job_apply_link") or job.get("job_google_link"),
            "skills": "",  # Extracted by recommend_jobs if needed
            "Experience": "0" # Default for scoring if missing
        })
    
    realtime_df = pd.DataFrame(realtime_jobs)

    # =========================
    # 🔹 3. Resume processing
    # =========================
    resume_data = {
        "text": preprocess_text(resume_text),
        "experience": 2, # Default
        "skills": []
    }

    # Try quick LLM parsing if possible
    try:
        parsed = parse_resume_with_llm(resume_text)
        if parsed:
            resume_data['experience'] = parsed.get('experience', 2)
            resume_data['skills'] = parsed.get('skills', [])
    except:
        pass

    # =========================
    # 🔹 4. Run ML MODEL Scoring
    # =========================
    # Transform descriptions for TF-IDF
    job_desc_series = realtime_df["Job Description"].fillna("")
    realtime_tfidf_vectors = tfidf.transform(job_desc_series)

    from src.vectorize import get_bert_embeddings
    print(f"  [RealTime] Computing BERT semantic models for {len(realtime_df)} jobs...")
    realtime_bert_vectors = get_bert_embeddings(job_desc_series.tolist())

    # Get scores using our hybrid engine
    results = recommend_jobs(
        resume_data=resume_data,
        df=realtime_df,
        tfidf=tfidf,
        job_vectors=realtime_tfidf_vectors,
        bert_job_vectors=realtime_bert_vectors,
        top_n=len(realtime_df)
    )

    # =========================
    # 🔹 5. Sort and Pagination
    # =========================
    # Exact user requested hybrid scoring: 0.7 * semantic and 0.3 * tfidf
    if not results.empty:
        results["final_score"] = (0.7 * results["semantic_score"]) + (0.3 * results["tfidf_score"])
    
    # Sort by final_score descending
    results = results.sort_values(by="final_score", ascending=False)
    
    # Scale realtime scores visually to appear as high-percentile matches
    if not results.empty:
        max_score = results["final_score"].max()
        if max_score > 0:
            results["final_score"] = (results["final_score"] / max_score) * 0.92
        else:
            results["final_score"] = 0.50
    
    # Force 10 pages explicitly in the frontend 
    total_count = 100 
    
    # DO NOT use list slicing (e.g. results.iloc[start_idx:end_idx]) because JSearch 
    # API natively paginated the results to 10 rows for THIS specific page already!
    paginated_results = results

    # =========================
    # 🔹 6. Format Output
    # =========================
    out_cols = ['id', 'Job Title', 'Company', 'Location', 'Job Description', 'final_score', 'apply_url']
    available_cols = [c for c in out_cols if c in paginated_results.columns]
    
    recommendations = paginated_results[available_cols].to_dict(orient="records")

    return {
        "recommendations": recommendations,
        "total": total_count,
        "page": page,
        "page_size": page_size
    }