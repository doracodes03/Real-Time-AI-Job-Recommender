from src.preprocess import preprocess_text
import joblib
import os
import pandas as pd
import requests
import numpy as np
from src.recommend import recommend_jobs
from src.llm_parser import parse_resume_with_llm
from src.llm_explainer import explain_match
from src.collaborative import save_interaction, get_cf_recommendations, get_hybrid_recommendations, get_saved_jobs
from src.vectorize import get_bert_embeddings, get_similarity
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load context globally from precomputed artifacts
ARTIFACTS_DIR = 'artifacts'
tfidf_path = os.path.join(ARTIFACTS_DIR, 'tfidf.pkl')
vectors_path = os.path.join(ARTIFACTS_DIR, 'job_vectors.pkl')
jobs_path = os.path.join(ARTIFACTS_DIR, 'jobs.pkl')
bert_vectors_path = os.path.join(ARTIFACTS_DIR, 'bert_job_vectors.pkl')

if os.path.exists(tfidf_path) and os.path.exists(vectors_path) and os.path.exists(jobs_path):
    print("Loading precomputed artifacts...")
    tfidf = joblib.load(tfidf_path)
    job_vectors = joblib.load(vectors_path)
    bert_job_vectors = joblib.load(bert_vectors_path) if os.path.exists(bert_vectors_path) else None
    df = pd.read_pickle(jobs_path)
    print(f"Loaded {len(df)} jobs successfully.")
else:
    print("WARNING: Artifacts not found. Falling back to empty/dummy data.")
    df = pd.DataFrame(columns=['id', 'Job Title', 'Company', 'Location', 'Skills', 'Job Description', 'Experience'])
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
    resume_data = parse_resume_with_llm(resume_text)
    resume_data["text"] = preprocess_text(resume_text)
    
    results = recommend_jobs(
        resume_data=resume_data,
        df=df,
        tfidf=tfidf,
        job_vectors=job_vectors,
        bert_job_vectors=bert_job_vectors,
        top_n=10
    )
    # Ensure id is included
    out_cols = ['id', 'Job Title', 'Company', 'Location', 'Skills', 'Job Description', 'final_score']
    # Add mapping for frontend if it expects 'Company Name'
    if 'Company' in results.columns:
        results['Company Name'] = results['Company']
        out_cols.append('Company Name')
        
    available_cols = [c for c in out_cols if c in results.columns]
    
    return {"recommendations": results[available_cols].to_dict(orient="records"), "resume_data": resume_data}

@app.get("/recommend/collaborative/{user_id}")
def recommend_collaborative(user_id: str):
    cf_res = get_cf_recommendations(user_id, df, top_n=10)
    if cf_res.empty:
        return {"recommendations": []}
    return {"recommendations": cf_res.to_dict(orient="records")}

@app.post("/recommend/hybrid/{user_id}")
def recommend_hybrid(user_id: str, resume_text: str = Form(...)):
    # 1. Content
    resume_data = parse_resume_with_llm(resume_text)
    resume_data["text"] = preprocess_text(resume_text)
    content_res = recommend_jobs(
        resume_data=resume_data,
        df=df,
        tfidf=tfidf,
        job_vectors=job_vectors,
        bert_job_vectors=bert_job_vectors,
        top_n=10
    )
    
    # 2. CF
    cf_res = get_cf_recommendations(user_id, df, top_n=10)
    
    # 3. Hybridize
    hybrid_res = get_hybrid_recommendations(user_id, content_res, cf_res, df)
    
    out_cols = ['id', 'Job Title', 'Company', 'Location', 'Skills', 'Job Description', 'hybrid_score']
    if 'Company' in hybrid_res.columns:
        hybrid_res['Company Name'] = hybrid_res['Company']
        out_cols.append('Company Name')
        
    available_cols = [c for c in out_cols if c in hybrid_res.columns]
    
    return {"recommendations": hybrid_res[available_cols].to_dict(orient="records")}

@app.post("/recommend/explain")
def explain_job(job_id: str = Form(...), resume_text: str = Form(...)):
    # 1. Find the job in our global dataframe
    job_row = df[df['id'] == str(job_id)]
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
    # Handle both 'Skills' and 'skills' column names
    job_skills = job.get('Skills') or job.get('skills') or "Not specified"
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
    saved_res = get_saved_jobs(user_id, df)
    if saved_res.empty:
        return {"recommendations": []}
    
    # Ensure Company Name is present for frontend
    if 'Company' in saved_res.columns:
        saved_res['Company Name'] = saved_res['Company']
        
    return {"recommendations": saved_res.to_dict(orient="records")}

def fetch_realtime_jobs(query: str, location: str):
    """Fetch real-time jobs from JSearch API (RapidAPI)"""
    api_key = os.getenv('JSEARCH_API_KEY')
    
    if not api_key:
        raise HTTPException(status_code=400, detail="JSEARCH_API_KEY environment variable not set. Get free key from https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch")
    
    url = "https://jsearch.p.rapidapi.com/search"
    
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "jsearch.p.rapidapi.com",
        "Content-Type": "application/json"
    }
    
    params = {
        "query": f"{query} jobs in {location}",
        "page": "1",
        "num_pages": "1",
        "country": "us",
        "date_posted": "all"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        
        jobs_data = response.json()
        jobs = jobs_data.get('data', [])
        
        if not jobs:
            return []
        
        job_list = []
        for job in jobs:
            job_list.append({
                'id': job.get('job_id', 'unknown'),
                'Job Title': job.get('job_title', ''),
                'Job Description': job.get('job_description', ''),
                'Skills': '',
                'Company': job.get('employer_name', ''),
                'Location': job.get('job_location', location),
                'apply_url': job.get('job_apply_link', ''),
            })
        
        return job_list
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid JSEARCH_API_KEY")
        elif e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        raise HTTPException(status_code=500, detail=f"API Error: {e.response.reason}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch jobs: {str(e)}")

@app.post("/recommend/realtime/{user_id}")
def recommend_realtime(user_id: str, resume_text: str = Form(...), query: str = Form(...), location: str = Form(...)):
    """Fetch and score real-time jobs from JSearch API"""
    try:
        # 1. Fetch real-time jobs from API
        realtime_jobs = fetch_realtime_jobs(query, location)
        
        if not realtime_jobs:
            return {"recommendations": [], "source": "realtime"}
        
        # 2. Parse resume
        resume_data = parse_resume_with_llm(resume_text)
        resume_data["text"] = preprocess_text(resume_text)
        
        # 3. Convert to DataFrame for compatibility
        jobs_df = pd.DataFrame(realtime_jobs)
        
        # 4. Score using BERT embeddings (semantic similarity)
        if bert_job_vectors is not None and len(jobs_df) > 0:
            # Combine job info for better matching
            jobs_df['combined_text'] = (
                jobs_df['Job Title'] + " " + 
                jobs_df['Job Description'] + " " + 
                jobs_df.get('Skills', '')
            )
            
            # Preprocess and vectorize
            job_texts = [preprocess_text(text) for text in jobs_df['combined_text']]
            job_vectors_realtime = get_bert_embeddings(job_texts)
            
            resume_vec = get_bert_embeddings([resume_data["text"]])[0]
            
            # Calculate similarity scores
            scores = get_similarity(resume_vec.reshape(1, -1), job_vectors_realtime)
            jobs_df['final_score'] = scores
        else:
            # Fallback: simple TF-IDF matching
            jobs_df['final_score'] = 0.5
        
        # 5. Sort by score and return top 10
        results = jobs_df.nlargest(10, 'final_score')
        results['Company Name'] = results['Company']
        
        out_cols = ['id', 'Job Title', 'Company', 'Company Name', 'Location', 'Skills', 'Job Description', 'final_score', 'apply_url']
        available_cols = [c for c in out_cols if c in results.columns]
        
        return {
            "recommendations": results[available_cols].to_dict(orient="records"),
            "source": "realtime",
            "count": len(results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Real-time recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
