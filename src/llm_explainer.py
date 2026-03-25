# src/llm_explainer.py

import os
from pydantic import BaseModel
from google import genai

class JobExplanation(BaseModel):
    score: int
    matched_skills: list[str]
    missing_skills: list[str]
    reason: str
    suggestions: list[str]

def explain_match(resume_data: dict, job_title: str, job_desc: str, job_skills: str, ranker_score: float) -> dict:
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_api_key_here":
        # Fallback if no API key
        return {
            "score": int(ranker_score * 100),
            "matched_skills": [],
            "missing_skills": ["Unknown"],
            "reason": "Missing GEMINI_API_KEY. LLM Reasoning skipped.",
            "suggestions": ["Add API key to .env"]
        }

    client = genai.Client()
    
    prompt = f"""
    You are an expert technical recruiter.
    Analyze the match between this candidate's resume and the job description.
    
    Candidate Data:
    {resume_data}
    
    Job Title: {job_title}
    Job Skills Required: {job_skills}
    Job Description: {job_desc}
    Algorithm Match Score: {ranker_score:.2f}
    
    Provide an explanation of why the job is suitable, the missing skills, and suggestions.
    Return ONLY a JSON matching the requested schema.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': JobExplanation,
        },
    )
    
    try:
        import json
        return json.loads(response.text)
    except Exception as e:
        return {
             "score": int(ranker_score * 100),
             "matched_skills": [],
             "missing_skills": [],
             "reason": f"Failed to generate reasoning: {e}",
             "suggestions": []
        }
