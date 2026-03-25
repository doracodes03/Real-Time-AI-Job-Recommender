# src/llm_parser.py

import os
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

load_dotenv()

class ParsedResume(BaseModel):
    skills: list[str]
    experience: int
    roles: list[str]
    education: str

def parse_resume_with_llm(resume_text: str) -> dict:
    # If no API key, fallback to a dummy response so the code runs without an API key
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_api_key_here":
        print("[Warning] GEMINI_API_KEY not found. Using dummy LLM parse.")
        from src.preprocess import extract_entities
        entities = extract_entities(resume_text)
        return {
            "skills": entities["skills"],
            "roles": entities["roles"],
            "experience": 4, # fallback
            "education": "Unknown"
        }

    client = genai.Client()
    
    prompt = f"""
    You are an expert technical recruiter. Parse the following resume text and extract the key information.
    Return ONLY a JSON object matching the requested schema.
    
    Resume Text:
    {resume_text}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': ParsedResume,
        },
    )
    
    # Parse the response back into a dict representation of the model
    try:
        import json
        return json.loads(response.text)
    except Exception as e:
        print(f"[Error] Failed to parse LLM output: {e}")
        return {
             "skills": [],
             "roles": [],
             "experience": 0,
             "education": "Unknown"
        }
