import requests
import os
from preprocess import preprocess_text
from vectorize import get_bert_embeddings, get_similarity

def fetch_jobs_from_jsearch(query, location, results_limit=20):
    """Fetch jobs from JSearch API (RapidAPI). Recommended for reliability."""
    api_key = os.getenv('JSEARCH_API_KEY')
    
    if not api_key:
        print("\n⚠️  JSearch API key not set.")
        print("   Get a free API key from: https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch")
        print("   Set it in your environment: set JSEARCH_API_KEY=your_key")
        return []
    
    url = "https://jsearch.p.rapidapi.com/search"
    
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "jsearch.p.rapidapi.com",
        "Content-Type": "application/json"
    }
    
    # Combine query and location as shown in the API example
    search_query = f"{query} jobs in {location}"
    
    params = {
        "query": search_query,
        "page": "1",
        "num_pages": "1",
        "country": "us",
        "date_posted": "all"
    }
    
    print(f"\n🔍 Fetching real-time jobs from JSearch API for '{query}' in {location}...")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        
        jobs_data = response.json()
        jobs = jobs_data.get('data', [])
        
        if not jobs:
            print(f"⚠️  No jobs found for '{query}' in {location}")
            return []
        
        job_list = []
        for job in jobs:
            job_list.append({
                'title': job.get('job_title', ''),
                'description': job.get('job_description', ''),
                'skills': '',
                'company': job.get('employer_name', ''),
                'location': job.get('job_location', location),
                'apply_url': job.get('job_apply_link', ''),
            })
        
        print(f"✅ Found {len(job_list)} jobs from JSearch API\n")
        return job_list
    
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error {e.response.status_code}: {e.response.reason}")
        if e.response.status_code == 401:
            print("   Invalid API key. Check your JSEARCH_API_KEY environment variable.")
        elif e.response.status_code == 429:
            print("   Rate limit exceeded. Try again later.")
        raise
    except Exception as e:
        print(f"\n❌ Error fetching from JSearch API: {e}")
        raise

def fetch_jobs_from_remotive(query, location, results_limit=20):
    """Fallback: Fetch jobs from Remotive API."""
    url = "https://remotive.com/api/v1/jobs"
    
    params = {
        'search': query,
        'limit': results_limit,
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    print(f"\n🔍 Fetching from Remotive API for '{query}'...")
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        jobs_data = response.json()
        jobs = jobs_data.get('jobs', [])
        
        if not jobs:
            print(f"⚠️  No jobs found for '{query}'")
            return []
        
        job_list = []
        for job in jobs:
            job_list.append({
                'title': job.get('title', ''),
                'description': job.get('description', ''),
                'skills': job.get('tags', []) if isinstance(job.get('tags'), list) else '',
                'company': job.get('company_name', ''),
                'location': job.get('location', location),
                'apply_url': job.get('url', ''),
            })
        
        print(f"✅ Found {len(job_list)} jobs from Remotive API\n")
        return job_list
    
    except Exception as e:
        print(f"   Remotive failed: {e}")
        return []

def fetch_jobs_from_api(query, location, results_limit=20):
    """Fetch jobs from multiple APIs, trying JSearch first (recommended)."""
    
    print("\n" + "="*80)
    print("🌐 REAL-TIME JOB FETCHING")
    print("="*80)
    
    # Try JSearch first (most reliable)
    jobs = fetch_jobs_from_jsearch(query, location, results_limit)
    
    # Fallback to Remotive if JSearch fails or no key
    if not jobs:
        print("\n   Trying Remotive API as fallback...")
        jobs = fetch_jobs_from_remotive(query, location, results_limit)
    
    if not jobs:
        print("\n❌ No jobs could be fetched from any API.")
        print("\n📝 SETUP INSTRUCTIONS:")
        print("   1. Sign up for free JSearch API: https://rapidapi.com/Lfromm/api/jsearch")
        print("   2. Copy your API key")
        print("   3. Set environment variable in PowerShell:")
        print("      $env:JSEARCH_API_KEY = 'your_api_key_here'")
        print("   4. Run this script again")
        raise Exception("No jobs available from API sources")
    
    return jobs

def get_resume_text():
    print("Enter your resume text (end with a blank line):")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    return '\n'.join(lines)

# 1. Get resume and fetch jobs from Remotive API
resume_text = get_resume_text()
query = input("Enter job search query (e.g., 'data scientist'): ") or "data scientist"
location = input("Enter location (e.g., 'India'): ") or "India"
N = int(input("How many top jobs to show? (default 5): ") or 5)

try:
    api_jobs = fetch_jobs_from_api(query=query, location=location)
    if not api_jobs:
        print("\n❌ No jobs found. Please try a different search query.")
        exit(1)
except Exception as e:
    print(f"\n❌ Failed to fetch jobs. Exiting.")
    exit(1)

# 2. Normalize and preprocess
for job in api_jobs:
    combined = f"{job['title']} {job['description']} {job.get('skills', '')}"
    job['clean_text'] = preprocess_text(combined)

# 3. Vectorize jobs and resume
job_texts = [job['clean_text'] for job in api_jobs]
job_vectors = get_bert_embeddings(job_texts)
resume_clean = preprocess_text(resume_text)
resume_vec = get_bert_embeddings([resume_clean])[0]

# 4. Compute similarity
scores = get_similarity(resume_vec, job_vectors)

# 5. Attach scores and sort
for job, score in zip(api_jobs, scores):
    job['score'] = float(score)
top_jobs = sorted(api_jobs, key=lambda x: x['score'], reverse=True)[:N]

print(f"\n{'='*80}")
print(f"🎯 Top {N} job matches for '{query}' in {location}")
print(f"{'='*80}\n")

for i, job in enumerate(top_jobs, 1):
    match_percent = min(100, int(job['score'] * 100))
    print(f"{i}. {job['title']}")
    print(f"   Company: {job['company']}")
    print(f"   Location: {job['location']}")
    print(f"   Match Score: {match_percent}% (Similarity: {job['score']:.3f})")
    if job.get('skills'):
        skills_text = ', '.join(job['skills']) if isinstance(job['skills'], list) else job['skills']
        print(f"   Skills: {skills_text}")
    print(f"   Apply: {job['apply_url']}")
    print(f"   Description: {job['description'][:120]}...")
    print()

print(f"{'='*80}")
print("✅ Recommendation pipeline complete! Real-time job matching successful.")
print(f"{'='*80}")