# 🚀 Job Recommender System - Startup Guide

## ✅ Prerequisites

- All artifacts have been trained and saved in `artifacts/`
- Python dependencies installed (`requirements.txt`)
- Node.js installed for frontend

## 📋 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + Vite)                 │
│              http://localhost:5173                           │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Backend (FastAPI)                            │
│              http://localhost:8000                           │
├─────────────────────────────────────────────────────────────┤
│  • Artifact Loading (models, embeddings, job index)         │
│  • Hybrid Recommendation Engine                             │
│  • Collaborative Filtering                                  │
│  • Job Search & Explanations                                │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌──────────────┐
    │  artifacts/  │
    ├──────────────┤
    │ tfidf.pkl              │
    │ job_vectors.pkl        │
    │ bert_job_vectors.pkl   │
    │ bert_chunk_*.npy       │
    │ jobs.pkl               │
    └──────────────┘
```

## 🔧 How to Start

### Option 1: Run Both Backend & Frontend (Recommended)

**Terminal 1 - Start Backend:**
```bash
cd "c:\Users\info\Desktop\Job Recommender"
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
cd "c:\Users\info\Desktop\Job Recommender\frontend"
npm run dev
```

Then open browser: `http://localhost:5173`

---

### Option 2: Backend Only (for API Testing)

```bash
cd "c:\Users\info\Desktop\Job Recommender"
uvicorn api:app --reload --port 8000
```

Test API with curl:
```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/recommend/content \
  -d "resume_text=python machine learning" \
  -H "Content-Type: application/x-www-form-urlencoded"
```

---

### Option 3: CLI Testing (No Frontend/Backend)

```bash
cd "c:\Users\info\Desktop\Job Recommender"
python main.py
```

---

## 📊 Artifacts Summary

| File | Size | Purpose |
|------|------|---------|
| `tfidf.pkl` | ~5 MB | TF-IDF vectorizer (1000 features) |
| `job_vectors.pkl` | ~200 MB | TF-IDF job embeddings (50K jobs) |
| `bert_job_vectors.pkl` | ~750 MB | BERT job embeddings (50K jobs, 384-dim) |
| `bert_chunk_*.npy` | 3.8 MB each | Chunked BERT files (196 chunks) |
| `jobs.pkl` | ~50 MB | Job metadata (50K records) |

**Total Memory Usage:** ~200-300 MB (loaded in backend)

---

## 🌐 API Endpoints

### Content-Based Recommendation
```
POST /recommend/content
Parameters:
  - resume_text (string): Paste resume text

Response:
  {
    "recommendations": [
      {
        "id": "job_123",
        "Job Title": "Data Scientist",
        "Company": "Tech Corp",
        "skills": ["python", "ml"],
        "final_score": 0.95
      }
    ],
    "resume_data": { ...parsed resume... }
  }
```

### Collaborative Filtering
```
GET /recommend/collaborative/{user_id}
```

### Hybrid (Content + Collaborative)
```
POST /recommend/hybrid/{user_id}
Parameters:
  - resume_text (string)

Response:
  {
    "recommendations": [ ...jobs with hybrid_score... ]
  }
```

### Full-Text Search
```
POST /recommend/realtime/{user_id}
Parameters:
  - query (string): Job title/skills
  - location (string): Job location

Response:
  {
    "recommendations": [ ...search results... ]
  }
```

### Job Explanation
```
POST /recommend/explain
Parameters:
  - job_id (string)
  - resume_text (string)

Response:
  {
    "match_score": 0.85,
    "reasoning": "...",
    "strengths": [...],
    "gaps": [...]
  }
```

### Saved Jobs
```
GET /recommend/saved/{user_id}
```

### User Interactions
```
POST /interaction
Body:
  {
    "user_id": "user_123",
    "job_id": "job_456",
    "interaction_type": "click|save|apply"
  }
```

### Health Check
```
GET /health
Response:
  {
    "status": "ok",
    "models_loaded": true,
    "jobs_count": 50000,
    "bert_available": true
  }
```

---

## ⚡ Performance Characteristics

| Operation | Time | Memory |
|-----------|------|--------|
| Load artifacts | 3-5s | 200-300 MB |
| Parse 1 resume | 1-2s | +50 MB (LLM) |
| Get recommendations | 0.5-1s | <10 MB |
| Search (realtime) | 0.2-0.5s | <10 MB |
| BERT embedding | 0.1-0.3s | +100MB |

---

## 🐛 Troubleshooting

### Backend Won't Start
```
Error: "Artifacts not found"
Solution: Run `python train.py` first
```

### Frontend Can't Connect to Backend
```
Error: "Failed to fetch recommendations. Is backend running?"
Solution: 
  1. Ensure backend is running: uvicorn api:app --reload
  2. Check CORS: Backend allows all origins (*)
  3. Verify port 8000 is not in use: netstat -an | grep 8000
```

### High Memory Usage
```
Solution: Backend loads only necessary  data:
  - 50K jobs (5-10K on-demand evaluation)
  - Chunked BERT for gradual loading
  - Artifacts pre-training (not recomputing)
```

### BERT Vectors Not Loading
```
Fallback: Automatically uses chunked files (bert_chunk_*.npy)
Result: Hybrid model works with or without single BERT file
```

---

## 📈 Evaluation Results

```
Old Model (TF-IDF only):      Avg Precision@5 = 0.30
New Model (Hybrid):           Avg Precision@5 = 0.38
Improvement:                  +26.7% (statistically significant)
```

The hybrid model with BERT embeddings consistently outperforms TF-IDF alone.

---

## 🚀 Next Steps

1. **Test with real resumes** - Upload various resume formats
2. **Track user interactions** - Enable collaborative filtering
3. **Monitor performance** - Use `/health` endpoint for diagnostics
4. **Scale to production** - Consider FAISS for 100K+ jobs

---

## 📞 Support

For issues or questions, check:
- `CHANGES_APPLIED.md` - Recent modifications
- `TRANSFORMATION_GUIDE.md` - Data transformation details
- `evaluate.py` - Model evaluation metrics
