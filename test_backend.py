#!/usr/bin/env python
"""Diagnose backend startup issues"""
import sys
import traceback

print("=" * 60)
print("🔍 Diagnosing Backend Issues")
print("=" * 60)

try:
    print("\n1️⃣  Testing API import...")
    from api import app
    print("   ✅ API imported successfully")
    
    print("\n2️⃣  Testing FastAPI setup...")
    print(f"   ✅ FastAPI app: {app}")
    print(f"   ✅ Routes: {len(app.routes)} endpoints registered")
    
    print("\n3️⃣  Testing artifact loading state...")
    from api import state
    print(f"   ✅ TF-IDF loaded: {state['tfidf'] is not None}")
    print(f"   ✅ Job vectors loaded: {state['job_vectors'] is not None}")
    print(f"   ✅ BERT vectors loaded: {state['bert_job_vectors'] is not None}")
    print(f"   ✅ Jobs dataframe: {len(state['df']) if state['df'] is not None else 0} records")
    
    print("\n4️⃣  Testing health endpoint with lifespan...")
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        response = client.get("/health")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.json()}")
    
    if response.status_code == 200:
        print("\n✅ ALL CHECKS PASSED - Backend is ready!")
    else:
        print(f"\n❌ Health check failed: {response.text}")
        
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
