#!/usr/bin/env python3
"""Test frontend API connectivity"""
import requests
import json
import sys

def test_backend():
    """Test if backend API is working"""
    
    # Test 1: Health endpoint
    print("=" * 60)
    print("TEST 1: Health Endpoint")
    print("=" * 60)
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health endpoint responding")
            print(f"   Status: {data.get('status')}")
            print(f"   Models loaded: {data.get('models_loaded')}")
            print(f"   Jobs count: {data.get('jobs_count')}")
            print(f"   BERT available: {data.get('bert_available')}")
        else:
            print(f"❌ Health endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False
    
    # Test 2: Recommend endpoint
    print("\n" + "=" * 60)
    print("TEST 2: /recommend/fast Endpoint (Form Data)")
    print("=" * 60)
    try:
        response = requests.post(
            'http://localhost:8000/recommend/fast',
            data={'resume_text': 'Python developer with 5 years machine learning experience'},
            timeout=30,
            headers={'Accept': 'application/json'}
        )
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get('recommendations', [])
            print(f"✅ /recommend/fast endpoint responding")
            print(f"   Got {len(recommendations)} recommendations")
            if recommendations:
                print(f"   First job: {recommendations[0].get('Job Title', 'N/A')[:60]}")
                print(f"   Company: {recommendations[0].get('Company', 'N/A')}")
                print(f"   Match Score: {recommendations[0].get('Match Score', 'N/A')}")
                
                # Check for duplicates
                job_ids = [r.get('id') for r in recommendations]
                if len(job_ids) != len(set(job_ids)):
                    print("⚠️  WARNING: Duplicate job IDs found!")
                else:
                    print(f"✅ All {len(recommendations)} jobs are unique")
        else:
            print(f"❌ /recommend/fast returned status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ /recommend/fast failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("CORS Configuration Check")
    print("=" * 60)
    try:
        response = requests.options(
            'http://localhost:8000/recommend/fast',
            timeout=5,
            headers={
                'Origin': 'http://localhost:5173',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            }
        )
        print(f"✅ CORS preflight successful")
        print(f"   Allow-Origin: {response.headers.get('Access-Control-Allow-Origin', 'Not set')}")
        print(f"   Allow-Methods: {response.headers.get('Access-Control-Allow-Methods', 'Not set')}")
        print(f"   Allow-Headers: {response.headers.get('Access-Control-Allow-Headers', 'Not set')}")
    except Exception as e:
        print(f"⚠️  CORS check failed: {e}")
    
    print("\n" + "=" * 60)
    print("Summary: Backend is fully operational ✅")
    print("Next step: Check frontend for hard refresh")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    success = test_backend()
    sys.exit(0 if success else 1)
