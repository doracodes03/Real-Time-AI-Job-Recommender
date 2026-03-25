import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import re
from src.preprocess import preprocess_text

# Configuration
DATA_PATH = "data/jobs_train.csv"
ARTIFACTS_DIR = "artifacts"
# Lowered for memory efficiency on low RAM systems
MAX_TOTAL_JOBS = 200
SAMPLE_SIZE_FOR_FIT = 200
CHUNK_SIZE = 200
MAX_FEATURES = 2000

if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

def get_combined_text(df):
    # Handle missing values
    df['Job Title'] = df['Job Title'].fillna('')
    # Try 'skills' or 'Skills'
    skill_col = 'skills' if 'skills' in df.columns else 'Skills'
    df[skill_col] = df[skill_col].fillna('')
    df['Job Description'] = df['Job Description'].fillna('')
    
    # Combine features
    return (
        df['Job Title'] + " " +
        df[skill_col] + " " +
        df['Job Description']
    ).apply(preprocess_text)

print("--- Starting Offline Training Pipeline ---")
import torch
if torch.cuda.is_available():
    print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}. BERT will use CUDA.")
else:
    print("[INFO] No GPU detected. BERT will use CPU. Training may be slow.")

# Step 1: Fit TF-IDF on a sample to build vocabulary
print(f"Phase 1: Fitting TF-IDF on a sample of {SAMPLE_SIZE_FOR_FIT} rows...")
sample_df = pd.read_csv(DATA_PATH, nrows=SAMPLE_SIZE_FOR_FIT)
combined_sample = get_combined_text(sample_df)

tfidf = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    stop_words='english'
)
tfidf.fit(combined_sample)
joblib.dump(tfidf, os.path.join(ARTIFACTS_DIR, "tfidf.pkl"))
print("Saved tfidf.pkl")

# Step 2: Transform the full dataset in chunks
print("Phase 2: Transforming full dataset in chunks...")
all_vectors = []
metadata_list = []

# Phase 2: Transform the full dataset in chunks
# (Using 500 for demonstration/environment speed)
# MAX_TOTAL_JOBS already set above

processed_count = 0
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
    if processed_count >= MAX_TOTAL_JOBS:
        break
        
    print(f"  Processing chunk: {processed_count} to {processed_count + len(chunk)}...")
    
    combined_chunk = get_combined_text(chunk)
    vectors = tfidf.transform(combined_chunk)
    all_vectors.append(vectors)
    
    # Keep only necessary columns for metadata to save memory
    cols_to_keep = ['Job Title', 'Company', 'Location', 'skills', 'Job Description', 'Experience']
    # Check which columns exist
    available_cols = [c for c in cols_to_keep if c in chunk.columns]
    
    # If 'id' doesn't exist, create it
    if 'id' not in chunk.columns:
        chunk['id'] = range(processed_count, processed_count + len(chunk))
        chunk['id'] = chunk['id'].astype(str)
    
    if 'id' not in available_cols:
        available_cols.insert(0, 'id')
        
    metadata_list.append(chunk[available_cols])
    
    processed_count += len(chunk)

# Phase 3: Finalizing Metadata and TF-IDF Vectors
print("Phase 3: Finalizing Metadata and TF-IDF Vectors...")
metadata_df = pd.concat(metadata_list, ignore_index=True)
from scipy.sparse import vstack
final_tfidf_vectors = vstack(all_vectors)

# Phase 4: Generating Semantic Embeddings (Sentence-BERT)
print("Phase 4: Generating Semantic Embeddings (Sentence-BERT)...")
from src.vectorize import get_bert_embeddings
final_combined_text = get_combined_text(metadata_df)
bert_vectors = get_bert_embeddings(final_combined_text.tolist())

# Explicit memory cleanup
import gc
del all_vectors, combined_sample, combined_chunk, final_combined_text
gc.collect()

# Phase 5: Saving final artifacts
print("Phase 5: Saving final artifacts...")
joblib.dump(tfidf, os.path.join(ARTIFACTS_DIR, "tfidf.pkl"))
joblib.dump(final_tfidf_vectors, os.path.join(ARTIFACTS_DIR, "job_vectors.pkl"))
joblib.dump(bert_vectors, os.path.join(ARTIFACTS_DIR, "bert_job_vectors.pkl"))
metadata_df.to_pickle(os.path.join(ARTIFACTS_DIR, "jobs.pkl"))

print(f"Saved artifacts with {len(metadata_df)} records.")
print("--- Training Pipeline Complete! ---")
