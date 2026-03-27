import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import re
from src.preprocess import preprocess_text

# Configuration - Optimized for i5 + limited RAM
DATA_PATH = "data/jobs.csv"  # Use unified dataset - no train/test split needed
ARTIFACTS_DIR = "artifacts"
# Safe limit for i5 with BERT embeddings (50K-100K jobs)
MAX_TOTAL_JOBS = 50000
SAMPLE_SIZE_FOR_FIT = 2000
CHUNK_SIZE = 256  # Smaller chunks to prevent memory spikes
MAX_FEATURES = 1000  # Reduced from 2000 to save memory

if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

def get_combined_text(df):
    """Combine job features with description truncation for memory efficiency."""
    df = df.copy()  # Avoid SettingWithCopyWarning
    
    # Handle missing values
    df['Job Title'] = df['Job Title'].fillna('')
    # Try 'skills' or 'Skills'
    skill_col = 'skills' if 'skills' in df.columns else 'Skills'
    df[skill_col] = df[skill_col].fillna('')
    df['Job Description'] = df['Job Description'].fillna('')
    
    # ⚡ CRITICAL: Truncate descriptions to 500 chars to reduce memory footprint
    df['Job Description'] = df['Job Description'].str[:500]
    
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

# Step 1.5: Load and sample data (no train/test split - this is retrieval, not supervised learning)
print(f"Phase 1.5: Loading dataset from {DATA_PATH}...")
all_data = pd.read_csv(DATA_PATH)
print(f"  Total rows available: {len(all_data)}")

# ⚡ SMART SAMPLING: Use 50K jobs for MVP (prevents crashes)
if len(all_data) > MAX_TOTAL_JOBS:
    print(f"  Sampling {MAX_TOTAL_JOBS} jobs (too many for i5 + limited RAM)...")
    all_data = all_data.sample(n=MAX_TOTAL_JOBS, random_state=42)
    print(f"  Using sampled dataset: {len(all_data)} jobs")

# Step 2: Fit TF-IDF on a sample to build vocabulary
print(f"Phase 2: Fitting TF-IDF on a sample of {SAMPLE_SIZE_FOR_FIT} rows...")
sample_df = all_data.head(SAMPLE_SIZE_FOR_FIT) if len(all_data) > SAMPLE_SIZE_FOR_FIT else all_data
combined_sample = get_combined_text(sample_df)

tfidf = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    stop_words='english'
)
tfidf.fit(combined_sample)
joblib.dump(tfidf, os.path.join(ARTIFACTS_DIR, "tfidf.pkl"))
print("Saved tfidf.pkl")

# Step 3: Transform dataset in chunks
print("Phase 3: Transforming dataset in chunks...")
all_vectors = []

# ⚡ Process already-sampled data in chunks
processed_count = 0
for chunk_num, i in enumerate(range(0, len(all_data), CHUNK_SIZE)):
    chunk = all_data.iloc[i:i+CHUNK_SIZE].copy()
    
    print(f"  TF-IDF chunk {chunk_num}: {i} to {min(i+CHUNK_SIZE, len(all_data))}...")
    
    combined_chunk = get_combined_text(chunk)
    vectors = tfidf.transform(combined_chunk)
    all_vectors.append(vectors)
    
    processed_count += len(chunk)

# Step 4: Stack TF-IDF vectors
print("Phase 4: Stacking TF-IDF vectors...")
from scipy.sparse import vstack
final_tfidf_vectors = vstack(all_vectors)
print(f"  TF-IDF matrix shape: {final_tfidf_vectors.shape}")

# Keep metadata with only essential columns
print("Phase 5: Preparing metadata...")
cols_to_keep = ['Job Title', 'Company', 'Location', 'skills', 'Job Description', 'Experience']
available_cols = [c for c in cols_to_keep if c in all_data.columns]

if 'id' not in all_data.columns:
    all_data['id'] = range(len(all_data))
    all_data['id'] = all_data['id'].astype(str)

if 'id' not in available_cols:
    available_cols.insert(0, 'id')

metadata_df = all_data[available_cols].reset_index(drop=True)
print(f"  Metadata prepared: {len(metadata_df)} records")

# Step 6: Generate Semantic Embeddings (Sentence-BERT) - CHUNKED TO PREVENT CRASHES
print("Phase 6: Generating BERT Embeddings (chunked)...")
from src.vectorize import get_bert_embeddings

bert_vectors_list = []

# ⚡ CRITICAL: Process BERT in batches to prevent memory spikes
for batch_num, i in enumerate(range(0, len(metadata_df), CHUNK_SIZE)):
    batch_start = i
    batch_end = min(i + CHUNK_SIZE, len(metadata_df))
    print(f"  BERT batch {batch_num}: {batch_start} to {batch_end}/{len(metadata_df)}...")
    
    batch_df = metadata_df.iloc[batch_start:batch_end]
    batch_texts = get_combined_text(batch_df).tolist()
    
    # Generate embeddings for this batch only
    batch_embeddings = get_bert_embeddings(batch_texts)
    
    # ⚡ SAVE TO DISK immediately to free RAM (don't accumulate in memory)
    chunk_file = os.path.join(ARTIFACTS_DIR, f"bert_chunk_{batch_num:04d}.npy")
    np.save(chunk_file, batch_embeddings)
    print(f"    Saved {chunk_file}")
    
    bert_vectors_list.append(chunk_file)

# Step 7: Reload and stack BERT vectors from disk
print("Phase 7: Stacking BERT embeddings from disk...")
bert_vectors = np.vstack([np.load(f) for f in bert_vectors_list])
print(f"  BERT vectors shape: {bert_vectors.shape}")

# Explicit memory cleanup
import gc
del all_vectors, combined_sample, combined_chunk
gc.collect()

# Step 8: Saving final artifacts
print("Phase 8: Saving final artifacts...")
joblib.dump(tfidf, os.path.join(ARTIFACTS_DIR, "tfidf.pkl"))
joblib.dump(final_tfidf_vectors, os.path.join(ARTIFACTS_DIR, "job_vectors.pkl"))
joblib.dump(bert_vectors, os.path.join(ARTIFACTS_DIR, "bert_job_vectors.pkl"))
metadata_df.to_pickle(os.path.join(ARTIFACTS_DIR, "jobs.pkl"))

print(f"\n✅ Saved artifacts with {len(metadata_df)} records.")
print(f"   - TF-IDF vectors: {final_tfidf_vectors.shape}")
print(f"   - BERT vectors: {bert_vectors.shape}")
print("--- Training Pipeline Complete! ---")
