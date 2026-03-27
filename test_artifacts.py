import pandas as pd
import os

artifacts_dir = r'c:/Users/info/Desktop/Job Recommender/artifacts'
jobs_path = os.path.join(artifacts_dir, 'jobs.pkl')

print("Loading jobs.pkl...")
jobs = pd.read_pickle(jobs_path)
print(f"Shape: {jobs.shape}")
print(f"Columns: {list(jobs.columns)}")
print(f"\nFirst row:")
print(jobs.iloc[0])
