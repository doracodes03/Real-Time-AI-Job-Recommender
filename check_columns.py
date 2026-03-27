import pickle
import pandas as pd
import os

print("=" * 60)
print("Checking artifacts/jobs.pkl")
print("=" * 60)
pkl_path = "artifacts/jobs.pkl"
if os.path.exists(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
            print(f"Type: DataFrame")
            print(f"Shape: {data.shape}")
            print(f"\nColumns: {list(data.columns)}")
            print(f"\nFirst few rows:")
            print(data.head(2))
        else:
            print(f"Type: {type(data)}")
            print(f"Content: {data}")
    except Exception as e:
        print(f"Error reading pkl: {e}")
else:
    print(f"File not found: {pkl_path}")

print("\n" + "=" * 60)
print("Checking data/jobs.csv")
print("=" * 60)
csv_path = "data/jobs.csv"
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head(2))
    except Exception as e:
        print(f"Error reading csv: {e}")
else:
    print(f"File not found: {csv_path}")
