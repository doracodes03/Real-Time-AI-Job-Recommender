import pandas as pd
from sklearn.model_selection import train_test_split

# Load the large job descriptions file
jobs = pd.read_csv('data/job_descriptions.csv')

# Random 80/20 split
train, test = train_test_split(jobs, test_size=0.2, random_state=42)

# Save to new files
train.to_csv('data/jobs_train.csv', index=False)
test.to_csv('data/jobs_test.csv', index=False)

print(f"Train set: {len(train)} rows\nTest set: {len(test)} rows")
