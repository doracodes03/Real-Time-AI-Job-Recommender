import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

INTERACTION_WEIGHTS = {
    'click': 1,
    'save': 2,
    'apply': 3
}

def load_interactions():
    try:
        df = pd.read_csv('data/interactions.csv')
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['user_id', 'job_id', 'interaction_type', 'timestamp'])

def save_interaction(user_id, job_id, interaction_type):
    df = load_interactions()
    new_row = pd.DataFrame([{
        'user_id': user_id,
        'job_id': job_id,
        'interaction_type': interaction_type,
        'timestamp': pd.Timestamp.now()
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv('data/interactions.csv', index=False)

def build_user_item_matrix(interactions_df):
    if interactions_df.empty:
        return pd.DataFrame()
    
    # Map interaction weights
    interactions_df['weight'] = interactions_df['interaction_type'].map(INTERACTION_WEIGHTS).fillna(1)
    
    # Group by user and job, take max weight
    grouped = interactions_df.groupby(['user_id', 'job_id'])['weight'].max().reset_index()
    
    # Pivot
    matrix = grouped.pivot(index='user_id', columns='job_id', values='weight').fillna(0)
    return matrix

def get_cf_recommendations(user_id, jobs_df, top_n=5):
    interactions = load_interactions()
    matrix = build_user_item_matrix(interactions)
    
    # If no interactions at all or user not in matrix
    if matrix.empty or user_id not in matrix.index:
        return pd.DataFrame()
    
    # Compute item-item similarity
    # matrix shape: (n_users, n_jobs)
    # Cosine similarity between columns (jobs)
    item_sim = cosine_similarity(matrix.T)
    item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)
    
    # Get user's interactions
    user_ratings = matrix.loc[user_id]
    
    # Jobs already interacted with
    interacted_jobs = set(user_ratings[user_ratings > 0].index)
    
    scores = {}
    for job_id in item_sim_df.columns:
        if job_id in interacted_jobs:
            continue # Skip already interacted
        
        # Calculate predicted score based on item similarity
        sim_scores = item_sim_df[job_id]
        
        # Weighted sum of similarities using user's ratings
        score = (sim_scores * user_ratings).sum() / (sim_scores.sum() + 1e-9)
        scores[job_id] = score
        
    # Sort scores
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create output dataframe
    results = []
    for job_id, score in sorted_scores[:top_n]:
        # get job details — skip if job_id no longer exists in jobs_df
        matched = jobs_df[jobs_df['id'] == job_id]
        if matched.empty:
            continue
        job_info = matched.iloc[0].to_dict()
        job_info['cf_score'] = score
        results.append(job_info)
        
    return pd.DataFrame(results)

def get_hybrid_recommendations(user_id, content_results_df, cf_results_df, jobs_df):
    """
    content_results_df has 'final_score' (from content recommender)
    cf_results_df has 'cf_score'
    Combines them into 'hybrid_score': 0.6 * content + 0.4 * cf
    """
    if 'id' not in content_results_df.columns:
        # If content results don't have id but only Title, we need to map id.
        # Ensure content_results_df maintains 'id'
        pass
        
    # Standardize content scores (0 to 1 range approx)
    content_max = content_results_df['final_score'].max() if not content_results_df.empty else 1
    if content_max > 0:
        content_results_df['norm_content_score'] = content_results_df['final_score'] / content_max
    else:
        content_results_df['norm_content_score'] = 0
        
    # Standardize CF scores
    if not cf_results_df.empty:
        cf_max = cf_results_df['cf_score'].max()
        if cf_max > 0:
            cf_results_df['norm_cf_score'] = cf_results_df['cf_score'] / cf_max
        else:
            cf_results_df['norm_cf_score'] = 0
    else:
        cf_results_df = pd.DataFrame(columns=['id', 'norm_cf_score'])
        
    # Merge
    merged = pd.merge(content_results_df, cf_results_df[['id', 'norm_cf_score']], on='id', how='left')
    merged['norm_cf_score'] = merged['norm_cf_score'].fillna(0)
    
    # Calculate Hybrid
    merged['hybrid_score'] = (0.6 * merged['norm_content_score']) + (0.4 * merged['norm_cf_score'])
    
    # Sort
    hybrid_jobs = merged.sort_values(by='hybrid_score', ascending=False)
    
    return hybrid_jobs

def get_saved_jobs(user_id, jobs_df):
    """
    Returns the list of jobs previously saved by the user.
    """
    interactions = load_interactions()
    if interactions.empty:
        return pd.DataFrame()
        
    # Filter for this user and 'save' interaction
    saved_ids = interactions[
        (interactions['user_id'] == user_id) & 
        (interactions['interaction_type'] == 'save')
    ]['job_id'].unique()
    
    if len(saved_ids) == 0:
        return pd.DataFrame()
        
    # Get job details - ensure id is string for comparison if needed
    saved_jobs = jobs_df[jobs_df['id'].astype(str).isin([str(sid) for sid in saved_ids])]
    return saved_jobs
