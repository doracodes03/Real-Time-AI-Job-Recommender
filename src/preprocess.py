# src/preprocess.py

import re


# Custom dictionary for skills and roles (can be expanded dynamically)
SKILLS_DICT = [
    "python", "sql", "machine learning", "ml", "r", "javascript", "react", "css", "java", "spring boot", "aws", "docker", "kubernetes", "ci/cd", "ci cd"
]
ROLES_DICT = [
    "data scientist", "python developer", "frontend engineer", "backend developer", "devops engineer", "analyst", "engineer", "developer"
]

# For semantic skill expansion
from src.vectorize import get_bert_embeddings
import numpy as np

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text.strip()



def extract_entities(text, dynamic_skills=None, semantic_expand=False, threshold=0.75):
    """
    Extract skills and roles from text using custom dictionary.
    Optionally expand skills using semantic similarity (Sentence-BERT).
    Args:
        text (str): Input text.
        dynamic_skills (list): Additional skills to consider (optional).
        semantic_expand (bool): If True, use embeddings to match unseen skills.
        threshold (float): Cosine similarity threshold for semantic skill match.
    Returns:
        dict: {"skills": [...], "roles": [...]}
    """
    text = str(text).lower()
    extracted_skills = []
    extracted_roles = []
    skills_dict = SKILLS_DICT.copy()
    if dynamic_skills:
        skills_dict += [s for s in dynamic_skills if s not in skills_dict]

    # Exact match using regex
    for skill in skills_dict:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            extracted_skills.append(skill)

    # Semantic expansion for unseen skills
    if semantic_expand:
        # Tokenize text and check for semantic similarity to known skills
        words = set(text.split())
        # Only check words not already matched
        candidate_words = [w for w in words if w not in extracted_skills and len(w) > 2]
        if candidate_words:
            # Get embeddings for both skills and candidate words
            skill_embs = get_bert_embeddings(skills_dict)
            word_embs = get_bert_embeddings(candidate_words)
            # Compute cosine similarity
            sim_matrix = np.dot(word_embs, np.array(skill_embs).T) / (
                np.linalg.norm(word_embs, axis=1, keepdims=True) * np.linalg.norm(skill_embs, axis=1)
            )
            for idx, word in enumerate(candidate_words):
                max_sim = np.max(sim_matrix[idx])
                if max_sim >= threshold:
                    # Add the closest skill if not already present
                    closest_skill = skills_dict[np.argmax(sim_matrix[idx])]
                    if closest_skill not in extracted_skills:
                        extracted_skills.append(closest_skill)

    for role in ROLES_DICT:
        pattern = r'\b' + re.escape(role) + r'\b'
        if re.search(pattern, text):
            extracted_roles.append(role)

    return {
        "skills": list(set(extracted_skills)),
        "roles": list(set(extracted_roles))
    }

# Utility for semantic skill overlap (for recommend.py)
def semantic_skill_overlap(resume_skills, job_skills, threshold=0.75):
    """
    Compute semantic overlap between two skill lists using embeddings.
    Returns a score between 0 and 1.
    """
    if not resume_skills or not job_skills:
        return 0.0
    resume_embs = get_bert_embeddings(resume_skills)
    job_embs = get_bert_embeddings(job_skills)
    sim_matrix = np.dot(resume_embs, np.array(job_embs).T) / (
        np.linalg.norm(resume_embs, axis=1, keepdims=True) * np.linalg.norm(job_embs, axis=1)
    )
    # Count matches above threshold
    matches = (sim_matrix >= threshold).sum()
    return matches / len(job_skills)