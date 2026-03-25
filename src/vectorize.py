# src/vectorize.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Global model instance for efficiency
_bert_model = None

def get_bert_model(model_name='all-MiniLM-L6-v2'):
    """Lazy load the BERT model."""
    global _bert_model
    if _bert_model is None:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
        _bert_model = SentenceTransformer(model_name, device=device)
    return _bert_model

def get_tfidf_entities(corpus):
    tfidf = TfidfVectorizer(max_features=5000)
    vectors = tfidf.fit_transform(corpus)
    return tfidf, vectors

def get_bert_embeddings(corpus):
    """Generate semantic embeddings using Sentence-BERT."""
    model = get_bert_model()
    # Ensure corpus is a list
    if isinstance(corpus, str):
        corpus = [corpus]
    # Batch encoding for memory efficiency
    batch_size = 32  # Adjust as needed for your GPU/CPU RAM
    embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_emb)
    embeddings = np.vstack(embeddings)
    return embeddings

def get_similarity(vec1, vec2):
    """Compute cosine similarity between two sets of vectors."""
    # Ensure vectors are 2D for cosine_similarity
    if len(vec1.shape) == 1:
        vec1 = vec1.reshape(1, -1)
    if len(vec2.shape) == 1:
        vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2).flatten()
