# paste.py
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

def check_plagiarism_tfidf(text1, text2):
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
        return similarity
    except:
        return 0.0

def check_plagiarism_neural(text1, text2, model):
    try:
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        overall_similarity = util.cos_sim(embedding1, embedding2)[0][0].item()
        return overall_similarity
    except:
        return 0.0

def check_paragraph_similarity(paragraphs1, paragraphs2, model, threshold=0.8):
    similar_paragraphs = []
    if isinstance(paragraphs1, list):
        paragraphs1 = [p for p in paragraphs1 if len(p.strip()) > 30]
    if isinstance(paragraphs2, list):
        paragraphs2 = [p for p in paragraphs2 if len(p.strip()) > 30]
    if not paragraphs1 or not paragraphs2:
        return []
    for i, p1 in enumerate(paragraphs1):
        if isinstance(p1, str) and len(p1.strip()) < 30:
            continue
        max_sim = 0
        best_match = ""
        best_idx = -1
        for j, p2 in enumerate(paragraphs2):
            if isinstance(p2, str) and len(p2.strip()) < 30:
                continue
            try:
                emb1 = model.encode(p1, convert_to_tensor=True)
                emb2 = model.encode(p2, convert_to_tensor=True)
                sim = util.cos_sim(emb1, emb2)[0][0].item()
                if sim > max_sim:
                    max_sim = sim
                    best_match = p2
                    best_idx = j
            except:
                continue
        if max_sim > threshold:
            similar_paragraphs.append({
                "doc1_paragraph": p1,
                "doc2_paragraph": best_match,
                "doc1_index": i,
                "doc2_index": best_idx,
                "similarity": max_sim
            })
    return similar_paragraphs