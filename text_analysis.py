#text_analysis.py
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

# TF-IDF asosida o'xshashlikni tekshirish
def check_plagiarism_tfidf(text1, text2):
    # TF-IDF vektorlashtirish
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        # Kosinusli o'xshashlikni hisoblash
        similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
        return similarity
    except:
        # Agar bitta yoki ikkala matn ham bo'sh bo'lsa
        return 0.0

# Neyron tarmoq yordamida semantik o'xshashlikni aniqlash
def check_plagiarism_neural(text1, text2, model):
    try:
        # Umumiy matnlarni vektorlashtirish
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        
        # Umumiy o'xshashlikni hisoblash (Cosine Similarity)
        overall_similarity = util.cos_sim(embedding1, embedding2)[0][0].item()
        
        return overall_similarity
    except:
        # Agar bitta yoki ikkala matn ham bo'sh bo'lsa
        return 0.0

# Jumlalar o'xshashligini tekshirish
def check_sentence_similarity(text1, text2, model, threshold=0.8):
    # Matnni jumlaarga bo'lish
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    
    if not sentences1 or not sentences2:
        return []
    
    similar_sentences = []
    
    # Progress bar yaratish
    total_comparisons = len(sentences1) * len(sentences2)
    with tqdm(total=total_comparisons, desc="Jumlalarni tekshirish", disable=True) as pbar:
        # Har bir jumlani tekshirish
        for i, s1 in enumerate(sentences1):
            if len(s1.split()) < 5:  # Juda qisqa jumlalarni o'tkazib yuborish
                continue
                
            best_match = {"sentence": "", "similarity": 0, "index": -1}
            
            for j, s2 in enumerate(sentences2):
                if len(s2.split()) < 5:  # Juda qisqa jumlalarni o'tkazib yuborish
                    continue
                    
                try:
                    # Jumlalarni vektorlashtirish
                    emb1 = model.encode(s1, convert_to_tensor=True)
                    emb2 = model.encode(s2, convert_to_tensor=True)
                    
                    # O'xshashlikni hisoblash
                    sim = util.cos_sim(emb1, emb2)[0][0].item()
                    
                    if sim > best_match["similarity"]:
                        best_match = {"sentence": s2, "similarity": sim, "index": j}
                except:
                    continue
                
                pbar.update(1)
            
            # Agar o'xshashlik darajasi yuqori bo'lsa
            if best_match["similarity"] > threshold:
                similar_sentences.append({
                    "doc1_sentence": s1,
                    "doc2_sentence": best_match["sentence"],
                    "doc1_index": i,
                    "doc2_index": best_match["index"],
                    "similarity": best_match["similarity"]
                })
    
    return similar_sentences

# Paragraf darajasida o'xshashlikni tekshirish
def check_paragraph_similarity(paragraphs1, paragraphs2, model, threshold=0.8):
    similar_paragraphs = []
    
    # Paragraf listlarini string arrayga aylantirish
    if isinstance(paragraphs1, list):
        paragraphs1 = [p for p in paragraphs1 if len(p.strip()) > 30]
    if isinstance(paragraphs2, list):
        paragraphs2 = [p for p in paragraphs2 if len(p.strip()) > 30]
    
    if not paragraphs1 or not paragraphs2:
        return []
    
    # Har bir paragrafni tekshirish
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
                # Paragraflarni vektorlashtirish
                emb1 = model.encode(p1, convert_to_tensor=True)
                emb2 = model.encode(p2, convert_to_tensor=True)
                
                # O'xshashlikni hisoblash
                sim = util.cos_sim(emb1, emb2)[0][0].item()
                
                if sim > max_sim:
                    max_sim = sim
                    best_match = p2
                    best_idx = j
            except:
                continue
        
        # Agar o'xshashlik darajasi yuqori bo'lsa
        if max_sim > threshold:
            similar_paragraphs.append({
                "doc1_paragraph": p1,
                "doc2_paragraph": best_match,
                "doc1_index": i,
                "doc2_index": best_idx,
                "similarity": max_sim
            })
    
    return similar_paragraphs