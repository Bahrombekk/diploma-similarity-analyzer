from docx import Document
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import re

# NLTK resurslarini yuklash
nltk.download('punkt')

# Word fayldan matn chiqarish
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + ' '
    return text.strip()

# Matnni tozalash (maxsus belgilar va kichik harfga o'tkazish)
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

# Neyron tarmoq yordamida o'xshashlikni aniqlash
def check_plagiarism_neural(doc1_path, doc2_path, threshold=0.8):
    # Matnlarni chiqarish
    text1 = extract_text_from_docx(doc1_path)
    text2 = extract_text_from_docx(doc2_path)
    
    # Matnlarni jumlalarga bo'lish
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    
    # Sentence Transformers modelini yuklash
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Matnlarni umumiy vektorlashtirish
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    
    # Umumiy o'xshashlikni hisoblash (Cosine Similarity)
    overall_similarity = util.cos_sim(embedding1, embedding2)[0][0].item()
    
    # Jumla darajasida o'xshashlikni aniqlash
    similar_sentences = []
    for s1 in sentences1:
        for s2 in sentences2:
            emb1 = model.encode(s1, convert_to_tensor=True)
            emb2 = model.encode(s2, convert_to_tensor=True)
            sim = util.cos_sim(emb1, emb2)[0][0].item()
            if sim > threshold:
                similar_sentences.append((s1, s2, sim))
    
    # Natijalarni chop etish
    print(f"Umumiy semantik o'xshashlik: {overall_similarity * 100:.2f}%")
    
    if similar_sentences:
        print("\nO'xshash jumlalar topildi:")
        for s1, s2, sim in similar_sentences:
            print(f"Sizning diplom: {s1}")
            print(f"O'quvchi diplom: {s2}")
            print(f"O'xshashlik darajasi: {sim * 100:.2f}%\n")
    else:
        print("\nO'xshash jumlalar topilmadi.")
    
    return overall_similarity

# Fayllarni tekshirish
doc1_path = '/home/bahrombek/Desktop/New Folder 1/DOCX Document 1.docx'  # Sizning diplom ishingiz
doc2_path = '/home/bahrombek/Desktop/New Folder 1/DOCX Document 2.docx'  # O'quvchining diplom ishi
check_plagiarism_neural(doc1_path, doc2_path, threshold=0.8)