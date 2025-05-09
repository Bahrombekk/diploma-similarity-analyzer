#chapter_analysis.py
import os
import time
from sentence_transformers import SentenceTransformer
from utils import extract_text_from_docx, clean_text, split_by_chapters, format_elapsed_time
from paste import check_plagiarism_tfidf, check_plagiarism_neural, check_paragraph_similarity
from visualization import visualize_similarity, generate_report

def analyze_chapters(doc1_path, doc2_path, output_dir="./plagiarism_report"):
    """
    Boblar bo'yicha plagiat tekshiruvini amalga oshirish
    """
    print(f"\n{'='*50}")
    print(f"BOBLAR BO'YICHA PLAGIAT TEKSHIRUVI")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # Modelni yuklash
    print("Modelni yuklash...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 1-hujjatni yuklash
    print(f"1-hujjatni o'qish: {os.path.basename(doc1_path)}")
    doc1_paragraphs = extract_text_from_docx(doc1_path)
    if not doc1_paragraphs:
        print("XATO: 1-hujjat paragraflarni olishda xatolik yuz berdi.")
        return
        
    # 2-hujjatni yuklash
    print(f"2-hujjatni o'qish: {os.path.basename(doc2_path)}")
    doc2_paragraphs = extract_text_from_docx(doc2_path)
    if not doc2_paragraphs:
        print("XATO: 2-hujjat paragraflarni olishda xatolik yuz berdi.")
        return
    
    # Hujjatlarni boblar bo'yicha ajratish
    print("Hujjatlarni boblar bo'yicha ajratish...")
    doc1_chapters = split_by_chapters(doc1_paragraphs)
    doc2_chapters = split_by_chapters(doc2_paragraphs)
    
    # Har bir bobni tekshirish
    similarities = {}
    chapter_paragraph_sim = {}
    
    print(f"\n{'='*50}")
    print(f"BOBLAR BO'YICHA TEKSHIRUV BOSHLANDI")
    print(f"{'='*50}")
    
    for ch_num, ch_paras in doc1_chapters.items():
        ch_name = "Kirish" if ch_num == "0" else f"{ch_num}-bob"
        print(f"\n{ch_name} tahlili...")
        
        # Bobni matn ko'rinishiga o'tkazish
        ch_text = " ".join(ch_paras)
        ch_text_clean = clean_text(ch_text)
        
        # 2-hujjatda shu nomerli bob bormi?
        if ch_num in doc2_chapters:
            doc2_ch_paras = doc2_chapters[ch_num]
            doc2_ch_text = " ".join(doc2_ch_paras)
            doc2_ch_text_clean = clean_text(doc2_ch_text)
            
            # TF-IDF asosida tekshirish
            tfidf_sim = check_plagiarism_tfidf(ch_text_clean, doc2_ch_text_clean)
            print(f"  - TF-IDF o'xshashlik: {tfidf_sim*100:.1f}%")
            
            # Semantik o'xshashlikni tekshirish
            neural_sim = check_plagiarism_neural(ch_text_clean, doc2_ch_text_clean, model)
            print(f"  - Semantik o'xshashlik: {neural_sim*100:.1f}%")
            
            # Umumiy o'xshashlik (o'rtacha)
            overall_sim = (tfidf_sim + neural_sim) / 2
            print(f"  - Umumiy o'xshashlik: {overall_sim*100:.1f}%")
            
            # Paragraflar o'xshashligini tekshirish
            print("  - Paragraflar o'xshashligini tekshirish...")
            paragraph_similarities = check_paragraph_similarity(ch_paras, doc2_ch_paras, model, threshold=0.7)
            print(f"    {len(paragraph_similarities)} ta o'xshash paragraf topildi")
            
            # Natijalarni saqlash
            similarities[ch_num] = {
                "tfidf": tfidf_sim,
                "neural": neural_sim,
                "overall": overall_sim
            }
            
            chapter_paragraph_sim[ch_num] = paragraph_similarities
        else:
            print(f"  - 2-hujjatda mos {ch_name} topilmadi.")
            similarities[ch_num] = {
                "tfidf": 0.0,
                "neural": 0.0,
                "overall": 0.0
            }
    
    # O'xshash boblarni topish
    for ch_num, ch_paras in doc2_chapters.items():
        if ch_num not in doc1_chapters:
            ch_name = "Kirish" if ch_num == "0" else f"{ch_num}-bob"
            print(f"\n2-hujjatdagi {ch_name} 1-hujjatda mavjud emas.")
    
    # Hisobot yaratish
    print(f"\n{'='*50}")
    print("HISOBOT YARATILMOQDA...")
    
    # Vizualizatsiya qilish
    chart_path = visualize_similarity(similarities, output_dir)
    
    # Hisobot yaratish
    doc1_name = os.path.basename(doc1_path)
    doc2_name = os.path.basename(doc2_path)
    report_path = generate_report(doc1_name, doc2_name, similarities, chapter_paragraph_sim, output_dir)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"TEKSHIRUV YAKUNLANDI")
    print(f"{'='*50}")
    print(f"Sarflangan vaqt: {format_elapsed_time(elapsed)}")
    print(f"Hisobot yaratildi: {report_path}")
    print(f"Vizualizatsiya: {chart_path}")
    
    return {
        "report_path": report_path,
        "chart_path": chart_path,
        "similarities": similarities,
        "chapter_paragraph_sim": chapter_paragraph_sim
    }