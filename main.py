# main.py
import os
import sys
import argparse
from chapter_analysis import analyze_chapters
from sentence_transformers import SentenceTransformer
from utils import extract_text_from_docx, format_elapsed_time
from paste import check_plagiarism_tfidf, check_plagiarism_neural
from text_analysis import check_sentence_similarity

def main():
    # Argumentlarni tahlil qilish
    parser = argparse.ArgumentParser(description='Plagiat tekshiruv tizimi')
    parser.add_argument('--doc1', required=True, help='Birinchi Word hujjat yo\'li')
    parser.add_argument('--doc2', required=True, help='Ikkinchi Word hujjat yo\'li')
    parser.add_argument('--output', default='./plagiarism_report', help='Hisobot saqlash papkasi')
    parser.add_argument('--mode', choices=['full', 'chapters', 'sentences'], default='full', 
                        help='Tekshiruv rejimi: full (to\'liq), chapters (boblar), sentences (jumlalar)')
    
    args = parser.parse_args()
    
    # Fayl mavjudligini tekshirish
    if not os.path.exists(args.doc1):
        print(f"XATO: {args.doc1} fayli topilmadi.")
        return
    
    if not os.path.exists(args.doc2):
        print(f"XATO: {args.doc2} fayli topilmadi.")
        return
    
    # Hisobot papkasini yaratish
    os.makedirs(args.output, exist_ok=True)
    
    # Boblar bo'yicha tahlil
    if args.mode in ['full', 'chapters']:
        chapter_results = analyze_chapters(args.doc1, args.doc2, args.output)
    
    # Jumlalar bo'yicha tahlil
    if args.mode in ['full', 'sentences']:
        print(f"\n{'='*50}")
        print(f"JUMLALAR BO'YICHA PLAGIAT TEKSHIRUVI")
        print(f"{'='*50}")
        print("Jumlalar bo'yicha tahlil qilish...")
        
        # Fayllardan matnni olish
        doc1_paragraphs = extract_text_from_docx(args.doc1)
        doc2_paragraphs = extract_text_from_docx(args.doc2)
        
        if not doc1_paragraphs or not doc2_paragraphs:
            print("XATO: Hujjatlardan matnni o'qishda muammo yuz berdi.")
            return
            
        # Matnlarni birlashtirish
        doc1_text = " ".join(doc1_paragraphs)
        doc2_text = " ".join(doc2_paragraphs)
        
        # Modelni yuklash
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Jumlalar o'xshashligini tekshirish
        print("Jumlalar o'xshashligini tekshirish...")
        similar_sentences = check_sentence_similarity(doc1_text, doc2_text, model, threshold=0.8)
        
        print(f"Jami {len(similar_sentences)} ta o'xshash jumla topildi.")
        
        # Top 5 o'xshash jumlalarni ko'rsatish
        if similar_sentences:
            print("\nEng yuqori o'xshashlikka ega jumlalar:")
            similar_sentences.sort(key=lambda x: x['similarity'], reverse=True)
            for i, sim in enumerate(similar_sentences[:5]):
                print(f"\n{i+1}. O'xshashlik: {sim['similarity']*100:.1f}%")
                print(f"   1-hujjat: \"{sim['doc1_sentence'][:100]}...\"")
                print(f"   2-hujjat: \"{sim['doc2_sentence'][:100]}...\"")
    
    print(f"\n{'='*50}")
    print("Tekshiruv yakunlandi.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()