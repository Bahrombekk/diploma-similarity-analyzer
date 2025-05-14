import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from docx import Document
import argparse
import os

# --- AI detection modeli yuklanadi ---
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

# --- Word fayldan matn ajratish funksiyasi ---
def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        full_text = []

        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)

        return "\n".join(full_text)
    except Exception as e:
        print(f"❌ Xato: Word faylni o'qishda muammo: {str(e)}")
        return ""

# --- AI yozgan bo'lish ehtimolini aniqlovchi funksiya ---
def detect_ai_generated(text):
    if not text.strip():
        return {"human_score": 0.0, "ai_score": 0.0}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        return {
            "human_score": probs[0][0].item(),
            "ai_score": probs[0][1].item()
        }

# --- Asosiy bajaruvchi funksiya ---
def main(docx_path):
    if not os.path.exists(docx_path):
        print(f"❌ Fayl topilmadi: {docx_path}")
        return

    print(f"📄 Word fayldan matn olinmoqda: {docx_path}")
    text = extract_text_from_docx(docx_path)

    print("🧠 AI yozgan bo‘lish ehtimoli aniqlanmoqda...")
    result = detect_ai_generated(text)

    print("\n--- Natija ---")
    print(f"👤 Inson tomonidan yozilgan ehtimoli: {result['human_score'] * 100:.2f}%")
    print(f"🤖 AI tomonidan yozilgan ehtimoli: {result['ai_score'] * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI yozgan matnni aniqlash")
    parser.add_argument("--docx", required=True, help="Word (.docx) fayl yo‘li")
    args = parser.parse_args()

    main(args.docx)
