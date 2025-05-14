import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os
import fitz  # PyMuPDF

# --- Modelni yuklash ---
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

# --- PDF dan matn olish funksiyasi ---
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                full_text.append(text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"❌ Xato: PDF o‘qishda muammo: {str(e)}")
        return ""

# --- AI ehtimolini aniqlash funksiyasi ---
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

# --- Asosiy funksiya ---
def main(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"❌ Fayl topilmadi: {pdf_path}")
        return

    print(f"📄 PDF fayldan matn olinmoqda: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    print("🧠 AI yozgan bo‘lish ehtimoli aniqlanmoqda...")
    result = detect_ai_generated(text)

    print("\n--- Natija ---")
    print(f"👤 Inson tomonidan yozilgan ehtimoli: {result['human_score'] * 100:.2f}%")
    print(f"🤖 AI tomonidan yozilgan ehtimoli: {result['ai_score'] * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF matn AI yozilganmi?")
    parser.add_argument("--pdf", required=True, help="PDF fayl yo‘li")
    args = parser.parse_args()

    main(args.pdf)
