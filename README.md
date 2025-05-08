# 🧠 Plagiarism Checker (Uzbek & English)

A powerful AI-powered tool for detecting potential plagiarism in academic documents using both traditional NLP techniques (TF-IDF) and deep semantic analysis (Sentence Transformers).

---

## 📌 Overview

This project provides a multi-stage plagiarism detection pipeline that analyzes `.docx`-formatted academic documents such as theses and dissertations. It evaluates both surface-level (lexical) and deep semantic similarity to ensure comprehensive assessment.

Key features include:
- Extraction and cleaning of DOCX text content
- Stopword removal (supports Uzbek and English)
- TF-IDF based similarity analysis
- SentenceTransformer-based deep semantic comparison
- Sentence- and paragraph-level alignment
- Visual reports with graphs, histograms, and heatmaps
- HTML report generation for detailed analysis

---

## ⚙️ Technologies Used

- Python 🐍
- [NLTK](https://www.nltk.org/) – tokenization & stopword handling
- [Sentence Transformers](https://www.sbert.net/) – semantic similarity
- [scikit-learn](https://scikit-learn.org/) – TF-IDF Vectorizer
- [matplotlib & seaborn](https://matplotlib.org/) – visualization
- [docx](https://python-docx.readthedocs.io/en/latest/) – Word document parsing
- tqdm, NumPy, pandas – processing & statistics

---

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/plagiarism-checker-uz-en.git
cd plagiarism-checker-uz-en
Create virtual environment & install dependencies:

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
requirements.txt includes:

python-docx

nltk

sentence-transformers

scikit-learn

matplotlib

seaborn

tqdm

pandas

Download NLTK resources (first run only):

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
📝 Usage
bash
Copy
Edit
python plagiarism_checker.py --doc1 path/to/first.docx --doc2 path/to/second.docx --language uzbek --threshold 0.7
Options:

--language: uzbek or english

--threshold: Similarity threshold (default is 0.7)

--output: Custom output directory (default: ./plagiarism_report)

📊 Example Output
Summary of overall, TF-IDF, and semantic similarity percentages

Bar charts and histograms

HTML report with highlighted similar paragraphs and sentences

Heatmaps showing paragraph-to-paragraph matching

📁 Output sample:

Copy
Edit
plagiarism_report/
│
├── plagiarism_report_YYYYMMDD_HHMMSS.html
├── similarity_chart_YYYYMMDD_HHMMSS.png
├── paragraph_similarity_YYYYMMDD_HHMMSS.png
├── sentence_similarity_YYYYMMDD_HHMMSS.png
└── paragraph_heatmap_YYYYMMDD_HHMMSS.png
🎓 Academic Context
This tool was developed to support academic integrity by enabling automatic detection of potential plagiarism in higher education. It is especially useful for:

Thesis supervision

Quality assurance teams

Journal reviewers

Educators and researchers

🏷 License
MIT License. You are free to use, modify, and distribute with attribution.

✉️ Contact
Developed by [Your Name]
📧 your.email@example.com
🔗 LinkedIn • GitHub

