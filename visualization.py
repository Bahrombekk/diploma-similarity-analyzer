# visualization.py
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Natijalarni vizualizatsiya qilish
def visualize_similarity(similarities, output_dir="./plagiarism_report"):
    # Hisobot papkasini yaratish
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Boblar bo'yicha o'xshashlik diagrammasi
    chapter_nums = list(similarities.keys())
    chapter_labels = []
    
    for ch in chapter_nums:
        if ch == "0":
            chapter_labels.append("Kirish")
        else:
            chapter_labels.append(f"{ch}-bob")
    
    tfidf_values = [similarities[ch]["tfidf"] * 100 for ch in chapter_nums]
    neural_values = [similarities[ch]["neural"] * 100 for ch in chapter_nums]
    overall_values = [similarities[ch]["overall"] * 100 for ch in chapter_nums]
    
    # Bob-bob bo'yicha o'xshashlik darajalari
    plt.figure(figsize=(12, 8))
    x = np.arange(len(chapter_labels))
    width = 0.25
    
    plt.bar(x - width, tfidf_values, width, label='TF-IDF', color='skyblue')
    plt.bar(x, neural_values, width, label='Semantik', color='lightgreen')
    plt.bar(x + width, overall_values, width, label='Umumiy', color='salmon')
    
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.7)  # Plagiat chegarasi
    
    plt.xlabel('Boblar')
    plt.ylabel('O\'xshashlik foizi (%)')
    plt.title('Boblar bo\'yicha o\'xshashlik darajalari')
    plt.xticks(x, chapter_labels)
    plt.legend()
    plt.ylim(0, 100)
    
    for i, v in enumerate(overall_values):
        plt.text(i + width/2, v + 2, f'{v:.1f}%', ha='center')
    
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{output_dir}/chapter_similarity_chart_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    # Heatmap - boblar o'xshashligi
    plt.figure(figsize=(10, 8))
    data = np.array(overall_values).reshape(1, -1)
    df = pd.DataFrame(data, columns=chapter_labels, index=["O'xshashlik (%)"])
    
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=.5, cbar_kws={'label': 'O\'xshashlik %'})
    plt.title('Boblar o\'xshashligi (foizlarda)')
    plt.savefig(f"{output_dir}/chapter_heatmap_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')
    return f"{output_dir}/chapter_similarity_chart_{timestamp}.png"

# Hisobot yaratish
def generate_report(doc1_name, doc2_name, similarities, chapter_paragraph_sim, output_dir="./plagiarism_report"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{output_dir}/chapter_plagiarism_report_{timestamp}.html"
    
    # Umumiy o'xshashlik hisoblash
    chapter_nums = list(similarities.keys())
    
    # Boblar uchun jadval yaratish
    chapters_table = "<h2>Boblar bo'yicha o'xshashlik:</h2>"
    chapters_table += "<table border='1' style='width:100%; border-collapse: collapse;'>"
    chapters_table += "<tr><th>Bob</th><th>TF-IDF</th><th>Semantik</th><th>Umumiy</th><th>O'xshash paragraflar</th><th>Plagiat darajasi</th></tr>"
    
    total_overall = 0
    for ch in chapter_nums:
        ch_name = "Kirish" if ch == "0" else f"{ch}-bob"
        tfidf = similarities[ch]["tfidf"] * 100
        neural = similarities[ch]["neural"] * 100
        overall = similarities[ch]["overall"] * 100
        total_overall += overall
        
        # Plagiat darajasi tasnifi
        plagiarism_level = "Past"
        level_color = "green"
        if overall > 70:
            plagiarism_level = "Juda yuqori"
            level_color = "red"
        elif overall > 50:
            plagiarism_level = "Yuqori"
            level_color = "orange"
        elif overall > 30:
            plagiarism_level = "O'rta"
            level_color = "yellow"
        
        sim_paracount = len(chapter_paragraph_sim.get(ch, []))
        
        chapters_table += f"""
        <tr>
            <td>{ch_name}</td>
            <td>{tfidf:.1f}%</td>
            <td>{neural:.1f}%</td>
            <td>{overall:.1f}%</td>
            <td>{sim_paracount}</td>
            <td style='color:{level_color};'>{plagiarism_level}</td>
        </tr>
        """
    
    chapters_table += "</table>"
    
    # Umumiy o'xshashlik foizi
    if chapter_nums:
        avg_overall = total_overall / len(chapter_nums)
    else:
        avg_overall = 0
    
    # O'xshashlik darajasi tasnifi
    overall_level = "Past"
    if avg_overall > 70:
        overall_level = "Juda yuqori"
    elif avg_overall > 50:
        overall_level = "Yuqori"
    elif avg_overall > 30:
        overall_level = "O'rta"
    
    # Progress bar rangini aniqlash
    if avg_overall > 70:
        progress_color = "#ff4d4d"
    elif avg_overall > 50:
        progress_color = "#ffa64d"
    elif avg_overall > 30:
        progress_color = "#ffff4d"
    else:
        progress_color = "#4dff4d"
    
    # Har bir bob uchun o'xshash paragraflar jadvalini yaratish
    chapter_paragraphs_html = ""
    for ch in chapter_nums:
        ch_name = "Kirish" if ch == "0" else f"{ch}-bob"
        if ch in chapter_paragraph_sim and chapter_paragraph_sim[ch]:
            chapter_paragraphs_html += f"<h3>{ch_name} o'xshash paragraflar:</h3>"
            chapter_paragraphs_html += "<table border='1' style='width:100%; border-collapse: collapse;'>"
            chapter_paragraphs_html += "<tr><th>№</th><th>1-hujjat paragrafi</th><th>2-hujjat paragrafi</th><th>O'xshashlik</th></tr>"
            
            for i, item in enumerate(chapter_paragraph_sim[ch]):
                chapter_paragraphs_html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{item['doc1_paragraph'][:200]}...</td>
                    <td>{item['doc2_paragraph'][:200]}...</td>
                    <td>{item['similarity']*100:.1f}%</td>
                </tr>
                """
            
            chapter_paragraphs_html += "</table>"
    
    # Hisobot HTML shaklida
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Boblar bo'yicha plagiat tekshiruvi hisoboti</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; margin-top: 20px; }}
            h3 {{ color: #2980b9; margin-top: 15px; }}
            .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .warning {{ background-color: #ffe0e0; padding: 10px; border-radius: 5px; margin: 15px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .progress-container {{ width: 100%; background-color: #e0e0e0; border-radius: 5px; }}
            .progress-bar {{ height: 24px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Boblar bo'yicha plagiat tekshiruvi hisoboti</h1>
        <p><strong>Tekshirilgan sana:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        <p><strong>1-hujjat nomi:</strong> {doc1_name}</p>
        <p><strong>2-hujjat nomi:</strong> {doc2_name}</p>
        
        <div class="summary">
            <h2>Umumiy natijalar:</h2>
            <p><strong>Umumiy o'xshashlik darajasi:</strong> {avg_overall:.2f}%</p>
            <p><strong>Plagiat darajasi:</strong> {overall_level}</p>
            
            <div class="progress-container">
                <div class="progress-bar" style="width: {avg_overall}%; background-color: {progress_color};">{avg_overall:.1f}%</div>
            </div>
        </div>
        
        {"<div class='warning'><strong>Ogohlantirish:</strong> Ushbu ikki hujjat orasida sezilarli darajada o'xshashlik aniqlandi.</div>" if avg_overall > 50 else ""}
        
        {chapters_table}
        
        {chapter_paragraphs_html}
        
        <p><em>Eslatma: Ushbu hisobot avtomatik tahlil natijasi hisoblanadi va mutlaq haqiqat emas. Yakuniy qaror qabul qilish uchun mutaxassis ko'rigi tavsiya etiladi.</em></p>
    </body>
    </html>
    """
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return report_file