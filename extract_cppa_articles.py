

import fitz
import re

#  CPPA Regulations PDF (Colab Drive mounted at /mnt/gdrive)
pdf_path = "/mnt/gdrive/MyDrive/cppa_regs.pdf"
doc = fitz.open(pdf_path)

#  Define article extraction ranges (from TOC)
article_ranges = {
    "article_2": ("§ 7010", "§ 7020"),
    "article_3": ("§ 7020", "§ 7050"),
    "article_7": ("§ 7050", "§ 7100"),
    "article_8": ("§ 7100", "§ 7300"),
}

def extract_article_text(doc, start_marker, end_marker):
    extracting = False
    extracted_text = ""
    for page in doc:
        text = page.get_text()
        for line in text.split('\n'):
            if start_marker in line:
                extracting = True
            if extracting:
                extracted_text += line + '\n'
            if extracting and end_marker in line:
                extracting = False
    return extracted_text.strip()

#  Extract and save each article
for name, (start, end) in article_ranges.items():
    text = extract_article_text(doc, start, end)
    with open(f"{name}_full.txt", "w", encoding='utf-8') as f:
        f.write(text)

print(" Articles 2, 3, 7, and 8 extracted and saved as article_2_full.txt, article_3_full.txt, article_7_full.txt, article_8_full.txt")
