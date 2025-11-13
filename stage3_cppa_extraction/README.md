# CPPA Article Extraction Script

This script (`extract_cppa_articles.py`) automates the extraction of specific Articles (2, 3, 7, and 8)from the *California Consumer Privacy Act (CPPA) Regulations* PDF document.  
It uses the PyMuPDF (`fitz`) library to locate section markers (e.g., “§7010”, “§7020”) and extract all text between them into clean, structured `.txt` files.

---

##  Purpose
The goal of this script is to create machine-readable text files for legal NLP tasks such as:
- Legal clause classification and retrieval  
- Fine-tuning LegalBERT  models  
- CPRA/CCPA compliance detection using Natural Language Inference (NLI)  
- Policy–regulation alignment studies  

---

##  How It Works
1. Opens the official CPPA Regulations PDF .  
2. Defines start and end markers for each target article:  
   - Article 2 → (§7010 – §7020)  
   - Article 3 → (§7020 – §7050)  
   - Article 7 → (§7050 – §7100)  
   - Article 8 → (§7100 – §7300)  
3. Reads the text of each page line by line.  
4. Extracts content that falls between the given markers.  
5. Saves each extracted section into a separate text file.  

---

##  Output
The following files are generated after running the script:
article_2_full.txt
article_3_full.txt
article_7_full.txt
article_8_full.txt


##  Requirements
Ensure the following dependencies are installed:

```bash
python>=3.9
PyMuPDF>=1.22.0   # Provides the 'fitz' module
re
