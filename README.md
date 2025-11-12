
This script, `extract_opp115_policies.py`, aggregates privacy policy text from the OPP-115 dataset.  
It combines policy content from multiple CSV files into a single, structured dataset for downstream NLP or compliance analysis.

---

##  Purpose
The goal is to consolidate all privacy policy statements (from the 4th column of each OPP-115 CSV) into a single file that can be used for:
- LegalBERT 
- Natural Language Inference (NLI) model fine-tuning  
- Privacy compliance classification tasks  

---

##  How It Works
1. Reads all `.csv` files in the specified directory (`/mnt/gdrive/MyDrive/OPP-115/pretty_print`).  
2. Checks each file for at least 4 columns.  
3. Extracts policy text from column 4 (index 3).  
4. Skips files with fewer than 4 columns.  
5. Combines all policies into a single Pandas DataFrame.  
6. Saves the merged data as a unified CSV file.

---

##  Output
After execution, a single CSV file is generated:





##  Requirements

Make sure the following dependencies are installed:

```bash
python>=3.9
pandas>=2.0.0
numpy
tqdm
pip install -r requirements.txt
