
# CPRA–LegalBERT Sequential Transfer Learning (SLDA Pipeline)

### Mukesh Yadav  
Department of Applied Data Science  
Clarkson University, Potsdam, USA  
📧 yadavm@clarkson.edu  

### Shafique A. Chaudhry  
David D. Reh School of Business  
Clarkson University, Potsdam, USA  
📧 schaudhr@clarkson.edu  

---

#  Abstract
This repository presents a complete research pipeline for detecting CPRA compliance violations in privacy policies using Sequential Legal Domain Adaptation (SLDA), combining LegalBERT, SNLI pretraining, GPT‑generated NLI pairs, SBERT clustering, FAISS retrieval, and multi-stage fine-tuning across CPRA articles.
  
📄 The full research PDF is available here:

## 📄 Download Research Paper (Tracked)

You can download the full research paper here:

 **https://bit.ly/cpra-paper**




---

# 📂 Repository Structure
```
cpra-legalbert-sequential-transfer/
│
├── Docs/
│   └── _Enhancing_CPRA_Compliance_Detection_Using_LegalBERT_and_NLI_Style_Inference_.pdf
│  
│
├── stage1_legalbert_snli/
│   ├── README.md
│   └── legalbert_snli_fine_tuning.py
│
├── stage2_opp115_preprocessing/
│   ├── README.md
│   ├── extract_opp115_policies_py.py
│   └── opp115_all_policies_combined (3).csv
│
├── stage3_cppa_extraction/
│   ├── README.md
│   ├── article_2_full (1).txt
│   ├── article_3_full.txt
│   ├── article_7_full.txt
│   ├── article_8_full.txt
│   └── extract_cppa_articles.py
│
├── stage4_gpt_nli_labeling/
│   ├── README.md
│   ├── article2_nli_semantic_pairs_faiss.csv
│   ├── article2_nli_semantic_pairs_labeled.csv
│   ├── article3_nli_semantic_pairs_faiss.csv
│   ├── article3_nli_semantic_pairs_labeled.csv
│   ├── article7_nli_semantic_pairs_faiss.csv
│   ├── article7_nli_semantic_pairs_labeled.csv
│   ├── article8_nli_semantic_pairs_faiss.csv
│   ├── article8_nli_semantic_pairs_labeled.csv
│   └── stage4_gpt_nli_labeling.py
│
├── stage5_dataset_stats/
│   ├── README.md
│   ├── article_row_counts_gray_singlecol (1).png
│   ├── class_distribution_grouped (1).png
│   └── stage_5__dataset_size_and_label_composition.py
│
├── stage6_external_validation/
│   ├── README.md
│   ├── confusion_matrix_snli_ieee (1).png
│   └── stage_6__external_validation_using_snli.py
│
├── stage7_slda_training/
│   ├── README.md
│   ├── config.json
│   ├── slda_train.py
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── stage8_slda_results/
│   ├── README.md
│   ├── slda_f1_grouped_research (1).png
│   ├── slda_results.csv
│   └── stage8_slda_results.py
│
├── .gitignore
├── LICENSE
└── README.md


```

---

#  SLDA: Sequential Legal Domain Adaptation  

SLDA is a two-stage adaptation strategy:

### **Stage‑1: Broad Legal Adaptation**
Train LegalBERT on the combined dataset of CPRA Articles 2, 3, 7, and 8.

### **Stage‑2: Article‑Specific Specialization**
Fine‑tune separate models for Article 2/3/7/8 starting from the Stage‑1 backbone.

More detailed theory is in the full README version previously generated.

---

#  Results Summary
Macro‑F1 improvements after Stage‑2 fine‑tuning:

| Article | Stage‑1 F1 | Stage‑2 F1 | Gain |
|---------|------------|------------|------|
| A2 | 0.740 | 0.764 | +0.024 |
| A3 | 0.484 | 0.778 | +0.294 |
| A7 | 0.712 | 0.791 | +0.080 |
| A8 | 0.901 | 0.928 | +0.027 |

---


##  Citation

If you use this repository, please cite:

**Yadav, M., & Chaudhry, S. A. (2025). _Enhancing CPRA Compliance Detection Using LegalBERT and NLI-Style Inference_.  
Clarkson University, Potsdam, USA.**

Mukesh Yadav  
Department of Applied Data Science  
Clarkson University, Potsdam, USA  
Email: yadavm@clarkson.edu  

Shafique A. Chaudhry  
David D. Reh School of Business  
Clarkson University, Potsdam, USA  
Email: schaudhr@clarkson.edu  

GitHub Repository: https://github.com/mukeshyadavm/cpra-legalbert-sequential-transfer


---
  
