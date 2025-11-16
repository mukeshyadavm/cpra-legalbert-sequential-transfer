
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

The full research PDF is available here:  
📄 **[Download Research Paper](docs/_Enhancing_CPRA_Compliance_Detection_Using_LegalBERT_and_NLI_Style_Inference_.pdf)**

---

# 📂 Repository Structure
```
cpra-legalbert-sequential-transfer/
│
├── stage1_legalbert_snli/
├── stage2_opp115_preprocessing/
├── stage3_cppa_extraction/
├── stage4_gpt_nli_labeling/
├── stage5_dataset_stats/
├── stage6_external_validation/
├── stage7_slda_training/
├── stage8_slda_results/
└── docs/
    └── _Enhancing_CPRA_Compliance_Detection_Using_LegalBERT_and_NLI_Style_Inference_.pdf
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

#  Citation
If you use this repository, consider citing:

```
Yadav, M., & Chaudhry, S. A. (2025). Enhancing CPRA Compliance Detection Using LegalBERT and NLI-Style Inference.
Clarkson University.
```

---
  
