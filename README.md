# LegalBERT SNLI Fine-Tuning

This notebook (`LegalBERT_SNLI_fine_tuning.ipynb`) performs the first stage of the Sequential Transfer Learning pipeline for CPRA compliance detection.

##  Purpose
Fine-tune **LegalBERT** on the **SNLI (Stanford Natural Language Inference)** dataset to teach the model:
- Entailment
- Contradiction
- Neutral reasoning

This stage builds general NLI capability before adapting the model to CPRA-specific legal clauses.

##  Workflow
1. Load SNLI dataset  
2. Preprocess & tokenize using LegalBERT  
3. Train LegalBERT on 3-way NLI labels  
4. Evaluate model performance  
5. Save fine-tuned model weights for next training stage



##  How to Run
1. Open the notebook in VS Code or Google Colab  
2. Select GPU runtime (recommended)  
3. Run all cells in sequence  
4. Save resulting model weights

##  Output
The notebook produces:
- Fine-tuned LegalBERT SNLI model  
- Accuracy/loss metrics

##  Requirements
### Environment
- **Python ≥ 3.10**
- **GPU:** NVIDIA T4/A100 (recommended)
- **RAM:** ≥ 12 GB
- **Virtual environment:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate   # macOS/Linux
  venv\Scripts\activate      # Windows
