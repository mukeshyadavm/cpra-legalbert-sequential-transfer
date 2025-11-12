# LegalBERT SNLI Fine-Tuning

This notebook (`LegalBERT_SNLI_fine_tuning.ipynb`) performs the first stage of the Sequential Transfer Learning pipeline for CPRA compliance detection.

##  Purpose
Fine-tune LegalBERT on the SNLI (Stanford Natural Language Inference) dataset to teach the model:
- Entailment  
- Contradiction  
- Neutral reasoning  

This stage builds general NLI capability before adapting the model to CPRA-specific legal clauses.

---

##  Workflow
1. Load SNLI dataset  
2. Preprocess & tokenize using LegalBERT  
3. Train LegalBERT on 3-way NLI labels  
4. Evaluate model performance  
5. Save fine-tuned model weights for the next training stage  

---

##  Requirements
Make sure the following dependencies are installed before running the notebook or script:

```bash
# Python version
python>=3.9

# Core libraries
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
tokenizers>=0.13.3
numpy
pandas
scikit-learn
tqdm
matplotlib
seaborn

# Optional (for Colab or Drive mounting)
google-colab
huggingface_hub


