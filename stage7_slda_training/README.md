# Stage 7 — SLDA: Sequential Legal Domain Adaptation (A100-Optimized)

This Script contains a complete training script for Sequential Legal Domain Adaptation (SLDA) using a SNLI-finetuned LegalBERT model and CPRA article-specific NLI datasets.

The script trains:

1. Stage-1 (combined CPRA adaptation) – one model on the union of Articles 2, 3, 7, and 8.  
2. Stage-2 (article specialization) – one model per article, initialized from Stage-1.  

It also evaluates all models across all article validation sets and writes the results to a CSV.

---

## 1. Script Overview

The main script (e.g. `slda_train.py`) does the following:

1. Installs and imports required libraries (`transformers`, `evaluate`, `torch`, `pandas`, etc.).
2. Enables A100-specific speed optimizations (TF32, BF16, `torch.compile`).
3. Loads a SNLI-finetuned LegalBERT checkpoint.
4. Loads four CPRA article NLI datasets from CSV.
5. Normalizes labels into a canonical mapping `{entailment, contradiction, neutral}` → `{0, 1, 2}`.
6. Trains a **Stage-1 combined model** with class-weighted cross-entropy.
7. Evaluates the Stage-1 model on each article’s validation set.
8. For each article:
   - Reloads the Stage-1 checkpoint  
   - Trains a specialized **Stage-2 model** on that article  
   - Evaluates it on all article validation sets  
9. Saves all metrics into `slda_results.csv` and all checkpoints under `OUT_DIR`.

---

## 2. Inputs and Paths

At the top of the script:

```python
BASE_CKPT = "/mnt/gdrive/MyDrive/legalbert-snli-finetuned"

ARTICLE_FILES: Dict[str, str] = {
    "Article 2": "/mnt/gdrive/MyDrive/NLI_Results/article2_nli_semantic_pairs_labeled.csv",
    "Article 3": "/mnt/gdrive/MyDrive/NLI_Results/article3_nli_semantic_pairs_labeled.csv",
    "Article 7": "/mnt/gdrive/MyDrive/NLI_Results/article7_nli_semantic_pairs_labeled.csv",
    "Article 8": "/mnt/gdrive/MyDrive/NLI_Results/article8_nli_semantic_pairs_labeled.csv",
}

OUT_DIR = "/mnt/gdrive/MyDrive/SLDA_checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

All checkpoints are written to:
/mnt/gdrive/MyDrive/SLDA_checkpoints

## 3. Hyperparameters and A100 Knobs
SEED          = 42
VAL_FRAC      = 0.1
STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 3
LR_STAGE1     = 1e-5
LR_STAGE2     = 1e-5
BATCH_TRAIN   = 32
BATCH_EVAL    = 64
MAX_LEN       = 256
ARTICLE_ORDER = ["Article 2", "Article 3", "Article 7", "Article 8"]

GPU optimizations:
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
USE_BF16 = torch.cuda.is_available()

---
## 4. Dataset Class and Metrics
Pair dataset
The class PairDS wraps each dataframe into a PyTorch dataset:
class PairDS(Dataset):
    def __init__(self, df, tokenizer, label2id, max_len=256):
        self.prem = df["premise"].astype(str).tolist()
        self.hyp  = df["hypothesis"].astype(str).tolist()
        self.y    = [label2id[normalize_label(v)] for v in df["label"].tolist()]
        self.tok  = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        enc = self.tok(self.prem[i], self.hyp[i],
                       truncation=True, max_length=self.max_len)
        enc["labels"] = self.y[i]
        return {k: torch.tensor(v) for k, v in enc.items()}


Metrics

Using evaluate:

Overall accuracy

f1_macro (macro-F1 across 3 classes)

Per-class F1: f1_entailment, f1_contradiction, f1_neutral

The make_metrics() factory builds the compute_metrics function passed to Trainer.

## 5. Class-Weighted Trainer

The script defines a custom WeightedTrainer:
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kw):
        super().__init__(*args, **kw)
        self.class_weights = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=None if self.class_weights is None else self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


Class weights are calculated from the training split via class_weights_from_df(), using inverse label frequency to address the neutral-heavy imbalance.
---
## 6. Stage-1: Combined CPRA Adaptation

1.Load all article CSVs and concatenate.
2.Normalize labels and shuffle.
3.Split into train and validation (90/10).
4.Compute class weights.
5.Wrap into PairDS.
6.Train WeightedTrainer with:
out_stage1 = os.path.join(OUT_DIR, "stage1_combined")
args1 = fast_args(out_stage1, STAGE1_EPOCHS, LR_STAGE1)
trainer1 = WeightedTrainer(**build_trainer_kwargs(base_kwargs1))
trainer1.train()
trainer1.save_model(out_stage1)
tokenizer.save_pretrained(out_stage1)

The Stage-1 checkpoint is saved under:
SLDA_checkpoints/stage1_combined/
---

## 7. Stage-1 Snapshot Evaluation
The helper eval_on_all(model):
.Builds a validation set for each article.
.Runs the model on each article’s validation slice.
.Returns per-article accuracy and macro-F1.
The script calls:
m_stage1 = AutoModelForSequenceClassification.from_pretrained(stage1_model_path).to(device)
stage1_snap = eval_on_all(m_stage1)
print("Stage-1 snapshot:", stage1_snap)

This gives how the combined Stage-1 model performs on each article individually.
---
## 8. Stage-2: Article-Specific Specialization
For each art in ARTICLE_ORDER:

1.Load that article’s CSV and do a fresh 90/10 split.
2. class weights from the article’s training split.
3.Load the Stage-1 checkpoint as the starting model.
4.Train with the same WeightedTrainer for STAGE2_EPOCHS, LR=LR_STAGE2.
5.Save the specialized checkpoint to:
SLDA_checkpoints/stage2_Article_2/
SLDA_checkpoints/stage2_Article_3/
SLDA_checkpoints/stage2_Article_7/
SLDA_checkpoints/stage2_Article_8/
6.Evaluate each specialized model on all article validation sets using eval_on_all.
Append all metrics to a list all_results.
This creates a full cross-article evaluation matrix: “model trained on article X, evaluated on article Y”.
---
## 9. Metrics CSV
At the end:
res_df = pd.DataFrame(all_results)
res_path = os.path.join(OUT_DIR, "slda_results.csv")
res_df.to_csv(res_path, index=False)
print("Saved metrics to:", res_path)
print("Done. Checkpoints in:", OUT_DIR)

The file slda_results.csv contains one row per evaluation:
.stage – e.g., Stage2_Article 3
.article_model – which article the model was trained on
.eval_on – which article’s validation set it was evaluated on
.acc – accuracy
.f1_macro – macro-F1

---

## Requriments

```python
!pip install -q transformers datasets evaluate accelerate

import os, random, inspect
import pandas as pd
import numpy as np
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
import evaluate
