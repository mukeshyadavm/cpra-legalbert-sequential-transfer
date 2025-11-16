# SLDA --- Sequential Legal Domain Adaptation (A100-Optimized)

This repository provides a complete implementation of **Sequential Legal
Domain Adaptation (SLDA)** using a SNLI-finetuned **LegalBERT** model
and CPRA article-specific NLI datasets.\
The approach adapts a base LegalBERT model to CPRA regulations in two
sequential stages:

-   **Stage-1 (Combined CPRA Adaptation):** Train one model on Articles
    **2, 3, 7, 8** combined.\
-   **Stage-2 (Article Specialization):** Train one specialized model
    **per article**, initialized from Stage-1.

All models are evaluated across **all** CPRA article validation sets,
producing a full cross-article evaluation matrix saved as a CSV.

------------------------------------------------------------------------

## 1. Overview

This project includes:

-   Full A100-optimized training script (`slda_train.py`)
-   Custom weighted trainer (for class imbalance)
-   Article-specific dataset loaders
-   Evaluation utilities
-   Checkpoint saving for all models
-   CSV summarizing all metrics

The pipeline uses:

-   **LegalBERT (SNLI-Finetuned)**
-   **CPRA Article NLI datasets**
-   **A100 performance knobs (TF32, BF16)**\
-   **HuggingFace Trainer + PyTorch**

------------------------------------------------------------------------

## 2. Inputs and Paths

``` python
BASE_CKPT = "/mnt/gdrive/MyDrive/legalbert-snli-finetuned"

ARTICLE_FILES = {
    "Article 2": "/mnt/gdrive/MyDrive/NLI_Results/article2_nli_semantic_pairs_labeled.csv",
    "Article 3": "/mnt/gdrive/MyDrive/NLI_Results/article3_nli_semantic_pairs_labeled.csv",
    "Article 7": "/mnt/gdrive/MyDrive/NLI_Results/article7_nli_semantic_pairs_labeled.csv",
    "Article 8": "/mnt/gdrive/MyDrive/NLI_Results/article8_nli_semantic_pairs_labeled.csv",
}

OUT_DIR = "/mnt/gdrive/MyDrive/SLDA_checkpoints"
```

All model checkpoints are stored here:

    /mnt/gdrive/MyDrive/SLDA_checkpoints

------------------------------------------------------------------------

## 3. Hyperparameters and A100 Knobs

``` python
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
```

### GPU Acceleration

``` python
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
USE_BF16 = torch.cuda.is_available()
```

------------------------------------------------------------------------

## 4. Dataset Class & Metrics

### Pair Dataset

``` python
class PairDS(Dataset):
    def __init__(self, df, tokenizer, label2id, max_len=256):
        self.prem = df["premise"].astype(str).tolist()
        self.hyp  = df["hypothesis"].astype(str).tolist()
        self.y    = [label2id[normalize_label(v)] for v in df["label"].tolist()]
        self.tok  = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        enc = self.tok(self.prem[i], self.hyp[i],
                       truncation=True, max_length=self.max_len)
        enc["labels"] = self.y[i]
        return {k: torch.tensor(v) for k, v in enc.items()}
```

### Metrics

-   **accuracy**
-   **macro-F1**
-   per-class-F1
    -   entailment\
    -   contradiction\
    -   neutral

Metrics are computed using a custom `make_metrics()` factory.

------------------------------------------------------------------------

## 5. Class-Weighted Trainer

``` python
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kw):
        super().__init__(*args, **kw)
        self.class_weights = None if class_weights is None else torch.tensor(
            class_weights, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=None if self.class_weights is None else self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
```

------------------------------------------------------------------------

## 6. Stage-1: Combined CPRA Adaptation

### Steps

1.  Load all article CSVs.
2.  Concatenate & shuffle.
3.  Normalize labels.
4.  Train/val split (90/10).
5.  Compute class weights.
6.  Train Stage-1 model.

------------------------------------------------------------------------

## 7. Stage-1 Evaluation

``` python
m_stage1 = AutoModelForSequenceClassification.from_pretrained(stage1_model_path).to(device)
stage1_snap = eval_on_all(m_stage1)
```

------------------------------------------------------------------------

## 8. Stage-2: Article Specialization

For each article:

1.  Load dataset\
2.  Compute class weights\
3.  Load Stage-1 checkpoint\
4.  Train specialized model\
5.  Evaluate on all articles\
6.  Append results to CSV

------------------------------------------------------------------------

## 9. Metrics CSV

Saved as:

    slda_results.csv

Contains:

-   stage\
-   article_model\
-   eval_on\
-   acc\
-   f1_macro

------------------------------------------------------------------------

## 10. Requirements

### Install

``` bash
pip install -q transformers datasets evaluate accelerate
```

### Imports

``` python
import os, random, inspect
import pandas as pd
import numpy as np
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import evaluate
```

------------------------------------------------------------------------
