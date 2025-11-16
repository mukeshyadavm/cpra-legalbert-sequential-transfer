## VIII. Sequential Legal Domain Adaptation (SLDA): Theory, Setup, and Results

This section summarizes the motivation, methodology, and empirical outcomes of our
two-stage Sequential Legal Domain Adaptation (SLDA) pipeline for CPRA-based legal
Natural Language Inference (NLI). The goal is to adapt a general NLI model to the
legal domain and then specialize it to specific CPRA articles without catastrophic
forgetting.

---

### a) Problem Framing

We cast legal NLI as a **3-way classification** problem:

- entailment (0)
- contradiction (1)
- neutral (2)

with paired text inputs *(premise p, hypothesis h)*.

Our objective is:

1. Adapt a general NLI model to **legal language**
2. Specialize it to **each CPRA article**
3. Avoid **catastrophic forgetting**

We implement a **two-stage sequential adaptation** scheme:

- **Stage-1:** Train on the combined CPRA Articles **2, 3, 7, 8**
- **Stage-2:** Fine-tune a separate article-specific classifier initialized from Stage-1

---

### b) Base Model and Label Normalization

All experiments use the same base checkpoint:

```
/mnt/gdrive/MyDrive/legalbert-snli-finetuned
```

derived from LegalBERT fine-tuned on SNLI.

Canonical label mapping:

- entailment → 0
- contradiction → 1
- neutral → 2

---

### c) Objective and Class Imbalance

We apply **class-weighted cross-entropy** to counter heavy skew toward neutral.

Stage-1 class counts:

- entailment: 856
- contradiction: 1610
- neutral: 14430

Weights:

```
[1.886, 1.003, 0.112]
```

---

### d) Data Splits and Schedule

**Stage-1**
- Concatenate all articles
- Stratified 90/10 train/validation split

**Stage-2**
- Fresh 90/10 split per article
- Initialization from Stage-1 checkpoint

**Hyperparameters (shared)**  
3 epochs, lr=1e-5, batch size 32/64, max length=256, AdamW (fused), warmup=0.06  
A100 optimizations: **BF16 + torch.compile**

---

### e) Metrics

- Accuracy  
- Macro-F1  
- Per-class F1 (entailment/contradiction/neutral)

Macro-F1 is emphasized due to imbalance.

---

### f) Stage-1 → Stage-2 Outcomes

**Stage-1 snapshot per article**

| Article | Acc | Macro-F1 |
|--------|------|-----------|
| A2 | 0.913 | 0.740 |
| A3 | 0.853 | 0.484 |
| A7 | 0.914 | 0.712 |
| A8 | 0.955 | 0.901 |

**Best Stage-2 model per article**

| Article | Acc | Macro-F1 |
|--------|------|-----------|
| A2 | 0.920 | 0.764 |
| A3 | 0.930 | 0.778 |
| A7 | 0.942 | 0.791 |
| A8 | 0.959 | 0.928 |

**Largest improvement:** Article 3 Macro-F1: **0.484 → 0.778 (+0.294)**

---

### 📊 Macro-F1 Comparison Plot (Stage-1 vs Stage-2)

<img src="slda_f1_grouped_research (1).png" width="420"/>

---

### g) Why Stage-2 Helps

- Stage-1 learns broad legal semantics  
- Stage-2 specializes to article-specific distributions  
- Class weights improve minority-class F1  
- Reduces domain-shift effects (terminology, negation patterns)

---

### h) Per-Class Behavior

Stage-2 per-class F1:

- A2: Ent 0.837, Contra 0.500, Neutral 0.954  
- A3: 0.571, 0.800, 0.961  
- A7: 0.571, 0.835, 0.965  
- A8: 0.923, 0.884, 0.977  

Neutral is highest due to prevalence; gains mainly occur in entailment and contradiction.

---

### i) Takeaways

1. Stage-1 produces a strong general-purpose legal encoder.  
2. Stage-2 specialization closes article-specific gaps.  
3. Macro-F1 is the most informative metric under imbalance.  
4. For low compute, LoRA/adapters/prefix-tuning provide efficient alternatives.
