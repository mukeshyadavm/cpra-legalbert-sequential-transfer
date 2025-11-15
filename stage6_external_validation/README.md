# Stage 6 — External Validation (SNLI Evaluation)

This stage evaluates the quality of the GPT-generated NLI labels by comparing them against the gold-standard annotations from the Stanford Natural Language Inference (SNLI) dataset. The goal is to verify whether the labeling framework used in earlier stages generalizes beyond CPRA-specific text and remains consistent on an established benchmark dataset.

---

## I. Evaluation Setup

We use the official SNLI test split and apply the same GPT-based labeling instructions that were used during the CPRA dataset construction.  
GPT-4 assigns each SNLI pair one of the following labels:

- Entailment
- Contradiction
- Neutral

Predictions are compared directly with the SNLI gold labels to evaluate agreement and assess classification reliability.

---

## II. Quantitative Results

GPT-4 achieves strong alignment with human-annotated SNLI labels:

- Accuracy: 89.4%  
- Macro-F1:0.89

Per-class performance:

| Class            | F1 Score |
|------------------|---------|
| Contradiction    | 0.93|
| Entailment       | 0.91|
| Neutral          | 0.84|

These results demonstrate both semantic robustness and efficient generalization beyond CPRA policy text.

---

## III. Confusion Matrix Analysis

The confusion matrix (Figure 5) shows a strongly diagonal pattern, indicating that GPT-4 closely matches human annotations. The largest error region is between Neutral and Entailment, which is a known ambiguity area where statements may be weakly supportive or context-dependent.

Very few cases of Contradiction are misclassified, showing strong reliability in detecting direct inconsistencies—an essential capability for compliance and legal-risk detection.

---

## IV. Implications

These results validate that GPT-based NLI labeling is:

- Reliable enough for automated legal-text classification  
- Consistent with human judgment on a standard benchmark  
- Scalable and suitable for large privacy-policy corpora  
- Robust in identifying entailed statements and contradictions  
- Realistic in handling ambiguous Neutral cases  

Overall, the findings confirm that GPT-generated NLI labels provide a trustworthy foundation for downstream CPRA compliance analysis.

---

## V. Figure — Confusion Matrix

<img src="class_distribution_grouped (1).png" width="550"/>


