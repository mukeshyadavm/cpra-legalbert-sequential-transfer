## Stage 5. Dataset Size and Label Composition

### Overview
Figure 3 shows the total number of premise–hypothesis pairs generated for each CPRA/CCPA article.  
Figure 4 shows the label composition — Neutral, Entailment, and Contradiction — for the same subsets.

Together, these figures summarize the overall scale of the dataset and the difficulty of the classification task.  
Larger datasets generally support more robust modeling, while label imbalance increases the complexity of supervised learning.

---

### Dataset Segmentation and Construction
Each article-specific subset is created by pairing:

- Premises: Privacy-policy sentences from the OPP-115 corpus  
- Hypotheses: Sentences derived from CPRA/CCPA Articles 2, 3, 7, and 8

Hypotheses represent concrete obligations or rights under California privacy law.  
A semantic-similarity filter (FAISS + Sentence Transformers) ensures that only topically relevant premise–hypothesis pairs are kept.  
This produces legally coherent subsets that directly map back to specific statutory provisions.

---

### Dataset Size Analysis
The resulting dataset varies widely across articles:

| Article | Labeled Pairs |
|---------|---------------|
| Article 2 | 2,639 |
| Article 3 | 6,854 |
| Article 7 | 6,854 |
| Article 8 | 2,427 |

Articles 3 and 7 contain the largest number of labeled examples, reflecting how often these legal themes appear in public-facing privacy policies.  
Articles 2 and 8 appear less frequently, resulting in smaller subsets.

---

### Class Composition
Across all articles, the Neutral class is dominant.  
This indicates that many policy statements relate indirectly to the legal hypothesis or provide general information rather than explicit commitments.

- Entailment labels are consistently rare  
- Contradiction appears more frequently in Article 8 than in others  

These imbalances suggest the need for training strategies such as class-balanced losses or sampling-based corrections to avoid bias toward the majority class.

---

### Figures

#### Figure 3 — Dataset Size per Article
<img src="article_row_counts_gray_singlecol (1).png" width="420"/>

#### Figure 4 — Class Distribution Across Articles
<img src="class_distribution_grouped (1).png" width="550"/>

---


