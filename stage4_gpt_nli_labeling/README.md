# Stage 4 — NLI Labeling with GPT-4 (Articles 3, 2, 7, 8)

This stage performs large-scale Natural Language Inference (NLI) labeling between  
California Privacy Rights Act (CPRA) *articles* and OPP-115 privacy policy premises.

The goal is to generate high-quality NLI labels:
- Entailment
- Contradiction
- Neutral

using a combination of:
1. Semantic Retrieval (FAISS + Sentence Transformers)
2. Batch GPT-4-Turbo Labeling with JSONL output

---

##  Overview of the Pipeline

### Step 1 — Load OPP-115 Premises
We load `opp115_all_policies_combined.csv` and extract all policy sentences as premises.

### Step 2 — Load CPRA Article (Article 3 example)
Example input: article_3_full.txt

The full article text is sentence-segmented into hypotheses.

### step 3 — Semantic Matching with FAISS
We embed:
- All OPP-115 premises  
- All Article 3 hypotheses  

Using:all-MiniLM-L6-v2

We build a FAISS inner-product search index to retrieve semantically similar pairs  
above a configurable similarity threshold (`0.6`).

Results are saved as: article3_nli_semantic_pairs_faiss.csv


### Step 4 — GPT-4-Turbo Legal NLI Labeling
The candidate sentence pairs are labeled using GPT-4-Turbo.

The script:
- Automatically resumes from the last saved CSV  
- Handles batching  
- Retries failed batches  
- Outputs JSON Lines → merged into CSV
- Saves incremental progress to avoid API waste

Output file example: article3_nli_semantic_pairs_labeled.csv


---

##  Labeling Other CPRA Articles (2, 7, 8)

Articles 2, 7, and 8 were labeled using the exact same workflow as Article 3:

1. Extract full article text  
2. Split into hypotheses  
3. Run FAISS retrieval  
4. Send semantic pairs to GPT-4-Turbo for NLI classification  
5. Save labeled CSVs  

Corresponding files follow the same naming pattern:
article2_nli_semantic_pairs_labeled.csv
article7_nli_semantic_pairs_labeled.csv
article8_nli_semantic_pairs_labeled.csv


This ensures consistent, reproducible results across all articles.

---



##  Requirements

Install the following dependencies:
```bash
pip install faiss-cpu
pip install sentence-transformers
pip install pandas
pip install openai

