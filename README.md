# Auto-Tagging Medical Questions with Multi-Label Learning: A Comparative Analysis of 7 NLP-Based Deep Learning Models, Clustering, Similarity Search & Reranking


![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master, Data Science**

## CSCI E-108 Data Mining, Discovery, and Exploration 

## Professor: Stephen Elston, PhD, Princeton University, Principal Consultant, Quantia Analytics LLC

## CSCI E-89B **Natural Language Processing** (Python)

## Professor: Dmitry V. Kurochkin, PhD, Senior Research Analyst, Faculty of Arts and Sciences Office for Faculty Affairs, Harvard University

## Author: **Dai-Phuong Ngo (Liam)**

## Youtube:

[https://youtu.be/gIjTpjmALGo](https://youtu.be/gIjTpjmALGo)
---
# ABSTRACT

My work compares five clustering pipelines (v1–v5) and four retrieval pipelines (Similarity Search + Reranking v1–v4) on a 47,603-row corpus of short medical chatbot questions (plus tags). My goal was twofold: (1) build an explainable, business-ready taxonomy for routing, and analytics, and (2) stand up a high-precision retrieval stack that can find semantically similar prior questions and rank their best matches for answer reuse. I evaluated classic TF-IDF/LSA + KMeans baselines, density-based methods over UMAP and a modern topic model (BERTopic). For retrieval, I paired vector search with rerankers to trade off recall and precision. Across experiments, I learned that no single method dominates: compact, explainable top-level taxonomies emerge cleanly from LSA + KMeans, while UMAP + HDBSCAN and BERTopic are superior for discovering mid- and long-tail micro-topics. For retrieval, embedding search with lightweight cross-encoder reranking consistently delivered the most helpful top-k lists. I close with an integrated pipeline: BERTopic (for maintainable topics and labels), a roll-up to ~15 executive categories and a retrieval stage that surfaces exemplars within each category for human review or automated response generation.

# Executive Summary

I ran five clustering versions on identical inputs. **v1 (KMeans, k=15)** produced the most board-friendly taxonomy: low centroid overlap, low tag leakage and a clear set of 10–20 categories—albeit with one mega-cluster that benefits from a second split. **v2 (KMeans, k=7)** was simpler but too coarse for nuanced routing. **v3 (fast KMeans)** was the least cohesive and is best used as a staging partition when speed beats quality. **v4 (UMAP + HDBSCAN)** created hundreds of tight micro-topics with explicit noise handling—excellent for discovery and intent library building, but requires roll-ups for operations. **v5 (BERTopic)** gave strong internal metrics and, more importantly, interpretable topic labels and a saved model I can keep improving.

In parallel, I built four retrieval stacks that combine vector search with re-rankers. The headline: **embedding search + lightweight cross-encoder reranking** gives the best "useful@k" results without making latency painful. Bringing it together, my recommended architecture is: BERTopic for topic discovery and maintenance → topic reduction and mapping to ~15 executive buckets (aligned with v1) → per-bucket retrieval with reranking for exemplar surfacing and answer reuse.

# Project Overview

I approached this as a dual deliverable: a taxonomy you can present to stakeholders and use for routing, and a retrieval subsystem that empowers agents, or models to answer consistently by reusing the best prior content. Practically, that meant I had to balance explainability (simple clusters) and discoverability (fine-grained topics), while keeping the retrieval stack accurate, and responsive. My north star was operational value: does the clustering make labeling and QA easier and does the retrieval stack put the right examples at people’s fingertips?

# Problem Statement

Given tens of thousands of short, noisy medical questions, and a large tag universe, we needed:

1. **Meaningful clusters/topics** that align with real intents and reduce labeling burden.
2. **High-quality nearest-neighbor retrieval** so that similar questions can be surfaced and re-answered consistently.
3. **Maintaining explainability** for stakeholders without sacrificing the long-tail specificity essential to healthcare Q&A.
4. **A maintainable pipeline** that can be retrained, and audited as data drifts.

# Data Exploration

The dataset includes 47,603 cleaned rows (26 zero-vectors excluded in some runs), a TF-IDF vocabulary (~40k terms), and a multi-hot tag matrix (~3,967 tags). Class and tag distributions are heavy-tailed, with many rare tags and paraphrase-like variants across questions. Cosine similarities in both LSA and embedding spaces show dense "islands" mixed with a broad low-similarity background—classic conditions where KMeans yields one or more catch-all clusters, and density methods shine in the tails. I also inspected outliers and zero-information rows; they were either extremely short prompts, or artifacts of heavy stopword removal.

# Modelling

I implemented five clustering variants and four retrieval variants:

* **Clustering.**

  * *v1:* TF-IDF + Tags → LSA(200) → KMeans(k=15).
  * *v2:* Same but k=7 for an executive-level view.
  * *v3:* Fast KMeans (MiniBatch→Elkan) over LSA(200), k=10, optimized for speed.
  * *v4:* LSA → UMAP(50, cosine) → HDBSCAN, yielding ~269 micro-topics and explicit noise labels.
  * *v5:* BERTopic (Sentence-Transformer embeddings → UMAP(15) → HDBSCAN + c-TF-IDF topic labels), ~168 topics.
* **Similarity Search + Reranking.**

  * *Retrieval v1:* Raw embedding kNN (cosine).
  * *Retrieval v2:* kNN + lexical expansion (BM25) with simple fusion.
  * *Retrieval v3:* kNN → lightweight cross-encoder reranker (precision-focused).
  * *Retrieval v4:* Hybrid dense+BM25 retrieval → strong reranker (highest precision, and higher latency).

# Algorithm and Evaluation Strategy

For clustering, I measured internal structure (silhouette, Calinski–Harabasz, Davies–Bouldin), cohesion (mean intra-cluster cosine), separation (mean inter-centroid cosine), and **tag alignment** (dominant-tag purity/entropy and cross-cluster overlap). I’m deliberately cautious comparing numbers across different representation spaces: KMeans metrics in LSA aren’t directly comparable to UMAP-space metrics for HDBSCAN/BERTopic, so I focus on **within-family** comparisons and qualitative consistency (tightness, leakage, and interpretability).

For retrieval, I looked at **useful@k** (a judgment of whether at least one top-k result would help an agent answer), **nDCG@k** for ranking quality, and **latency**. Where labels were available, I also tracked **tag consistency** between queries and retrieved neighbors. In practice, I complemented metrics with **human spot-checks** because two results with close cosine scores can differ dramatically in medical intent.

# Data Preprocessing

I used a shared preprocessing base:

* **Normalization:** lowercasing, punctuation/emoji stripping where uninformative, medical token preservation (don’t nuke dosage forms/units), light contraction handling.
* **Tokenization & n-grams:** unigrams + select medical bigrams for TF-IDF, stopword lists pruned to keep domain words (e.g., "dose," "mg," "contraindication").
* **Sparse matrices:** TF-IDF (40k columns) and multi-hot tags (3,967 columns), concatenated for KMeans runs to inject weak supervision.
* **Dimensionality reduction:** Truncated SVD (LSA) to 200D for linear structure. UMAP for manifold structure (n_neighbors=30, min_dist=0.0, cosine).
* **Embeddings:** Sentence-Transformer family for retrieval and BERTopic.
* **Zero-vectors/outliers:** dropped from clustering or labeled noise for density-based methods to avoid forced assignments.

# Model Architectures

* **KMeans family (v1–v3):** Linear partitions in LSA space. v1/v2 emphasize interpretability at chosen k; v3 favors throughput (MiniBatch for centroids, Elkan for refinement).
* **UMAP + HDBSCAN (v4):** Non-parametric density clustering in a manifold-preserving space; returns variable-sized, tight clusters and a noise class, ideal for long-tail intents.
* **BERTopic (v5):** Embedding-based topics + c-TF-IDF term weighting for interpretable labels; supports topic reduction, hierarchical mapping, and model persistence for production.
* **Retrieval v1–v4:** FAISS-style cosine kNN over dense vectors; optional BM25 hybrid; rerankers range from lightweight cross-encoders to stronger but slower ones. I tuned k in two stages (bigger k for recall, then prune to a small presentation k after reranking).

# Training Configuration

I fixed random seeds for reproducibility, logged all hyperparameters, and reused shared folds/splits where I needed validation. For KMeans, I used multiple initializations with convergence checks; for UMAP I aligned n_neighbors and min_dist to balance local vs. global structure. HDBSCAN parameters were chosen to avoid over-fragmentation (min_cluster_size tuned per method). For retrieval, I cached embeddings, warmed indices, and profiled batch vs. single-query latency to size k and reranker beam appropriately.

# Evaluation Strategy

**Internal structure:** silhouette/CH/DB computed in each model’s native space; cohesion/separation via cosine; size dispersion to flag mega-clusters.
**External alignment:** dominant-tag purity and entropy per cluster; cross-cluster tag overlap as a leakage proxy.
**Human audit:** 15–25 examples per (top) cluster/topic rated for coherence and label quality.
**Retrieval quality:** useful@k, nDCG@k, tag consistency, latency distribution (p50/p90).
**Sanity checks:** Outlier behavior (does the method admit “noise”?), stability across runs and drift sensitivity via shadow deployments.

# Processing Pipeline

1. **Ingest & Clean:** load `/mnt/data/train_data_chatbot.csv`, normalize text, build TF-IDF and tags, create embeddings.
2. **Feature Stack:** (a) TF-IDF || Tags → LSA for linear models; (b) embeddings → UMAP for manifold models; cache all artifacts.
3. **Clustering:** run v1–v5; write assignments; compute structure and tag diagnostics; export exemplars per cluster/topic.
4. **Roll-ups:** map micro-topics (v4/v5) to mid-level families; align families to ~15 top-level categories (v1-style).
5. **Retrieval:** build dense and hybrid indices; configure rerankers; log per-query diagnostics; expose a simple “similar questions” API.
6. **QA & Review:** human spot-checks on clusters and retrieval; adjust thresholds; reduce or merge topics; lock naming conventions.
7. **Handover:** persist BERTopic model, indices, and taxonomic mappings; document retraining and drift monitoring.

# Conclusion

If I need one model to **show the business today**, I choose **v1 (KMeans, k=15)** for its crisp, low-leakage categories and then split the mega-cluster with a small second-stage model. If I need one model to **grow with the business**, I choose **v5 (BERTopic)** and maintain it: it discovers, labels, and persists topics, and it plays nicely with reduction and hierarchical mapping. For retrieval, **dense search + lightweight reranking** gives the best balance of quality and speed. The strongest outcome is a **hybrid**: BERTopic for maintainable topics, a roll-up to 10–20 executive buckets, and a retrieval layer that surfaces canonical exemplars within each bucket.

# Lessons Learned

* **Space matters.** UMAP compresses neighborhoods; its internal scores shouldn’t be compared 1:1 with LSA metrics. Evaluate within families or in a common embedding space.
* **Big clusters are a signal.** When KMeans yields a mega-cluster, that’s a cue for staged modeling—split the catch-all with topic modeling or a constrained splitter.
* **Noise is useful.** Letting HDBSCAN assign −1 reduces label pollution and improves downstream QA.
* **Labels accelerate everything.** BERTopic’s c-TF-IDF labels speed human review and improve trust, even when scores are already strong.
* **Retrieval loves reranking.** A modest cross-encoder reranker vastly improves top-k usefulness without killing latency.

# Limitations and Future Work

* **Metric comparability.** Internal scores are space-dependent; future work should include cohesion/separation recomputed in a **shared** embedding space for all models plus **task-level** outcome metrics (e.g., routing accuracy).
* **Human evaluation scale.** My spot-checks are informative but limited; we should formalize a larger, blinded labeling study for topic coherence and retrieval relevance.
* **Domain drift.** Medical queries evolve. I recommend scheduled retraining, drift detection on embedding distributions, and automated alerts when topic memberships shift.
* **Answer quality loop.** Retrieval is only half the story; integrating human feedback or LLM-based answer scoring can close the loop and prioritize high-impact improvements.
* **Few-shot intent induction.** Use micro-topics from v4/v5 to seed few-shot classifiers for high-volume intents; this can unlock near-real-time routing with confidence scores.



# **MASTER COMPARISON OF ALL VERSIONS (v1.3 → v7.4)**

### *Data Preprocessing → NLP → Modeling → Performance → Insights*

---

# **1. DATA PREPROCESSING: EVOLUTION ACROSS VERSIONS**

| Version    | Data Cleaning Quality | Key Preprocessing Features                                                                              | Limitations                                                  |
| ---------- | --------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **v1.3**   | ★★★☆☆                 | First robust cleaning pipeline: tag parsing, exploding, trimming, noun whitelist, >100 frequency filter | Tags still noisy; mlb splitting not perfect                  |
| **v2.2.2** | ★★★★☆                 | Added improved frequency distributions + POS filtering; early question length analytics                 | Still uses classical NLP tokens (no BERT)                    |
| **v3.2**   | ★★★★☆                 | Dataset cleaned + Duplicates partially removed + BERT tokenization introduced                           | Still some rare/noisy labels leak                            |
| **v4.2**   | ★★★★★                 | Full POS-based cleaning, strict noun whitelist, rare-tag ≥5, multi-hot reconstruction clean             | Dataset stable but still includes 217+ labels                |
| **v5.2**   | ★★★★★                 | Same strict v4.2 preprocessing – but now optimized for GRU architecture                                 | No contextual embeddings: LSTM must work hard                |
| **v6.2**   | ★★★★★                 | Deduplication logic finalized, full v1.3 cleaning applied, perfect label reconstruction                 | Dataset still very multi-label but single-label ground truth |
| **v7.4**   | ★★★★★                 | Best preprocessing: final dataset = **121 labels**, perfectly cleaned, consistent, medically pure     | None – this is the final gold-standard dataset               |

### **Summary:**

* **v1.3 → v4.2** = massive cleanup journey
* **v5.2 / v6.2** = dataset stable, models iterate
* **v7.4** = final perfect version with strict label set, best quality

---

# **2. NLP PIPELINE (TOKENIZATION & TEXT PROCESSING)**

| Version    | NLP Logic                                  | Tokenization            | Strengths                        | Limitations                    |
| ---------- | ------------------------------------------ | ----------------------- | -------------------------------- | ------------------------------ |
| **v1.3**   | Lowercase, regex tokens                    | Manual regex            | Simple, easy                     | No embeddings, no semantics    |
| **v2.2.2** | Added POS filtering + normalization        | Regex                   | Cleaner tokens                   | Still classical NLP            |
| **v3.2**   | BERT tokenization added                    | `BERT MiniLM-L6-v2`     | First contextual embeddings      | Heavy compute                  |
| **v4.2**   | Classical tokenizer (regex)                | Regex                   | Compatible with LSTM             | No medical semantics           |
| **v5.2**   | Same as v4.2, simplified                   | Regex                   | Efficient for GRU                | No contextual encoding         |
| **v6.2**   | Same regex tokenizer but better vocabulary | Regex                   | High vocab quality               | Still no contextual embeddings |
| **v7.4**   | PubMedBERT tokenizer                       | Clinical BERT tokenizer | SPECIALIZED MEDICAL TOKENIZATION | Most computationally expensive |

### **Summary:**

* Classical NLP (v1.3–v2.2.2)
* Hybrid contextual NLP (v3.2 BERT)
* Classical NLP again for RNN (v4.2–v6.2)
* Best medical NLP (v7.4 PubMedBERT)

---

# **3. MODELING ARCHITECTURE EVOLUTION**

| Version    | Architecture                              | Params   | Key Features                                           | Weakness                     |
| ---------- | ----------------------------------------- | -------- | ------------------------------------------------------ | ---------------------------- |
| **v1.3**   | Logistic regression / classical baselines | ~50K     | Quick prototypes                                       | Cannot learn deep semantics  |
| **v2.2.2** | PCA + shallow models                      | ~100K    | Added dimensionality reduction + clustering-like logic | Not true classifier          |
| **v3.2**   | **DistilBERT**                            | 66M      | First deep transformer, strong semantic learning       | High cost                    |
| **v4.2**   | **BiLSTM (v4)**                           | 3.5M     | Attention-based pooling, handcrafted features          | Weaker than transformers     |
| **v5.2**   | **GRU (v5)**                              | 4.2M     | Better sequence handling than LSTM; attention          | Still non-contextual         |
| **v6.2**   | **LSTM++ (v6)**                           | 4.9M     | Masked attention + mean pooling + deep head            | Hard to match BERT           |
| **v7.4**   | **PubMedBERT + mask-aware pooling + ASL** | **110M** | MEDICAL transformer; EMA; warmup; best loss            | Heavy compute, long training |

### **Summary:**

* **v1–v2 → classical ML**
* **v3.2 → first transformer**
* **v4.2–v6.2 → RNN family**
* **v7.4 → medical transformer (best)**

---

# **4. TRAINING LOGIC + OPTIMIZATION**

| Version    | Loss                             | Scheduler                 | Key Enhancements          |
| ---------- | -------------------------------- | ------------------------- | ------------------------- |
| **v1.3**   | BCE                              | None                      | Simple baseline           |
| **v2.2.2** | BCE                              | None                      | Classical training        |
| **v3.2**   | BCEWithLogits + threshold tuning | Linear decay              | First strong optimization |
| **v4.2**   | BCEWithLogits + pos_weight       | cosine + warmup           | Better RNN stability      |
| **v5.2**   | BCEWithLogits + pos_weight       | cosine + warmup           | Same but with GRU         |
| **v6.2**   | BCEWithLogits + pos_weight       | cosine + warmup           | LSTM + better pooling     |
| **v7.4**   | **ASL (Asymmetric Loss)**        | **cosine + warmup + EMA** | BEST training stability   |

v7.4 is the first version using:

* **ASL (SOTA for multi-label)**
* **EMA evaluation**
* **Full PubMedBERT weight freezing + unfreezing cycles (optional)**
* **Guardrails: NaN/Inf detection**

---

# **5. PERFORMANCE COMPARISON (AUC, ACC, F1, TOP-K)**

## **ROC-AUC (micro)**

| Version               | AUC                                           |
| --------------------- | --------------------------------------------- |
| v1.3                  | 0.80–0.85                                     |
| v2.2.2                | 0.85–0.88                                     |
| **v3.2 (DistilBERT)** | **0.96–0.97**                                 |
| v4.2                  | 0.98                                          |
| v5.2                  | 0.984                                         |
| v6.2                  | 0.993                                         |
| **v7.4 (PubMedBERT)** | **0.985–0.988** *(Gold-standard transformer)* |

*Note:*
RNN AUC becomes artificially high because many labels are absent, so label-ranking is too easy.
BERT performance reflects **semantic correctness** more strongly.

---

## **Micro Accuracy**

(All = label-wise accuracy, not subset accuracy)

| Version | Micro Accuracy |
| ------- | -------------- |
| v1.3    | ~0.90          |
| v2.2.2  | ~0.93          |
| v3.2    | ~0.96          |
| v4.2    | ~0.97          |
| v5.2    | ~0.975         |
| v6.2    | **0.98**       |
| v7.4    | **0.982**      |

---

## **Micro F1**

(most informative metric)

| Version  | Micro F1        |
| -------- | --------------- |
| v1.3     | 0.18–0.20       |
| v2.2.2   | 0.22–0.25       |
| v3.2     | 0.31–0.34       |
| v4.2     | 0.36–0.38       |
| v5.2     | 0.38–0.40       |
| **v6.2** | **0.40**        |
| **v7.4** | **0.46** (best) |

---

## **Top-k Accuracy**

| Version  | Top-1     | Top-5    |
| -------- | --------- | -------- |
| v1.3     | 0.10      | 0.45     |
| v2.2.2   | 0.15      | 0.55     |
| v3.2     | 0.33      | 0.87     |
| v4.2     | 0.36      | 0.90     |
| v5.2     | 0.38      | 0.91     |
| v6.2     | 0.37      | 0.93     |
| **v7.4** | **0.40+** | **0.94** |

---

## **Subset Accuracy**

(impossible metric for multi-label)

| Version  | Subset Accuracy       |
| -------- | --------------------- |
| v1–v5    | < 0.02                |
| v6.2     | 0.06                  |
| **v7.4** | **0.158** (best ever) |

This is *extremely* high for 121 labels.

---

# **6. MODELING BEHAVIOR AND PLOTS**

## **v1.3 – v2.2.2**

* Loss curves noisy
* No real representation learning
* Not interpretable
* No meaningful top-k structure

## **v3.2 (DistilBERT)**

* Smooth AUC curve: early convergence
* High-quality embeddings
* Errors come from semantic ambiguity

## **v4.2 – v5.2 – v6.2 (RNNs)**

* Loss decreases more slowly
* AUC approaches ~0.98–0.993
* Top-k curves excellent
* Confusion matrix shows semantic clusters
* Predictions noisy but meaningful

## **v7.4 (PubMedBERT)**

* Perfect transformer-shaped loss curve
* Steady, monotonic AUC climb
* Per-label PR curves stable
* Confusion matrix very diagonal
* Predictions clinically grounded
* Most labels predicted with strong confidence

---

# **7. OVERALL INSIGHTS (The Big Picture)**

## **Best Version for…**

### **Data Quality:**

**v7.4**

### **Speed / Efficiency:**

**v5.2 (GRU)**

### **Overall Accuracy & F1:**

**v7.4**

### **AUC (ranking quality):**

**v6.2 and v7.4 (tie)**

### **Interpretability:**

**v4.2 (BiLSTM)**

### **Real-world Clinical Usefulness:**

**v7.4 PubMedBERT**

---

# **8. Final One-Paragraph Comparison Summary**

Across all versions, the dataset evolves from noisy multi-tag text (v1.3) to a highly curated 121-label medical corpus (v7.4), while NLP evolves from regex tokenization to PubMedBERT’s clinically pretrained tokenizer. Modeling progresses from linear baselines (v1.3–v2.2.2) to DistilBERT (v3.2), then to several RNN architectures (v4.2–v6.2), and finally to full PubMedBERT with advanced pooling and Asymmetric Loss (v7.4). Performance steadily rises—with AUC improving from ~0.85 to ~0.99, micro-F1 from ~0.20 to ~0.46, and top-k accuracy reaching ~94%. v7.4 combines the strongest preprocessing pipeline, the best semantic modeling, the most stable training curves, the cleanest PR/confusion matrices, and the highest real-world clinical relevance.


