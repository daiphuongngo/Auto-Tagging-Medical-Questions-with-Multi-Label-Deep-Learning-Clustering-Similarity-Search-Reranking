# Auto-Tagging Medical Questions with Multi-Label Learning: A Comparative Analysis of 7 NLP-Based Deep Learning Models, Clustering, Similarity Search & Reranking


![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master, Data Science**

## CSCI E-108 Data Mining, Discovery and Exploration 

## Professor: Stephen Elston, PhD, Princeton University, Principal Consultant, Quantia Analytics LLC

## CSCI E-89B **Natural Language Processing** (Python)

## Professor: Dmitry V. Kurochkin, PhD, Senior Research Analyst, Faculty of Arts and Sciences Office for Faculty Affairs, Harvard University

## Author: **Dai-Phuong Ngo (Liam)**

---
# ABSTRACT

My work compares five clustering pipelines (v1–v5) and four retrieval pipelines (Similarity Search + Reranking v1–v4) on a 47,603-row corpus of short medical chatbot questions (plus tags). My goal was twofold: (1) build an explainable, business-ready taxonomy for routing and analytics and (2) stand up a high-precision retrieval stack that can find semantically similar prior questions and rank their best matches for answer reuse. I evaluated classic TF-IDF/LSA + KMeans baselines, density-based methods over UMAP and a modern topic model (BERTopic). For retrieval, I paired vector search with rerankers to trade off recall and precision. Across experiments, I learned that no single method dominates: compact, explainable top-level taxonomies emerge cleanly from LSA + KMeans, while UMAP + HDBSCAN and BERTopic are superior for discovering mid- and long-tail micro-topics. For retrieval, embedding search with lightweight cross-encoder reranking consistently delivered the most helpful top-k lists. I close with an integrated pipeline: BERTopic (for maintainable topics and labels), a roll-up to ~15 executive categories and a retrieval stage that surfaces exemplars within each category for human review or automated response generation.

# Executive Summary

I ran five clustering versions on identical inputs. **v1 (KMeans, k=15)** produced the most board-friendly taxonomy: low centroid overlap, low tag leakage, and a clear set of 10–20 categories—albeit with one mega-cluster that benefits from a second split. **v2 (KMeans, k=7)** was simpler but too coarse for nuanced routing. **v3 (fast KMeans)** was the least cohesive and is best used as a staging partition when speed beats quality. **v4 (UMAP + HDBSCAN)** created hundreds of tight micro-topics with explicit noise handling—excellent for discovery and intent library building, but requires roll-ups for operations. **v5 (BERTopic)** gave strong internal metrics and, more importantly, interpretable topic labels and a saved model I can keep improving.

In parallel, I built four retrieval stacks that combine vector search with re-rankers. The headline: **embedding search + lightweight cross-encoder reranking** gives the best “useful@k” results without making latency painful. Bringing it together, my recommended architecture is: BERTopic for topic discovery and maintenance → topic reduction and mapping to ~15 executive buckets (aligned with v1) → per-bucket retrieval with reranking for exemplar surfacing and answer reuse.

# Project Overview

I approached this as a dual deliverable: a taxonomy you can present to stakeholders and use for routing, and a retrieval subsystem that empowers agents or models to answer consistently by reusing the best prior content. Practically, that meant I had to balance explainability (simple clusters) and discoverability (fine-grained topics), while keeping the retrieval stack accurate and responsive. My north star was operational value: does the clustering make labeling and QA easier and does the retrieval stack put the right examples at people’s fingertips ?

# Problem Statement

Given tens of thousands of short, noisy medical questions and a large tag universe, we needed:

1. **Meaningful clusters/topics** that align with real intents and reduce labeling burden.
2. **High-quality nearest-neighbor retrieval** so that similar questions can be surfaced and re-answered consistently.
3. **Maintaining explainability** for stakeholders without sacrificing the long-tail specificity essential to healthcare Q&A.
4. **A maintainable pipeline** that can be retrained and audited as data drifts.

# Data Exploration

The dataset includes 47,603 cleaned rows (26 zero-vectors excluded in some runs), a TF-IDF vocabulary (~40k terms), and a multi-hot tag matrix (~3,967 tags). Class and tag distributions are heavy-tailed, with many rare tags and paraphrase-like variants across questions. Cosine similarities in both LSA and embedding spaces show dense “islands” mixed with a broad low-similarity background—classic conditions where KMeans yields one or more catch-all clusters and density methods shine in the tails. I also inspected outliers and zero-information rows; they were either extremely short prompts or artifacts of heavy stopword removal.

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
  * *Retrieval v4:* Hybrid dense+BM25 retrieval → strong reranker (highest precision, higher latency).

# Algorithm and Evaluation Strategy

For clustering, I measured internal structure (silhouette, Calinski–Harabasz, Davies–Bouldin), cohesion (mean intra-cluster cosine), separation (mean inter-centroid cosine), and **tag alignment** (dominant-tag purity/entropy and cross-cluster overlap). I’m deliberately cautious comparing numbers across different representation spaces: KMeans metrics in LSA aren’t directly comparable to UMAP-space metrics for HDBSCAN/BERTopic, so I focus on **within-family** comparisons and qualitative consistency (tightness, leakage, interpretability).

For retrieval, I looked at **useful@k** (a judgment of whether at least one top-k result would help an agent answer), **nDCG@k** for ranking quality, and **latency**. Where labels were available, I also tracked **tag consistency** between queries and retrieved neighbors. In practice, I complemented metrics with **human spot-checks** because two results with close cosine scores can differ dramatically in medical intent.

# Data Preprocessing

I used a shared preprocessing base:

* **Normalization:** lowercasing, punctuation/emoji stripping where uninformative, medical token preservation (don’t nuke dosage forms/units), light contraction handling.
* **Tokenization & n-grams:** unigrams + select medical bigrams for TF-IDF; stopword lists pruned to keep domain words (e.g., “dose,” “mg,” “contraindication”).
* **Sparse matrices:** TF-IDF (40k columns) and multi-hot tags (3,967 columns); concatenated for KMeans runs to inject weak supervision.
* **Dimensionality reduction:** Truncated SVD (LSA) to 200D for linear structure; UMAP for manifold structure (n_neighbors=30, min_dist=0.0, cosine).
* **Embeddings:** Sentence-Transformer family for retrieval and BERTopic.
* **Zero-vectors/outliers:** dropped from clustering, or labeled noise for density-based methods to avoid forced assignments.

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
**Retrieval quality:** useful@k, nDCG@k, tag consistency; latency distribution (p50/p90).
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
