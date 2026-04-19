# Adaptive Vector DBMS using FAISS (PQ, OPQ, AQ + Adaptive Reranking)

This project implements a **Vector Database Management System (Vector DBMS)** for efficient Approximate Nearest Neighbor (ANN) search using FAISS. It compares multiple vector quantization techniques and introduces an **Adaptive Reranking mechanism** to improve retrieval accuracy while maintaining scalability.

---

## Project Highlights

- Implements core ANN techniques:
  - Product Quantization (PQ)
  - Optimized Product Quantization (OPQ)
  - Additive Quantization (AQ)
- Uses IVF (Inverted File Index) for efficient large-scale search
- Proposes an **Adaptive Reranking mechanism** to improve accuracy
- Evaluates performance on multiple datasets:
  - MNIST (handwritten digits)
  - SIFT (image descriptors)
  - GloVe (word embeddings)
  - Deep Image Features (Deep1B-style dataset)

---

## System Overview

The system is inspired by modern Vector DBMS architectures used in AI search engines.

### Pipeline:

1. **IVF Indexing**
   - Clusters dataset into partitions
   - Reduces search space significantly

2. **Vector Quantization**
   - PQ: splits vectors into independent subspaces
   - OPQ: rotates vectors for better PQ performance
   - AQ: models dependencies between sub-vectors

3. **Adaptive Reranking**
   - Dynamically improves results based on query difficulty
   - Uses confidence gap between top results

---

## Adaptive Reranking (Key Contribution)

We define a confidence metric:
Δ = d₂ - d₁

- If Δ is small → results are ambiguous → rerank with more candidates  
- If Δ is large → accept result immediately  

This improves accuracy without always increasing computation.

---

## Datasets Used

| Dataset | Type | Dimensionality |
|--------|------|---------------|
| MNIST | Handwritten digits | 784 |
| SIFT | Image features | 128 |
| GloVe | Word embeddings | 100–300 |
| Deep1B | Large-scale image features | 96 |

---

## Results Summary

- AQ consistently outperforms PQ and OPQ
- Adaptive Reranking significantly boosts Recall@K
- Achieves near-exact search performance in many cases
- Trade-off: Slight increase in latency for higher accuracy

---

## Technologies Used

- Python 3.10+
- FAISS (Facebook AI Similarity Search)
- NumPy
- Scikit-learn (for dataset handling)

---

## Project Structure
project/
  sift.py          - SIFT 1M dataset experiments
  mnist.py         - MNIST dataset experiments
  deep.py          - Deep1B dataset experiments
  glove.py         - GloVe embeddings experiments
  README.md        - Project documentation



---

## Installation

```bash
pip install faiss-cpu numpy scikit-learn h5py matplotlib pandas

