#pip install scikit-learn
#pip install pandas
#pip install tensorflow-datasets tensorflow
#pip install importlib-resources
#C:\Users\CA ANJALI OFFICE\tensorflow_datasets\deep1b\1.0.0.
#pip install h5py
import faiss
import numpy as np
import time
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


import h5py
import numpy as np
import faiss
import time

# -----------------------------
# 1. LOAD DEEP1B (HDF5)
# -----------------------------
print("Loading Deep-Image-96 from HDF5...")

def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        # 'train' contains 1,000,000 vectors
        # 'test' contains 10,000 vectors
        xb = np.array(f['train'], dtype='float32')
        xq = np.array(f['test'], dtype='float32')
    return xb, xq

# Path to your downloaded file
h5_path = "deep-image-96-angular.hdf5"
xb, xq = load_hdf5(h5_path)

d = xb.shape[1] # Should be 96
print(f"Loaded Deep1B: xb={xb.shape}, xq={xq.shape}")

# -----------------------------
# 2. PREP & NORMALIZE
# -----------------------------
xb = np.ascontiguousarray(xb)
xq = np.ascontiguousarray(xq)

# The dataset is 'angular' (Cosine similarity), 
# so L2 on normalized vectors is the correct way to search.
faiss.normalize_L2(xb)
faiss.normalize_L2(xq)
# -----------------------------
# 2. PARAMETERS
# -----------------------------
k = 10
NLIST = 4096      # IMPORTANT: larger for 1M
NPROBE = 32       # increase for better recall

# -----------------------------
# 3. GROUND TRUTH (FAST VERSION)
# -----------------------------
print("\nComputing ground truth (this may take time)...")

index_gt = faiss.IndexFlatL2(d)
index_gt.add(xb)

_, I_gt = index_gt.search(xq, k)

def recall(pred):
    return np.mean([len(np.intersect1d(p, t)) / k for p, t in zip(pred, I_gt)])

def map_score(pred):
    aps = []
    for t, p in zip(I_gt, pred):
        hits, score = 0, 0
        for i, v in enumerate(p):
            if v in t:
                hits += 1
                score += hits / (i + 1)
        aps.append(score / k)
    return np.mean(aps)
# -----------------------------
# 4. BUILDERS
# -----------------------------
def build_pq(M):
    index = faiss.IndexPQ(d, M, 8)
    index.train(xb)
    index.add(xb)
    return index

def build_opq(M):
    opq = faiss.OPQMatrix(d, M)
    index = faiss.IndexPreTransform(opq, faiss.IndexPQ(d, M, 8))
    index.train(xb)
    index.add(xb)
    return index

def build_aq(M):
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFResidualQuantizer(
        quantizer, d, NLIST, M, 8
    )
    index.nprobe = NPROBE
    index.train(xb)
    index.add(xb)
    return index

# -----------------------------
# 5. AQ + ADAPTIVE RERANKING
# -----------------------------
def adaptive_search_fast(index, xq, k):
    results = []

    # coarse search
    _, I = index.search(xq, 40)

    for i in range(len(xq)):
        cand = I[i]

        vecs = xb[cand]
        diff = vecs - xq[i]
        dists = np.einsum('ij,ij->i', diff, diff)

        sorted_d = np.sort(dists)
        gap = sorted_d[min(k, len(sorted_d)-1)] - sorted_d[0]

        if gap < 0.05:
            _, cand = index.search(xq[i:i+1], 120)
            cand = cand[0]

            vecs = xb[cand]
            diff = vecs - xq[i]
            dists = np.einsum('ij,ij->i', diff, diff)

        results.append(cand[np.argsort(dists)[:k]])

    return np.array(results)

# -----------------------------
# 6. PRINT FUNCTION
# -----------------------------
def print_row(name, M, pred, latency):
    print(f"{name:<12} | {M:<3} | {recall(pred):.4f} | {map_score(pred):.4f} | {latency:.3f} ms")

# -----------------------------
# 7. MAIN EXPERIMENT
# -----------------------------
print("\nFULL M COMPARISON (PQ vs OPQ vs AQ vs AQ+Adaptive)")
print("=" * 95)
print(f"{'Method':<12} | {'M':<3} | {'Recall':<8} | {'MAP':<8} | Latency")
print("-" * 95)

for M in [8, 16]:

    pq = build_pq(M)
    opq = build_opq(M)
    aq = build_aq(M)

   
   # ---------------- PQ ----------------

    
    start = time.perf_counter()
    _, pred = pq.search(xq, k)
    lat = (time.perf_counter() - start) / len(xq) * 1000
    print_row("PQ", M, pred, lat)

    # ---------------- OPQ ----------------
    start = time.perf_counter()
    _, pred = opq.search(xq, k)
    lat = (time.perf_counter() - start) / len(xq) * 1000
    print_row("OPQ", M, pred, lat)

    # ---------------- AQ ----------------
    start = time.perf_counter()
    _, pred = aq.search(xq, k)
    lat = (time.perf_counter() - start) / len(xq) * 1000
    print_row("AQ", M, pred, lat)
    
    
    # ---------------- AQ + ADAPTIVE ----------------
    start = time.perf_counter()
    pred = adaptive_search_fast(aq, xq, k)
    lat = (time.perf_counter() - start) / len(xq) * 1000
    print_row("AQ+Adaptive", M, pred, lat)

print("=" * 95)