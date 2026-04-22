#pip install scikit-learn
#pip install pandas
import faiss
import numpy as np
import time
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# -----------------------------
# 1. LOAD MNIST
# -----------------------------
print("Loading MNIST dataset...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = np.asarray(mnist.data, dtype=np.float32)
y = np.asarray(mnist.target, dtype=np.int32)

nb = 60000
nq = 10000

#xb = np.ascontiguousarray(X[:nb], dtype=np.float32)
#xq = np.ascontiguousarray(X[nb:nb+nq], dtype=np.float32)


xb, xq = train_test_split(X, test_size=10000, random_state=42)


d = xb.shape[1]

print(f"Loaded MNIST subset: xb={xb.shape}, xq={xq.shape}, dim={d}")

# NOW SAFE FOR FAISS
faiss.normalize_L2(xb)
faiss.normalize_L2(xq)

# -----------------------------
# 2. PARAMETERS
# -----------------------------
k = 10
NLIST = 32
NPROBE = 8

# -----------------------------
# 3. GROUND TRUTH
# -----------------------------
print("\nComputing ground truth...")
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


# -----------------------------
# 8. HYPERPARAMETER STUDY
# -----------------------------
print("\n" + "="*95)
print("SECTION 8: DEEP HYPERPARAMETER STUDY (ADAPTIVE AQ)")
print("="*95)
print(f"{'M':<3} | {'nlist':<6} | {'nprobe':<6} | {'Delta':<6} | {'Recall':<8} | {'Latency':<10} | {'Trigger%'}")
print("-" * 95)

# Hyperparameter Grid
M_values = [4, 8, 16]          # Storage Compression
nlist_values = [32, 64, 128]    # Indexing Granularity
nprobe_values = [4, 8, 16]     # Search Breadth
delta_values = [0.02, 0.05, 0.1]  # Optimizer Sensitivity

for M_study in M_values:
    # Build a specific index for this M/NLIST combination
    for nl_study in nlist_values:
        quantizer_study = faiss.IndexFlatL2(d)
        index_study = faiss.IndexIVFResidualQuantizer(quantizer_study, d, nl_study, M_study, 8)
        index_study.train(xb)
        index_study.add(xb)
        
        for np_study in nprobe_values:
            index_study.nprobe = np_study
            
            for d_thresh in delta_values:
                triggers = 0
                results_study = []
                
                start_study = time.perf_counter()
                
                # Phase 1: Initial Coarse Search
                _, I_initial = index_study.search(xq, 40)
                
                for i in range(len(xq)):
                    cand = I_initial[i]
                    
                    # Refine distances for the initial pool
                    vecs = xb[cand]
                    diff = vecs - xq[i]
                    dists = np.einsum('ij,ij->i', diff, diff)
                    sorted_idx = np.argsort(dists)
                    sorted_d = dists[sorted_idx]
                    
                    # Calculate Confidence Gap
                    gap = sorted_d[min(k, len(sorted_d)-1)] - sorted_d[0]
                    
                    # Phase 2: Adaptive Decision
                    if gap < d_thresh:
                        triggers += 1
                        _, cand_exp = index_study.search(xq[i:i+1], 120)
                        cand = cand_exp[0]
                        vecs_exp = xb[cand]
                        diff_exp = vecs_exp - xq[i]
                        dists_exp = np.einsum('ij,ij->i', diff_exp, diff_exp)
                        results_study.append(cand[np.argsort(dists_exp)[:k]])
                    else:
                        results_study.append(cand[sorted_idx[:k]])
                
                lat_study = (time.perf_counter() - start_study) / len(xq) * 1000
                rec_study = recall(np.array(results_study))
                trigger_pct = (triggers / len(xq)) * 100
                
                print(f"{M_study:<3} | {nl_study:<6} | {np_study:<6} | {d_thresh:<6} | {rec_study:.4f} | {lat_study:.3f} ms  | {trigger_pct:.1f}%")

print("=" * 95)
