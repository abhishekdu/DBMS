import numpy as np

def read_fvecs(filename):
    data = np.fromfile(filename, dtype=np.float32)
    dim = data.view(np.int32)[0]   # first int = dimension
    return data.reshape(-1, dim + 1)[:, 1:]

vectors = read_fvecs("siftsmall_query.fvecs")
print(vectors.shape)

with open("siftsmall_query.fvecs", "rb") as f:
    print(f.read(64))  # first 64 bytes
    
import numpy as np

np.set_printoptions(precision=3, suppress=True)

print(vectors[0])      # first vector
print(vectors[:5])     # first 5 vectors