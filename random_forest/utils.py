import numpy as np

def vote(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


def entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    ent = -p * np.log2(p)
    return ent.sum()
