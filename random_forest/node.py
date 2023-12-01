import numpy as np

from .utils import entropy

class Node:
    def __init__(self) -> None:
        self.left: Node = None
        self.right: Node = None
        self.best_j: int = None
        self.best_val: float = None
        self.leaf: bool = False

        self.node_class: int = None
    

    def split(self, X: np.ndarray, y: np.ndarray) -> None:
        min_num_split: int = 8
        _, F = X.shape

        best_loss = np.inf

        for j in range(F):
            vals = np.sort(X[:, j])
            midpoint = (vals[:-1] + vals[1:]) / 2
            mv = np.unique(midpoint)
            inds = np.arange(min_num_split) * (mv.size // min_num_split)

            for m in mv[inds]:
                left_mask = X[:, j] > m
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Do not consider midpoints which cannot generate left-right children
                if y_left.size == 0 or y_right.size == 0:
                    continue

                p_left = y_left.size / y.size
                p_right = y_right.size / y.size

                loss = p_left * entropy(y_left) + p_right * entropy(y_right)
                if loss < best_loss:
                    best_loss = loss
                    self.best_j = j
                    self.best_val = m
    
    def ask(self, X: np.ndarray) -> np.ndarray:
        if not self.leaf:
            left_mask = X[:, self.best_j] > self.best_val
            return left_mask
        else:
            raise ValueError("Cannot ask a leaf node")
    
    def predict(self):
        if self.leaf:
            return self.node_class
        else:
            raise ValueError("Cannot predict on a non-leaf node")