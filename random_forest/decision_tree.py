import numpy as np

from .node import Node
from .utils import vote

class DecisionTree:
    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        self.root = Node()
    
    def _create(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node()

        majority_y = vote(y)
        is_single_class = np.all(y == majority_y)
        if is_single_class or depth == self.max_depth:
            node.leaf = True
            node.node_class = majority_y
        else:
            node.split(X, y)
            left_mask = node.ask(X)
            right_mask = ~left_mask

            node.left = self._create(X[left_mask], y[left_mask], depth + 1)
            node.right = self._create(X[right_mask], y[right_mask], depth + 1)
        
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.root = self._create(X, y, depth=0)
        print("\t@ Decision tree creation done @", flush=True)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        N = X_test.shape[0]
        pred_test = np.full(N, fill_value=-np.inf)

        for i in range(N):
            x = X_test[i]
            current_node = self.root
            
            while not current_node.leaf:
                go_left = x[current_node.best_j] > current_node.best_val
                if go_left:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            
            pred_test[i] = current_node.node_class
            if i % 500 == 0:
                print(f"\t@ {i}th prediction was done @", flush=True)
        assert  np.all(pred_test != -np.inf)

        return pred_test