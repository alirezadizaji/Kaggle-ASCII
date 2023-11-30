from typing import List, Tuple

import numpy as np

from .decision_tree import DecisionTree
from .utils import vote

class RandomForestClassifier:
    def __init__(self, n_estimators: int = 10, max_depth: int = 5, p_bootstraping: float = 0.5,
                p_featuring: float = 0.5, num_cls: int = 2) -> None:

        self.max_depth: int = max_depth
        self.n_estimators: int = n_estimators

        self.p_featuring: float = p_featuring
        self.p_bootstrapping: float = p_bootstraping

        self.estimators: List[Tuple[DecisionTree, np.ndarray]] = list()
        self.num_cls: int = num_cls

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        num_bootstrapping: int = int(self.p_bootstrapping * y.size)
        num_featuring: int = int(self.p_featuring * X.shape[1])

        for i in range(self.n_estimators):
            print(f"@@ {i + 1}th decision tree training: first estimator @@")
            rows_inds = np.random.choice(y.size, size=num_bootstrapping, replace=True)
            cols_inds = np.random.choice(X.shape[1], size=num_featuring, replace=False)

            sub_X = X[rows_inds][:, cols_inds]
            sub_y = y[rows_inds]
            
            dt = DecisionTree(self.max_depth)
            dt.fit(sub_X, sub_y)

            self.estimators.append((dt, cols_inds))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([dt.predict(X[:, cols_inds]) for dt, cols_inds in self.estimators]) # N_EST x N_SAMPLES
        pred = np.array([vote(y) for y in predictions.T])

        pred = pred.astype(np.uint8)
        one_hot_pred = np.eye(self.num_cls)[pred]
        return one_hot_pred