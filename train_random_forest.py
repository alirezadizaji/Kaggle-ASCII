import numpy as np

from random_forest.random_forest import RandomForestClassifier


if __name__ == "__main__":
    X = np.random.rand(10000, 32)
    X_test = np.random.rand(5000, 32)
    y = np.random.randint(0, 5, 10000)

    rfc = RandomForestClassifier(n_estimators=2)
    rfc.fit(X, y)
    rfc.predict(X_test)
