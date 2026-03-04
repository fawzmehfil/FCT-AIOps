from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(self, contamination: float, random_seed: int):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_seed,
            n_estimators=200,
        )
        self.trained = False

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)
        self.trained = True

    def is_anomaly(self, X_now: np.ndarray) -> bool:
        if not self.trained:
            return False
        pred = int(self.model.predict(X_now)[-1]) 
        return pred == -1