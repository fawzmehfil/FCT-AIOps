
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np


def feature_vector(metrics: List[Dict[str, Any]]) -> np.ndarray:
    services = sorted(m["service"] for m in metrics)
    row = []
    for svc in services:
        m = next(x for x in metrics if x["service"] == svc)
        row.extend([
            float(m["cpu"]),
            float(m["memory"]),
            float(m["latency_ms"]) / 1000.0,  # scale
            float(m["error_rate"]),
        ])
    return np.array([row], dtype=np.float32)


def suspect_service(metrics: List[Dict[str, Any]]) -> str:
    best = metrics[0]["service"]
    best_score = -1.0

    for m in metrics:
        cpu = float(m["cpu"])
        mem = float(m["memory"])
        lat = float(m["latency_ms"])
        err = float(m["error_rate"])

        score = 1.3 * err + 0.7 * cpu + 0.6 * mem + 0.002 * lat

        if score > best_score:
            best_score = score
            best = m["service"]

    return best