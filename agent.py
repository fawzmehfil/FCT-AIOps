from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from telemetry import feature_vector, suspect_service
from anomaly import IsolationForestDetector


@dataclass
class Decision:
    incident: bool
    service: Optional[str]
    rca: str
    actions: List[Dict[str, Any]]


class AIOpsAgentLite:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.detector = IsolationForestDetector(
            contamination=cfg["anomaly"]["contamination"],
            random_seed=cfg["anomaly"]["random_seed"],
        )
        self.warmup_ticks = int(cfg["run"]["warmup_ticks"])

        self._tick = 0
        self._train_rows: List[Any] = []

    def observe_and_maybe_train(self, metrics: List[Dict[str, Any]]) -> None:
        self._tick += 1
        X = feature_vector(metrics)

        if self._tick <= self.warmup_ticks:
            self._train_rows.append(X[0])
            return

        if not self.detector.trained:
            import numpy as np
            train_X = np.array(self._train_rows, dtype=np.float32)
            self.detector.fit(train_X)

    def decide(self, tools: Any, metrics: List[Dict[str, Any]]) -> Decision:
        """
        tools: Orchestrator-provided ACI tools (get_logs/get_traces/restart/scale/ticket)
        """
        if not self.detector.trained:
            return Decision(False, None, "Warmup: building baseline, not detecting yet.", [])

        X = feature_vector(metrics)
        anomaly = self.detector.is_anomaly(X)
        symptomatic = any(m["error_rate"] > 0.12 or m["latency_ms"] > 260 for m in metrics)
        if not (anomaly and symptomatic):
            return Decision(False, None, "Healthy: no actionable anomaly (passed ML gate or SLO gate).", [])

        svc = suspect_service(metrics)
        logs = tools.get_logs(svc)
        traces = tools.get_traces(svc)

        rca = self._rca(svc, metrics, logs, traces)
        actions = self._plan_actions(svc, rca)

        return Decision(True, svc, rca, actions)


    def _rca(self, svc: str, metrics: List[Dict[str, Any]], logs: str, traces: Dict[str, Any]) -> str:
        m = next(x for x in metrics if x["service"] == svc)
        cpu, mem, lat, err = m["cpu"], m["memory"], m["latency_ms"], m["error_rate"]

        sigs = []
        lower = logs.lower()
        if "econnrefused" in lower or "connection refused" in lower:
            sigs.append("connection_refused")
        if "timed out" in lower or "timeout" in lower:
            sigs.append("timeout")
        if "unauthorized" in lower or "permission denied" in lower or "auth failed" in lower:
            sigs.append("auth")
        if "oomkilled" in lower or "out of memory" in lower:
            sigs.append("oom")

        trace_errors = traces.get("errors", []) if isinstance(traces, dict) else []

        hypothesis = "Uncertain anomaly."
        if "oom" in sigs or mem > 0.90:
            hypothesis = "Likely memory pressure / OOM instability."
        elif cpu > 0.90:
            hypothesis = "Likely CPU saturation driving latency/errors."
        elif "connection_refused" in sigs:
            hypothesis = "Likely port/service unreachable (connection refused)."
        elif "auth" in sigs:
            hypothesis = "Likely auth/secret/config issue to a dependency."
        elif "timeout" in sigs:
            hypothesis = "Likely downstream slowness / network timeout."

        return (
            f"Anomaly detected.\n"
            f"Suspected service: {svc}\n"
            f"Metrics: cpu={cpu:.2f}, mem={mem:.2f}, latency_ms={lat:.1f}, error_rate={err:.2f}\n"
            f"Log signals: {', '.join(sigs) if sigs else 'none'}\n"
            f"Trace errors: {trace_errors[:2] if trace_errors else 'none'}\n"
            f"Root-cause hypothesis: {hypothesis}"
        )

    def _plan_actions(self, svc: str, rca: str) -> List[Dict[str, Any]]:
        max_actions = int(self.cfg["remediation"]["max_actions_per_incident"])
        safe_mode = bool(self.cfg["remediation"]["safe_mode"])

        lower = rca.lower()
        actions: List[Dict[str, Any]] = []

        if "memory pressure" in lower or "oom" in lower:
            actions.append({"tool": "restart_service", "args": {"service": svc}})
            actions.append({"tool": "scale_service", "args": {"service": svc, "replicas": 2}})

        elif "cpu saturation" in lower:
            actions.append({"tool": "scale_service", "args": {"service": svc, "replicas": 3}})

        elif "connection refused" in lower or "unreachable" in lower:
            actions.append({"tool": "restart_service", "args": {"service": svc}})

        elif "auth/secret/config" in lower or "auth" in lower:
            actions.append({"tool": "create_ticket", "args": {"severity": "high", "summary": f"Auth issue suspected in {svc}"}})

        elif "timeout" in lower:
            actions.append({"tool": "restart_service", "args": {"service": svc}})
            actions.append({"tool": "scale_service", "args": {"service": svc, "replicas": 2}})

        else:
            actions.append({"tool": "create_ticket", "args": {"severity": "medium", "summary": f"Uncertain anomaly in {svc}"}})

        if safe_mode:
            pass

        return actions[:max_actions]