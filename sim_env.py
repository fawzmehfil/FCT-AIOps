
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import random
import time
import math


class SimEnv:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.services = cfg["services"]
        self.rng = random.Random(cfg["anomaly"]["random_seed"])
        self.t = 0

        # state per service
        self.state: Dict[str, Dict[str, Any]] = {
            s: {"replicas": 1, "fault": None, "last_restart": 0.0} for s in self.services
        }

        # fault types: representative incident classes
        self.fault_types = ["cpu_spike", "memory_leak", "conn_refused", "timeout", "auth_denied"]

    # ---------------- Telemetry ----------------

    def _load(self) -> float:
        """Smooth load wave + noise (0..1)."""
        wave = 0.5 + 0.35 * math.sin(self.t / 8.0)
        noise = self.rng.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, wave + noise))

    def get_metrics(self) -> List[Dict[str, Any]]:
        wl = self._load()
        out = []

        for svc in self.services:
            st = self.state[svc]
            fault = st["fault"]
            replicas = max(1, int(st["replicas"]))

            # baseline
            cpu = 0.12 + 0.40 * wl
            mem = 0.18 + 0.32 * wl
            lat = 70 + 160 * wl
            err = 0.01 + 0.03 * wl

            # fault effects
            if fault == "cpu_spike":
                cpu = min(1.0, cpu + 0.60)
                lat += 140
                err += 0.07
            elif fault == "memory_leak":
                mem = min(1.0, mem + 0.70)
                lat += 160
                err += 0.08
            elif fault == "conn_refused":
                lat += 260
                err += 0.18
            elif fault == "timeout":
                lat += 230
                err += 0.14
            elif fault == "auth_denied":
                lat += 210
                err += 0.20

            # scaling helps a bit
            cpu = max(0.0, cpu - 0.05 * (replicas - 1))
            lat = max(10.0, lat - 20 * (replicas - 1))
            err = max(0.0, err - 0.02 * (replicas - 1))

            # noise
            cpu = self._clamp(cpu + self.rng.uniform(-0.02, 0.02))
            mem = self._clamp(mem + self.rng.uniform(-0.02, 0.02))
            lat = max(5.0, lat + self.rng.uniform(-10, 10))
            err = self._clamp(err + self.rng.uniform(-0.01, 0.01))

            out.append({
                "service": svc,
                "cpu": cpu,
                "memory": mem,
                "latency_ms": lat,
                "error_rate": err,
                "replicas": replicas,
                "fault": fault
            })

        return out

    def get_logs(self, service: str, tail: int = 50) -> str:
        fault = self.state[service]["fault"]
        if fault is None:
            return f"[{service}] INFO ok\n" * min(8, tail)

        if fault == "conn_refused":
            return (
                f"[{service}] ERROR connect() ECONNREFUSED 127.0.0.1:9090\n"
                f"[{service}] ERROR downstream unavailable\n"
            )
        if fault == "timeout":
            return (
                f"[{service}] ERROR request timed out contacting dependency\n"
                f"[{service}] WARN downstream latency high\n"
            )
        if fault == "auth_denied":
            return (
                f"[{service}] ERROR unauthorized / permission denied\n"
                f"[{service}] ERROR auth failed to db\n"
            )
        if fault == "cpu_spike":
            return (
                f"[{service}] WARN CPU saturated; queue depth increasing\n"
                f"[{service}] WARN p99 latency rising\n"
            )
        if fault == "memory_leak":
            return (
                f"[{service}] WARN memory usage rising steadily\n"
                f"[{service}] ERROR OOMKilled (simulated)\n"
            )

        return f"[{service}] ERROR unknown fault\n"

    def get_traces(self, service: str) -> Dict[str, Any]:
        """Simplified traces: list of error edges."""
        fault = self.state[service]["fault"]
        if fault is None:
            return {"service": service, "errors": []}

        mapping = {
            "conn_refused": [f"{service} -> dependency: connection refused"],
            "timeout": [f"{service} -> dependency: timeout"],
            "auth_denied": [f"{service} -> db: unauthorized"],
            "cpu_spike": [f"{service}: high latency due to CPU saturation"],
            "memory_leak": [f"{service}: OOMKilled"],
        }
        return {"service": service, "errors": mapping.get(fault, ["unknown"])}

    # ---------------- Actions ----------------

    def restart_service(self, service: str) -> Tuple[bool, str]:
        """
        Restart clears transient faults but not auth_denied (simulates config/secret issue).
        """
        self.state[service]["last_restart"] = time.time()
        fault = self.state[service]["fault"]

        if fault is None:
            return True, f"Restarted {service} (already healthy)."

        if fault == "auth_denied":
            return True, f"Restarted {service}, but auth_denied persists (likely config/secret)."

        self.state[service]["fault"] = None
        return True, f"Restarted {service} and cleared fault."

    def scale_service(self, service: str, replicas: int) -> Tuple[bool, str]:
        self.state[service]["replicas"] = max(1, int(replicas))
        return True, f"Scaled {service} to replicas={replicas}."

    def create_ticket(self, severity: str, summary: str) -> Tuple[bool, str]:
        return True, f"Ticket created: severity={severity}, summary={summary}"

    # ---------------- World Progression ----------------

    def step_world(self) -> None:
        """Advance time + sometimes inject new faults."""
        self.t += 1

        if self.rng.random() < 0.12:
            svc = self.rng.choice(self.services)
            if self.state[svc]["fault"] is None:
                self.state[svc]["fault"] = self.rng.choice(self.fault_types)

    @staticmethod
    def _clamp(x: float) -> float:
        return max(0.0, min(1.0, x))