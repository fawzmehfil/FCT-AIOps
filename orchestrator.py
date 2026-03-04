
from __future__ import annotations

from typing import Any, Dict, List
import os
import json
import time
from rich.console import Console
from rich.panel import Panel

from agent import AIOpsAgentLite

console = Console()


class Tools:
    def __init__(self, env: Any):
        self.env = env

    def get_logs(self, service: str) -> str:
        return self.env.get_logs(service)

    def get_traces(self, service: str) -> Dict[str, Any]:
        return self.env.get_traces(service)

    def restart_service(self, service: str) -> Dict[str, Any]:
        ok, msg = self.env.restart_service(service)
        return {
            "ok": ok,
            "tool": "restart_service",
            "service": service,
            "message": msg,
            "ts": time.time(),
        }

    def scale_service(self, service: str, replicas: int) -> Dict[str, Any]:
        ok, msg = self.env.scale_service(service, replicas)
        return {
            "ok": ok,
            "tool": "scale_service",
            "service": service,
            "replicas": replicas,
            "message": msg,
            "ts": time.time(),
        }

    def create_ticket(self, severity: str, summary: str) -> Dict[str, Any]:
        ok, msg = self.env.create_ticket(severity, summary)
        return {
            "ok": ok,
            "tool": "create_ticket",
            "severity": severity,
            "summary": summary,
            "message": msg,
            "ts": time.time(),
        }


class Orchestrator:
    def __init__(self, cfg: dict, env: Any):
        self.cfg = cfg
        self.env = env
        self.tools = Tools(env)
        self.agent = AIOpsAgentLite(cfg)

        self.tick = 0
        self.active_incident: Dict[str, Any] | None = None

        self.art_dir = cfg.get("artifacts_dir", "artifacts")
        os.makedirs(self.art_dir, exist_ok=True)

        self._ticket_last_ts: dict[tuple[str, str], float] = {}
        self._ticket_cooldown_s = 45.0

        self._incident_suppress_until: dict[str, float] = {}
        self._incident_suppress_s = 60.0 

    def step(self) -> None:
        self.tick += 1

        # 1) collect metrics
        metrics = self.env.get_metrics()

        # 2) allow agent to warmup/train
        self.agent.observe_and_maybe_train(metrics)

        # 3) agent decision
        decision = self.agent.decide(self.tools, metrics)

        # 4) normal/healthy tick path
        if not decision.incident:
            console.print(f"[green]{decision.rca}[/green]")
            self._write_snapshot(metrics, note="healthy")
            self.env.step_world()
            return

        # 4.1) incident suppression (prevents repaging every tick after escalation)
        signature = self._incident_signature(decision.service, decision.rca)
        now = time.time()
        suppress_until = self._incident_suppress_until.get(signature, 0.0)
        if now < suppress_until:
            console.print("[green]Suppressed repeat incident (cooldown).[/green]")
            self._write_snapshot(metrics, note="suppressed")
            self.env.step_world()
            return

        # 5) start incident if none active
        if self.active_incident is None:
            self.active_incident = {
                "id": f"inc-{int(time.time())}",
                "started_at": time.time(),
                "status": "OPEN",
                "signature": signature,
                "timeline": [],
            }
            console.print(Panel.fit(f"🚨 Incident started: {self.active_incident['id']}", style="red"))

        console.print(Panel(decision.rca, title="RCA", style="yellow"))

        # 6) remediation
        actions_results: List[Dict[str, Any]] = []
        if self.cfg["remediation"]["auto"]:
            for a in decision.actions:
                res = self._exec_action(a)
                actions_results.append(res)
                console.print(f"[bold]Action[/bold]: {res}")

        metrics_after = self.env.get_metrics()
        self.active_incident["timeline"].append({
            "tick": self.tick,
            "service": decision.service,
            "rca": decision.rca,
            "actions": actions_results,
            "metrics_before": metrics,
            "metrics_after": metrics_after,
        })

        escalated = any(r.get("tool") == "create_ticket" for r in actions_results)
        if escalated:
            self.active_incident["status"] = "ESCALATED"
            console.print("[yellow]Escalated to human via ticket. Marking incident as ESCALATED.[/yellow]")

            self._incident_suppress_until[signature] = time.time() + self._incident_suppress_s

            self._write_incident(self.active_incident)
            self._write_snapshot(metrics_after, note="escalated")
            self.active_incident = None
            self.env.step_world()
            return

        if self._is_healthy(metrics_after):
            self.active_incident["status"] = "RESOLVED"
            console.print(Panel.fit(f"✅ Incident resolved: {self.active_incident['id']}", style="green"))

            self._incident_suppress_until[signature] = time.time() + (self._incident_suppress_s / 2)

            self._write_incident(self.active_incident)
            self.active_incident = None
        else:
            console.print("[red]Still unhealthy after actions.[/red]")

        self._write_snapshot(metrics_after, note="incident")
        self.env.step_world()

    def _incident_signature(self, service: str, rca: str) -> str:
        r = rca.lower()
        if "unauthorized" in r or "auth" in r or "secret" in r:
            key = "auth"
        elif "oom" in r or "memory" in r:
            key = "oom"
        elif "connection refused" in r or "unreachable" in r:
            key = "conn_refused"
        elif "timeout" in r:
            key = "timeout"
        else:
            key = "other"
        return f"{service}:{key}"

    def _exec_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        tool = action["tool"]
        args = action.get("args", {})

        if tool == "restart_service":
            return self.tools.restart_service(**args)

        if tool == "scale_service":
            return self.tools.scale_service(**args)

        if tool == "create_ticket":
            sev = args.get("severity", "medium")
            summary = args.get("summary", "")
            key = (sev, summary)
            now = time.time()

            last = self._ticket_last_ts.get(key, 0.0)
            if now - last < self._ticket_cooldown_s:
                return {
                    "ok": True,
                    "tool": "create_ticket",
                    "severity": sev,
                    "summary": summary,
                    "message": f"Deduped ticket (cooldown {self._ticket_cooldown_s:.0f}s).",
                    "ts": now,
                }

            self._ticket_last_ts[key] = now
            return self.tools.create_ticket(severity=sev, summary=summary)

        return {"ok": False, "tool": tool, "message": "Unknown tool"}

    def _is_healthy(self, metrics: List[Dict[str, Any]]) -> bool:
        for m in metrics:
            if m["error_rate"] > 0.12:
                return False
            if m["latency_ms"] > 260:
                return False
        return True

    def _write_snapshot(self, metrics: List[Dict[str, Any]], note: str) -> None:
        path = os.path.join(self.art_dir, f"snapshot_{self.tick:04d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tick": self.tick, "note": note, "metrics": metrics}, f, indent=2)

    def _write_incident(self, inc: Dict[str, Any]) -> None:
        path = os.path.join(self.art_dir, f"{inc['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(inc, f, indent=2)