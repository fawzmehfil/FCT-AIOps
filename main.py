
import time
import yaml
from rich.console import Console
from rich.panel import Panel

from sim_env import SimEnv
from orchestrator import Orchestrator

console = Console()

def load_cfg(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()

    console.print(Panel.fit("Fawz AIOps Agent", title="aiops-agent"))

    env = SimEnv(cfg)
    orch = Orchestrator(cfg, env)

    ticks = int(cfg["run"]["ticks"])
    dt = float(cfg["run"]["tick_seconds"])

    for t in range(ticks):
        console.print(f"\n[bold]Tick {t+1}/{ticks}[/bold]")
        orch.step()
        time.sleep(dt)

    console.print("\n[bold green]Done.[/bold green]")
    console.print(f"Artifacts written to ./{cfg['artifacts_dir']}/")

if __name__ == "__main__":
    main()