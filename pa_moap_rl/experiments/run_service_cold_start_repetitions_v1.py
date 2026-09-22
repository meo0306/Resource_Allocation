"""Measure Stage7 PPO cold start in independent interpreter processes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


RUNNER_VERSION = "service_cold_start_repetitions_v1"


def _parse_json_stdout(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    if start < 0:
        raise ValueError(f"Cold probe did not emit JSON: {stdout[-500:]}")
    value = json.loads(stdout[start:])
    if not isinstance(value, dict):
        raise ValueError("Cold probe output must be a JSON object.")
    return value


def run(config_path: str | Path) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repetitions = int(config["measurement"]["cold_start_repetitions"])
    output_dir = Path(config["run"]["output_dir"]).resolve()
    output_path = output_dir / "cold_start_measurements.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite cold-start evidence: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    measurements: list[dict[str, object]] = []
    for repetition in range(repetitions):
        started = time.perf_counter()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pa_moap_rl.experiments.run_service_efficiency_v1",
                "--config",
                str(config_path),
                "--cold-probe",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
        process_wall_sec = time.perf_counter() - started
        probe = _parse_json_stdout(completed.stdout)
        measurements.append(
            {
                "repetition": repetition,
                "subprocess_wall_sec": process_wall_sec,
                "probe": probe,
                "stderr": completed.stderr,
            }
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "repetitions": repetitions,
        "measurements": measurements,
    }
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pa_moap_rl/configs/service_efficiency_benchmark_four_topology_v1_run02.yaml",
    )
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
