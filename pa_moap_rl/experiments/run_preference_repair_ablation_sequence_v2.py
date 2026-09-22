"""Run the confirmed pure-PPO A/B/C ablation arms sequentially."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from pa_moap_rl.experiments.run_preference_repair_ablation_v2 import (
    run_arm_from_config,
)


SEQUENCE_RUNNER_VERSION = "preference_repair_ablation_sequence_v1"
ARM_ORDER = ("A", "B", "C")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _output_root(config: dict[str, Any]) -> Path:
    roots = {
        Path(config["arms"][arm]["output_dir"]).parent.resolve()
        for arm in ARM_ORDER
    }
    if len(roots) != 1:
        raise ValueError("All A/B/C output directories must share one parent.")
    return roots.pop()


def run_sequence(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("status") != "user_confirmed":
        raise PermissionError("The A/B/C protocol must be user-confirmed before launch.")
    if list(config.get("execution", {}).get("arm_order", [])) != list(ARM_ORDER):
        raise ValueError("The confirmed protocol arm order must be A, B, C.")

    output_root = _output_root(config)
    state_path = output_root / "sequence_state.json"
    existing_arm_outputs = [
        Path(config["arms"][arm]["output_dir"])
        for arm in ARM_ORDER
        if Path(config["arms"][arm]["output_dir"]).exists()
        and any(Path(config["arms"][arm]["output_dir"]).iterdir())
    ]
    if existing_arm_outputs:
        raise FileExistsError(
            "Fresh sequence launch refuses existing arm outputs: "
            + ", ".join(str(item) for item in existing_arm_outputs)
        )
    if state_path.exists():
        raise FileExistsError(
            f"Sequence state already exists; inspect before any restart: {state_path}"
        )

    state: dict[str, Any] = {
        "sequence_runner_version": SEQUENCE_RUNNER_VERSION,
        "protocol_id": config["protocol_id"],
        "protocol_config": str(path),
        "protocol_config_sha256": _sha256(path),
        "process_id": os.getpid(),
        "status": "running",
        "arm_order": list(ARM_ORDER),
        "current_arm": None,
        "completed_arms": [],
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
    }
    _write_state(state_path, state)
    print(json.dumps(state, ensure_ascii=False, sort_keys=True), flush=True)

    try:
        for arm in ARM_ORDER:
            state["current_arm"] = arm
            state["arm_started_at_utc"] = _utc_now()
            state["updated_at_utc"] = _utc_now()
            _write_state(state_path, state)
            print(f"START arm={arm} at={state['arm_started_at_utc']}", flush=True)
            result = run_arm_from_config(path, arm_id=arm)
            state["completed_arms"].append(
                {
                    "arm_id": arm,
                    "finished_at_utc": _utc_now(),
                    "run_summary": str(result.get("run_summary", "")),
                    "latest_checkpoint": str(result.get("latest_checkpoint", "")),
                    "best_checkpoint": str(result.get("best_checkpoint", "")),
                }
            )
            state["updated_at_utc"] = _utc_now()
            _write_state(state_path, state)
            print(f"COMPLETE arm={arm}", flush=True)
        state["status"] = "completed"
        state["current_arm"] = None
        state["finished_at_utc"] = _utc_now()
        state["updated_at_utc"] = _utc_now()
        _write_state(state_path, state)
        print("COMPLETE sequence=A,B,C", flush=True)
        return state
    except BaseException as error:
        state["status"] = "failed"
        state["failed_arm"] = state.get("current_arm")
        state["error_type"] = type(error).__name__
        state["error"] = str(error)
        state["updated_at_utc"] = _utc_now()
        _write_state(state_path, state)
        print(
            f"FAILED arm={state.get('current_arm')} error={type(error).__name__}: {error}",
            flush=True,
        )
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pa_moap_rl/configs/preference_repair_ablation_protocol_v2.yaml",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_sequence(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
