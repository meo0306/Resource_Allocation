"""Build the deterministic objective-independent Stage7 service scenario bank."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pa_moap_rl.data.scenarios import generate_scenario_bank, save_scenario_bank


BUILDER_VERSION = "service_scenario_bank_builder_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(*, output: Path, count: int, seed: int) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite service scenario bank: {output}")
    scenarios = generate_scenario_bank(
        count=int(count),
        seed=int(seed),
        held_out=False,
        objective_independent=True,
        access_status="development_service_efficiency",
    )
    profile_pairs = {
        (
            tuple(float(value) for value in scenario.student_profile),
            tuple(float(value) for value in scenario.teacher_profile),
        )
        for scenario in scenarios
    }
    if len(profile_pairs) != len(scenarios):
        raise ValueError("Generated service scenarios are not profile-unique.")
    save_scenario_bank(scenarios, output)
    metadata = {
        "schema_version": 1,
        "builder_version": BUILDER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(scenarios),
        "profile_unique_count": len(profile_pairs),
        "seed": int(seed),
        "objective_independent": True,
        "access_status": "development_service_efficiency",
        "scenario_bank": str(output),
        "scenario_bank_sha256": _sha256(output),
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data_processed/four_topology_exploratory_v2/service_efficiency_scenarios_128_v1.json",
    )
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    result = build(output=Path(args.output), count=args.count, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
