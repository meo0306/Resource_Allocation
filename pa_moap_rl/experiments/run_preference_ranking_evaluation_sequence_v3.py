'''Run D/E controlled-cross inference and summarize confirmed gates.'''

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from pa_moap_rl.experiments.evaluate_freeze_candidate_response_v2 import run
from pa_moap_rl.experiments.summarize_preference_ranking_ablation_v3 import (
    summarize,
)


SEQUENCE_VERSION = 'preference_ranking_controlled_cross_sequence_v1'
CONFIGS = {
    'D': 'pa_moap_rl/configs/preference_ranking_response_arm_D_v3.yaml',
    'E': 'pa_moap_rl/configs/preference_ranking_response_arm_E_v3.yaml',
}
SUMMARY_CONFIG = (
    'pa_moap_rl/configs/preference_ranking_ablation_evaluation_v3.yaml'
)
STATE_PATH = Path(
    'results/four_topology_b075_pure_ppo_ranking_ablation_v1/'
    'controlled_cross_evaluation_v1/evaluation_sequence_state.json'
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    temporary.replace(STATE_PATH)


def main() -> int:
    if STATE_PATH.exists():
        existing = json.loads(STATE_PATH.read_text(encoding='utf-8'))
        raise FileExistsError(
            'D/E evaluation state already exists with status '
            + str(existing.get('status'))
        )
    state = {
        'sequence_version': SEQUENCE_VERSION,
        'status': 'running',
        'process_id': os.getpid(),
        'started_at_utc': _now(),
        'current_arm': None,
        'completed_arms': [],
        'training_executed': False,
    }
    _write(state)
    try:
        for arm_id, config in CONFIGS.items():
            state['current_arm'] = arm_id
            state['updated_at_utc'] = _now()
            _write(state)
            print('START arm=' + arm_id, flush=True)
            result = run(config)
            state['completed_arms'].append(
                {
                    'arm_id': arm_id,
                    'checkpoint_update': result['checkpoint_update'],
                    'finished_at_utc': _now(),
                }
            )
            _write(state)
            print('COMPLETE arm=' + arm_id, flush=True)
        decision = summarize(SUMMARY_CONFIG)
        state['status'] = 'completed'
        state['current_arm'] = None
        state['recommended_arm'] = decision['gate_decision']['recommended_arm']
        state['eligible_arms'] = decision['gate_decision']['eligible_arms']
        state['finished_at_utc'] = _now()
        state['updated_at_utc'] = _now()
        _write(state)
        print(json.dumps(state, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        state['status'] = 'failed'
        state['failed_arm'] = state.get('current_arm')
        state['error_type'] = type(error).__name__
        state['error'] = str(error)
        state['updated_at_utc'] = _now()
        _write(state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
