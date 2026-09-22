from pa_moap_rl.experiments.audit_preference_repair_checkpoint_trajectory_v2 import (
    select_followup_checkpoints,
)


def _row(arm: str, update: int, positive: float, rank: float) -> dict:
    return {
        "arm_id": arm,
        "update": update,
        "top_action_positive_rate": positive,
        "true_best_action_policy_rank_median": rank,
        "checkpoint_sha256": f"hash-{arm}-{update}",
        "model_version": "test",
    }


def test_followup_selection_is_registered_stable_and_deduplicated() -> None:
    rows = [
        _row("A", 0, 0.0, 30.0),
        _row("A", 10, 0.2, 20.0),
        _row("A", 20, 0.2, 10.0),
        _row("A", 30, 0.1, 5.0),
    ]
    selected = select_followup_checkpoints(rows, {"A": 20})
    assert [row["update"] for row in selected] == [20, 30]
    assert "max_top_action_positive_rate" in selected[0]["selection_reasons"]
    assert "protocol_selected" in selected[0]["selection_reasons"]
    assert selected[1]["selection_reasons"] == "min_median_true_best_rank"


def test_followup_selection_uses_earlier_update_for_exact_ties() -> None:
    rows = [_row("A", 0, 0.0, 10.0), _row("A", 10, 0.0, 10.0)]
    selected = select_followup_checkpoints(rows, {"A": 10})
    assert [row["update"] for row in selected] == [0, 10]
