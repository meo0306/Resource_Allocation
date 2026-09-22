'''Tests for chunked checkpoint evaluation input alignment.'''

from pa_moap_rl.experiments.evaluate_checkpoint import _slice_evaluation_inputs


def test_slice_preserves_full_run_scenario_phase() -> None:
    paths = list(range(10))
    scenarios = ['a', 'b', 'c']
    selected_paths, rotated = _slice_evaluation_inputs(
        paths,
        scenarios,
        offset=4,
        limit=3,
    )

    assert selected_paths == [4, 5, 6]
    assert [rotated[index % len(rotated)] for index in range(3)] == ['b', 'c', 'a']
