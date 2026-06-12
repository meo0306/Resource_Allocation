"""Tests for .sm parsing."""

import importlib

import pa_moap_rl
from pa_moap_rl.data.parse_sm import parse_sm


def test_package_imports() -> None:
    assert pa_moap_rl.__version__ == "0.1.0"


def test_phase0_modules_import() -> None:
    module_names = [
        "pa_moap_rl.data.parse_sm",
        "pa_moap_rl.configs.loader",
        "pa_moap_rl.data.selection_loader",
        "pa_moap_rl.data.build_assignment_instance",
        "pa_moap_rl.data.convert_pre_data",
        "pa_moap_rl.data.instance_schema",
        "pa_moap_rl.data.loader",
        "pa_moap_rl.envs.method_assignment_env",
        "pa_moap_rl.models.encoders",
        "pa_moap_rl.models.actor_critic",
        "pa_moap_rl.solvers.random_solver",
        "pa_moap_rl.solvers.greedy_solver",
        "pa_moap_rl.solvers.local_search_solver",
        "pa_moap_rl.solvers.ppo_solver",
        "pa_moap_rl.utils.scoring",
        "pa_moap_rl.utils.masks",
        "pa_moap_rl.utils.metrics",
        "pa_moap_rl.utils.seed",
        "pa_moap_rl.utils.io",
        "pa_moap_rl.experiments.train_ppo",
        "pa_moap_rl.experiments.train_ppo_batch",
        "pa_moap_rl.experiments.run_batch",
        "pa_moap_rl.experiments.run_ablation",
        "pa_moap_rl.experiments.analyze_results",
    ]
    for module_name in module_names:
        assert importlib.import_module(module_name)


def test_parse_example_sm_extracts_content_two_fields() -> None:
    sm = parse_sm("examples/inst_V60_w6-10_s670487.sm")

    assert sm.instance_name == "inst_V60_w6-10_s670487"
    assert sm.num_nodes_total == 60
    assert len(sm.nodes) == 60
    assert sm.nodes[0].original_id == 1
    assert sm.nodes[0].category_name == "理论"
    assert sm.nodes[0].cognitive_load == 0.205881
    assert sm.category_id.shape == (60,)
    assert sm.cognitive_load.shape == (60,)
    assert sm.id_mapping[1] == 0
    assert sm.capacities.shape == (3,)
