"""Smoke test to ensure python battle runner produces expected summary data."""

import importlib.util
import sys
from pathlib import Path

from battle_types import BattleData


def _load_module(name: str, filename: str):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    module_path = project_root / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module '{name}' from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_run_battle_module = _load_module("battle_runner_cli", "battle_runner_cli.py")


def test_python_battle_summary_structure():
    battle_data = _run_battle_module.run_python_battle(
        seed=1,
        max_duration=2.0,
        arena_size=(10.0, 10.0),
        bots_per_side=1,
        verbose=False,
    )

    assert isinstance(battle_data, BattleData)

    summary = battle_data.summary
    metadata = battle_data.metadata

    expected_summary_keys = {
        "control_system",
        "function_executions",
        "function_timeouts",
        "function_errors",
        "avg_function_time_ms",
        "bot_functions",
    }
    missing_keys = expected_summary_keys - set(summary.keys())
    assert not missing_keys, f"Missing summary keys: {missing_keys}"

    assert metadata["control_system"] == "python_functions"
    assert metadata["winner"].startswith("team_")
    assert metadata["duration"] <= 2.0
