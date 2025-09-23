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


def test_visibility_rays_does_not_change_outcome_small_arena():
    # Baseline with default rays (24 as of now, but pass explicitly)
    base = _run_battle_module.run_python_battle(
        seed=7,
        max_duration=2.0,
        arena_size=(5.0, 5.0),
        bots_per_side=1,
        verbose=False,
        visibility_rays=12,
    )

    assert isinstance(base, BattleData)

    # 4x more rays
    more = _run_battle_module.run_python_battle(
        seed=7,
        max_duration=2.0,
        arena_size=(5.0, 5.0),
        bots_per_side=1,
        verbose=False,
        visibility_rays=48,
    )

    assert isinstance(more, BattleData)

    # Outcomes (winner/termination) must match
    assert base.metadata.winner == more.metadata.winner
    assert base.metadata.reason == more.metadata.reason
    assert base.metadata.total_ticks == more.metadata.total_ticks

    # Final HP distribution must be identical
    assert base.summary.final_hp == more.summary.final_hp

