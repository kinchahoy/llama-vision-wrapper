"""Integration tests for executing example bot templates with the Python runner."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import example_llm_gen_control_code
from bot_runner_python import PythonFunctionRunner


def test_runner_executes_example_template():
    runner = PythonFunctionRunner(sandbox_enabled=True)
    template_source = example_llm_gen_control_code._get_default_templates()[0]

    assert runner.compile_bot_function(0, template_source)

    runner.bot_memory[0] = {"attack_mode": "hunter", "confidence": 0.82}

    observation = {
        "self": {
            "id": 0,
            "team": 0,
            "x": 2.0,
            "y": 1.0,
            "can_fire": True,
        },
        "visible_objects": [
            {
                "type": "enemy",
                "id": 5,
                "x": 7.5,
                "y": 1.5,
                "distance": 6.0,
                "angle": 25.0,
                "hp": 80,
                "velocity_x": 0.0,
                "velocity_y": 0.0,
            }
        ],
        "params": {
            "proj_speed": 12.0,
        },
    }

    action = runner.execute_bot_function(0, observation)

    assert action is not None
    assert action["action"] == "fire"
    assert action["signal"] == "firing"
    assert action["target_x"] == 7.5
    assert action["target_y"] == 1.5
    assert 0 not in runner.bot_memory


def test_runner_stops_long_running_bot():
    runner = PythonFunctionRunner(sandbox_enabled=False)
    runner.max_execution_time = 0.01

    slow_bot_source = """
def bot_function(observation):
    while True:
        pass
"""

    assert runner.compile_bot_function(0, slow_bot_source)

    observation = {
        "self": {"id": 0, "team": 0},
        "visible_objects": [],
        "params": {},
    }

    result = runner.execute_bot_function(0, observation)

    assert result is None
    bot_meta = runner.bot_functions[0]
    assert bot_meta.error_count == 1
    assert runner.execution_stats["total_timeouts"] == 1


def test_runner_blocks_file_access_attempt():
    runner = PythonFunctionRunner(sandbox_enabled=True)

    restricted_bot_source = """
def bot_function(observation):
    import os
    os.listdir('/')
    return {'action': 'rotate', 'angle': 0.0, 'signal': 'ready'}
"""

    assert runner.compile_bot_function(1, restricted_bot_source)

    observation = {
        "self": {"id": 1, "team": 0},
        "visible_objects": [],
        "params": {},
    }

    result = runner.execute_bot_function(1, observation)

    assert result is None
    bot_meta = runner.bot_functions[1]
    assert bot_meta.error_count == 1
    assert runner.execution_stats["total_errors"] >= 1
