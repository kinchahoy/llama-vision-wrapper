"""Test building LLM prompts and simple regeneration via local templates."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_bot_controller import PythonLLMController  # noqa: E402


def test_prompt_building_and_regeneration(capsys):
    controller = PythonLLMController(bot_count=2)
    # Inject example templates for development/testing
    example_templates = [
        """
def bot_function(observation):
    # Simple aggressive-like template for tests
    vis = observation.get('visible_objects', [])
    allowed = observation.get('allowed_signals', [])
    def choose(s): return s if s in allowed else 'none'
    enemies = [o for o in vis if o.get('type')=='enemy']
    if enemies and observation.get('self', {}).get('can_fire', False):
        t = enemies[0]
        return {'action':'fire','target_x': t.get('x'),'target_y': t.get('y'),'signal': choose('firing')}
    return {'action':'rotate','angle': 30.0,'signal': choose('ready')}
""",
    ]
    controller.function_templates = example_templates
    observation = {
        "self": {
            "id": 0,
            "team": 1,
            "x": 5.0,
            "y": 3.0,
            "theta_deg": 90.0,
            "vx": 0.5,
            "vy": 0.0,
            "hp": 95,
            "cooldown_remaining": 0.1,
            "can_fire": True,
            "time": 12.0,
            "tick": 240,
            "alive": True,
            "signal": "ready",
        },
        "visible_objects": [
            {
                "type": "enemy",
                "id": 7,
                "x": 8.0,
                "y": 4.0,
                "distance": 3.2,
                "angle": 30.0,
                "hp": 65,
                "velocity_x": -0.2,
                "velocity_y": 0.1,
            },
            {
                "type": "friend",
                "id": 2,
                "x": 4.0,
                "y": 2.0,
                "distance": 1.5,
                "angle": 200.0,
                "hp": 80,
                "velocity_x": 0.0,
                "velocity_y": 0.0,
                "signal": "need_backup",
            },
            {
                "type": "projectile",
                "x": 6.0,
                "y": 3.5,
                "distance": 1.2,
                "angle": 15.0,
                "velocity_x": -3.0,
                "velocity_y": 0.0,
                "ttl": 0.8,
                "team": "team_0",
            },
            {
                "type": "wall",
                "x": 5.0,
                "y": 0.0,
                "distance": 3.0,
                "angle": 270.0,
            },
        ],
        "params": {
            "proj_speed": 12.0,
            "proj_ttl": 1.5,
            "fire_rate": 2.0,
            "sense_range": 15.0,
        },
        "precomp": {
            "enemy_count": 1,
            "friend_count": 1,
            "enemies_close": 0,
            "enemies_mid": 1,
            "enemies_far": 0,
            "nearest_enemy": {
                "id": 7,
                "x": 8.0,
                "y": 4.0,
                "distance": 3.2,
                "angle": 30.0,
                "hp": 65,
                "aim_x": 8.5,
                "aim_y": 4.2,
                "safe_to_fire": True,
            },
            "nearest_friend": {
                "id": 2,
                "x": 4.0,
                "y": 2.0,
                "distance": 1.5,
                "angle": 200.0,
                "signal": "need_backup",
            },
            "nearest_wall": {
                "distance": 3.0,
                "angle": 270.0,
                "x": 5.0,
                "y": 0.0,
            },
            "incoming_projectiles": {
                "count_threats": 1,
                "min_time_to_approach": 0.3,
                "approach_dir_deg": 180.0,
            },
            "friend_signals": {"need_backup": 1},
        },
        "memory": {"last_enemy_id": 7},
    }

    allowed_signals = ["ready", "firing", "need_backup", "retreating"]
    prompt = controller.format_llm_prompt_from_observation(observation, allowed_signals)
    print("=== LLM Prompt ===\n" + prompt)

    # Fallback (no endpoint configured): should return a valid bot function string
    source = controller.regenerate_function_from_observation(
        observation, allowed_signals=allowed_signals, endpoint_url=None, api_key=None
    )
    print("\n=== Local Template Response ===\n" + source)

    captured = capsys.readouterr()
    assert "LLM Prompt" in captured.out
    assert "def bot_function" in source


    # Add this line to print the captured output to the console
    print(captured.out)

    assert "LLM Prompt" in captured.out
    assert "def bot_function" in source
