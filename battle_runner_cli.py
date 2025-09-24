"""
Python Function Battle Runner
Run battles using Python functions instead of DSL programs.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Protocol, Tuple, cast

from battle_arena import Arena
from llm_bot_controller import PythonLLMController
from battle_types import (
    BattleData,
    BattleMetadata,
    BattleSummary,
    BotFunctionSummary,
    RunnerStats,
)
from pydantic import ValidationError


FUNCTION_UPDATE_PERIOD = 0.2  # seconds between Python function evaluations
# Mock-LLM code rewrite cadence (tweakable)
LLM_REWRITE_ENABLED = True
LLM_REWRITE_PERIOD = 4.0  # seconds between re-generating bot code


class _RunnerModule(Protocol):
    PythonFunctionRunner: Any


class _GodotViewerModule(Protocol):
    def run_godot_viewer(self, battle_file: str) -> None:
        ...


class _PandaViewerModule(Protocol):
    def run_3d_viewer(self, battle_file: str) -> None:
        ...


class _Viewer2DModule(Protocol):
    BattleViewer: Any


def _load_module(module_name: str, file_name: str) -> ModuleType:
    module_path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_name}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_python_function_runner():
    """Load PythonFunctionRunner from the renamed runner module."""
    module = _load_module("bot_runner_python", "bot_runner_python.py")
    runner_module = cast(_RunnerModule, module)
    if not hasattr(runner_module, "PythonFunctionRunner"):
        raise AttributeError("PythonFunctionRunner attribute not found in module")
    return runner_module.PythonFunctionRunner


PythonFunctionRunner = _load_python_function_runner()


def run_python_battle(
    seed: int = 42,
    max_duration: float = 60.0,
    spawn_config: Optional[Dict[str, Any]] = None,
    bot_functions: Optional[Dict[int, str]] = None,
    verbose: bool = True,
    arena_size: Optional[Tuple[float, float]] = None,
    bots_per_side: Optional[int] = None,
    sim_unsafe: bool = False,
    visibility_rays: Optional[int] = None,
    use_wasmer: bool = False,
) -> BattleData:
    """
    Run a complete battle simulation using Python functions for bot control.

    Args:
        seed: Random seed for deterministic simulation
        max_duration: Maximum battle duration in seconds
        spawn_config: Bot spawning configuration
        bot_functions: Custom bot function source code by bot_id
        verbose: Whether to print progress
        arena_size: Arena dimensions (width, height)
        bots_per_side: Number of bots per team

    Returns:
        Battle data dictionary with results and timeline
    """

    # Create arena
    arena = Arena(seed, spawn_config, arena_size, bots_per_side)

    # Create Python function runner and LLM controller
    runner = PythonFunctionRunner(sandbox_enabled=not sim_unsafe, use_wasmer=use_wasmer)
    llm = PythonLLMController(arena.BOT_COUNT)
    if visibility_rays is not None and visibility_rays > 0:
        llm.visibility_ray_count = int(visibility_rays)

    if verbose and sim_unsafe:
        print("Simulation unsafe mode enabled: bot functions execute without RestrictedPython and may use Numba acceleration.")
    
    if verbose and use_wasmer:
        print("Wasmer mode enabled: bot functions will be compiled to WebAssembly.")

    if verbose:
        print(
            f"Starting {arena.BOTS_PER_SIDE}v{arena.BOTS_PER_SIDE} Python function battle (seed={seed})"
        )
        print(f"Arena size: {arena.ARENA_SIZE[0]}x{arena.ARENA_SIZE[1]}m")

    # Compile bot functions
    compilation_success = {}
    for bot_id in range(arena.BOT_COUNT):
        if bot_functions and bot_id in bot_functions:
            # Use custom function if provided
            source = bot_functions[bot_id]
        else:
            # Use LLM-generated function
            source = llm.get_bot_function_source(bot_id)

        success = runner.compile_bot_function(bot_id, source)
        compilation_success[bot_id] = success

        if verbose:
            status = "✓" if success else "✗"
            print(f"  Bot {bot_id}: {status}")

    # Check if any bots failed compilation
    failed_bots = [
        bot_id for bot_id, success in compilation_success.items() if not success
    ]
    if failed_bots:
        print(f"Warning: {len(failed_bots)} bot(s) failed compilation: {failed_bots}")

    if verbose:
        print("Battle starting...")

    start_time = time.time()
    last_llm_rewrite_time = -1e9

    steps_per_control = arena.physics_steps_per_control
    function_tick_interval = max(
        1, int(round(FUNCTION_UPDATE_PERIOD / arena.DT_CONTROL))
    )

    # Main simulation loop
    while True:
        # Physics step (cadence governed by arena.DT_PHYSICS)
        arena.step_physics()

        # Control step cadence derived from cached ratio
        if arena.physics_tick % steps_per_control == 0:
            arena.step_control()

            # Periodically regenerate bot source code via LLM (or local templates) and recompile
            if LLM_REWRITE_ENABLED and (arena.time - last_llm_rewrite_time) >= LLM_REWRITE_PERIOD:
                for bot_id in range(arena.BOT_COUNT):
                    if not arena._is_bot_alive(bot_id):
                        continue
                    try:
                        new_source = llm.regenerate_bot_function(
                            arena,
                            bot_id,
                            allowed_signals=runner.allowed_signals,
                            endpoint_url=None,  # set URL + API key to enable remote LLM
                            api_key=None,
                        )
                        llm.update_bot_function(bot_id, new_source)
                        runner.compile_bot_function(bot_id, new_source)
                        if verbose:
                            print(f"  [LLM] Regenerated bot {bot_id} code")
                    except Exception as e:
                        if verbose:
                            print(f"  [LLM] Failed to regenerate bot {bot_id}: {e}")
                last_llm_rewrite_time = arena.time

            # Execute Python functions for each bot based on configured cadence
            if arena.tick % function_tick_interval == 0:
                for bot_id in range(arena.BOT_COUNT):
                    if not arena._is_bot_alive(bot_id):
                        continue

                    # Generate inputs for Python function
                    observation = llm.build_observation(arena, bot_id)

                    # Execute bot function
                    action = runner.execute_bot_function(bot_id, observation)

                    # Apply action to arena
                    if action:
                        arena.set_single_bot_action(bot_id, action)

                        # Update bot's signal if provided in action
                        signal = action.get("signal")
                        if signal is not None and bot_id in arena.bot_data:
                            arena.bot_data[bot_id].signal = signal

                    llm.record_bot_action(arena, bot_id, action, bool(action))

            # Handle firing for all bots
            for bot_id in range(arena.BOT_COUNT):
                if (
                    arena._is_bot_alive(bot_id)
                    and hasattr(arena, "fire_command")
                    and arena.fire_command[bot_id]
                ):
                    arena.try_fire_projectile(bot_id)

            # Log state periodically for visualization
            if arena.tick % 12 == 0:
                arena.log_state()

        # Check for battle end
        is_over, winner, reason = arena.is_battle_over(max_duration)
        if is_over:
            arena.log_state()  # Final state
            break

    elapsed = time.time() - start_time
    assert winner is not None

    runner_stats = RunnerStats.model_validate(runner.get_runner_stats())

    if verbose:
        print(f"Battle complete: Team {winner} wins by {reason}")
        print(f"Duration: {arena.time:.1f}s simulated, {elapsed:.2f}s real")

        print(f"Function executions: {runner_stats.total_executions}")
        print(f"Timeouts: {runner_stats.total_timeouts}")
        print(f"Errors: {runner_stats.total_errors}")
        if runner_stats.total_executions > 0:
            avg_ms = runner_stats.avg_execution_time * 1000
            print(f"Average execution time: {avg_ms:.2f}ms")

        print("\nBot performance:")
        for bot_id in range(arena.BOT_COUNT):
            bot_info = runner.get_bot_function_info(bot_id)
            if bot_info:
                last_exec_ms = bot_info["last_execution_time"] * 1000
                print(
                    f"  Bot {bot_id}: "
                    f"{bot_info['error_count']} errors, "
                    f"{last_exec_ms:.2f}ms last exec"
                )

    metadata = BattleMetadata(
        seed=seed,
        duration=round(arena.time, 2),
        winner=f"team_{winner}",
        reason=reason,
        arena_size=arena.ARENA_SIZE,
        total_ticks=arena.tick,
        real_time=round(elapsed, 2),
        walls=arena.get_walls(),
        control_system="python_functions",
        compilation_success=compilation_success,
        runner_stats=runner_stats,
        sandbox_enabled=not sim_unsafe,
        use_wasmer=use_wasmer,
    )

    summary = _generate_python_battle_summary(arena, runner, runner_stats)

    battle_data = BattleData(
        metadata=metadata,
        timeline=list(arena.battle_log),
        summary=summary,
    )

    # Clean up
    arena.cleanup()

    return battle_data


def _generate_python_battle_summary(
    arena: Arena,
    runner: PythonFunctionRunner,
    runner_stats: RunnerStats,
) -> BattleSummary:
    """Generate comprehensive battle summary for Python function battles."""
    from battle_arena import _generate_battle_summary

    base_summary = _generate_battle_summary(arena)

    bot_function_info: Dict[int, BotFunctionSummary] = {}
    for bot_id in range(arena.BOT_COUNT):
        info = runner.get_bot_function_info(bot_id)
        if not info:
            continue
        bot_function_info[bot_id] = BotFunctionSummary(
            function_errors=info["error_count"],
            avg_exec_time_ms=round(info["last_execution_time"] * 1000, 2),
            version=info["version"],
        )

    python_summary = base_summary.model_copy(
        update={
            "control_system": "python_functions",
            "function_executions": runner_stats.total_executions,
            "function_timeouts": runner_stats.total_timeouts,
            "function_errors": runner_stats.total_errors,
            "avg_function_time_ms": round(runner_stats.avg_execution_time * 1000, 2),
            "bot_functions": bot_function_info,
            "numba_enabled": runner_stats.numba_available,
            "sandbox_enabled": runner_stats.sandbox_enabled,
        }
    )

    return python_summary


def run_interactive_viewer(
    battle_file: str, use_3d: bool = False, use_godot: bool = False
):
    """Launch interactive viewer with a saved Python battle JSON file."""
    if use_godot:
        godot_module = cast(
            _GodotViewerModule,
            _load_module("viewer_3d_godot", "viewer_3d_godot.py"),
        )
        godot_module.run_godot_viewer(battle_file)
        return

    if use_3d:
        panda_module = cast(
            _PandaViewerModule,
            _load_module("viewer_3d_panda", "viewer_3d_panda.py"),
        )
        panda_module.run_3d_viewer(battle_file)
        return

    viewer_2d_module = cast(
        _Viewer2DModule, _load_module("viewer_2d_pygame", "viewer_2d_pygame.py")
    )
    if not hasattr(viewer_2d_module, "BattleViewer"):
        raise AttributeError("viewer_2d_pygame.BattleViewer is missing")
    BattleViewer = viewer_2d_module.BattleViewer

    print(f"\n=== Interactive Viewer: {battle_file} ===")

    try:
        with open(battle_file, "r") as f:
            raw_data = json.load(f)
        battle_data = BattleData.model_validate(raw_data)
    except FileNotFoundError:
        print(f"Error: Battle file '{battle_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{battle_file}'")
        return
    except ValidationError as exc:
        print(f"Error: Battle data validation failed: {exc}")
        return

    print("Launching interactive viewer...")
    print("Controls:")
    print("  SPACE = Play/Pause")
    print("  ←/→ = Step frame by frame")
    print("  +/- = Adjust speed")
    print("  R = Reset to start")
    print("  F = Toggle FOV display")
    print("  T = Toggle projectile trails")
    print("  Q = Quit")
    print("  Click bots for info")
    print("  Drag timeline to scrub")

    # Launch viewer
    viewer = BattleViewer(battle_data)
    viewer.run()


def main():
    """Command line interface for running Python function battles."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "viewer":
        # Handle viewer command
        if len(sys.argv) > 2:
            battle_file = sys.argv[2]
            use_3d = len(sys.argv) > 3 and sys.argv[3] == "--3d"
            use_godot = len(sys.argv) > 3 and sys.argv[3] == "--godot3d"
            run_interactive_viewer(battle_file, use_3d, use_godot)
        else:
            print(
                "Usage: uv run python battle_runner_cli.py viewer <battle_file.json> [--3d|--godot3d]"
            )
            print("Examples:")
            print("  uv run python battle_runner_cli.py viewer battle_output.json")
            print("  uv run python battle_runner_cli.py viewer battle_output.json --3d")
            print(
                "  uv run python battle_runner_cli.py viewer battle_output.json --godot3d"
            )
        return

    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM Battle Sim with Python functions"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["battle", "viewer"],
        default="battle",
        help="Command to run",
    )
    parser.add_argument("battle_file", nargs="?", help="Battle file for viewer mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--duration", type=float, default=60.0, help="Max battle duration (seconds)"
    )
    parser.add_argument(
        "--arena-size", type=str, default="20,20", help='Arena size "width,height"'
    )
    parser.add_argument("--bots-per-side", type=int, default=2, help="Bots per team")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--3d", action="store_true", help="Use 3D viewer instead of 2D")
    parser.add_argument("--godot3d", action="store_true", help="Use Godot 4 3D viewer")
    parser.add_argument(
        "--sim-unsafe",
        action="store_true",
        help="Disable RestrictedPython sandbox for bot functions (unsafe, enables direct execution and Numba acceleration)",
    )
    parser.add_argument(
        "--use-wasmer",
        action="store_true",
        help="Use Wasmer WebAssembly runtime for bot function execution (experimental)",
    )
    parser.add_argument(
        "--visibility-rays",
        type=int,
        default=None,
        help="Override ray count for visibility fan (constant across runs)",
    )

    args = parser.parse_args()

    if args.command == "viewer":
        if args.battle_file:
            run_interactive_viewer(
                args.battle_file,
                getattr(args, "3d", False),
                getattr(args, "godot3d", False),
            )
        else:
            print("Error: viewer command requires a battle JSON file")
        print(
            "Usage: uv run python battle_runner_cli.py viewer <battle_file.json> [--3d|--godot3d]"
        )
        return

    # Parse arena size (accept both comma and 'x' separators)
    if "x" in args.arena_size:
        arena_size = tuple(map(float, args.arena_size.split("x")))
    else:
        arena_size = tuple(map(float, args.arena_size.split(",")))

    # Run battle
    battle_data = run_python_battle(
        seed=args.seed,
        max_duration=args.duration,
        arena_size=arena_size,
        bots_per_side=args.bots_per_side,
        verbose=not args.quiet,
        sim_unsafe=args.sim_unsafe,
        visibility_rays=args.visibility_rays,
        use_wasmer=args.use_wasmer,
    )

    # Save results
    output_file = args.output or f"python_battle_{args.seed}.json"
    with open(output_file, "w") as f:
        json.dump(battle_data.model_dump_jsonable(), f, indent=2)
    if not args.quiet:
        print(f"Battle data saved to {output_file}")

    # Print summary
    if not args.quiet:
        summary = battle_data.summary
        metadata = battle_data.metadata

        print("\n=== PYTHON BATTLE SUMMARY ===")
        print(f"Winner: {metadata.winner} ({metadata.reason})")
        print(f"Duration: {metadata.duration}s")
        print(f"Total shots: {summary.total_shots}")
        print(f"Hit rate: {summary.hit_rate:.1%}")
        if summary.mvp.bot_id is not None:
            print(f"MVP: Bot {summary.mvp.bot_id} (score: {summary.mvp.score})")
        else:
            print("MVP: n/a")

        function_executions = summary.function_executions or 0
        avg_function_time = summary.avg_function_time_ms or 0.0
        print(f"Function executions: {function_executions}")
        print(f"Average function time: {avg_function_time:.2f}ms")
        print("\nTo view this battle:")
        print(f"  2D: uv run python battle_runner_cli.py viewer {output_file}")
        print(f"  3D: uv run python battle_runner_cli.py viewer {output_file} --3d")
        print(
            f"  Godot: uv run python battle_runner_cli.py viewer {output_file} --godot3d"
        )


if __name__ == "__main__":
    main()
