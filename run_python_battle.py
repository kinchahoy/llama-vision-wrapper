"""
Python Function Battle Runner
Run battles using Python functions instead of DSL programs.
"""

import time
import json
from typing import Dict, Optional, Tuple

from battle_sim import Arena
from python_runner import PythonFunctionRunner
from python_llm import PythonLLMController


def run_python_battle(
    seed: int = 42,
    max_duration: float = 60.0,
    spawn_config: Dict = None,
    bot_functions: Dict[int, str] = None,
    verbose: bool = True,
    arena_size: Optional[Tuple[float, float]] = None,
    bots_per_side: Optional[int] = None,
) -> Dict:
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
    runner = PythonFunctionRunner()
    llm = PythonLLMController(arena.BOT_COUNT)

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
            personality = llm.get_bot_personality(bot_id)
            status = "✓" if success else "✗"
            print(f"  Bot {bot_id} ({personality}): {status}")

    # Check if any bots failed compilation
    failed_bots = [
        bot_id for bot_id, success in compilation_success.items() if not success
    ]
    if failed_bots:
        print(f"Warning: {len(failed_bots)} bot(s) failed compilation: {failed_bots}")

    if verbose:
        print("Battle starting...")

    start_time = time.time()

    # Main simulation loop
    while True:
        # Physics step (240Hz)
        arena.step_physics()

        # Control step (120Hz)
        if arena.physics_tick % 2 == 0:  # 240Hz -> 120Hz
            arena.step_control()

            # Execute Python functions for each bot (~5Hz for LLM updates)
            if arena.tick % 24 == 0:  # 120Hz -> 5Hz for function execution
                for bot_id in range(arena.BOT_COUNT):
                    if not arena._is_bot_alive(bot_id):
                        continue

                    # Generate inputs for Python function
                    visible_objects = llm.generate_visible_objects(arena, bot_id)
                    move_history = llm.generate_move_history(arena, bot_id)

                    # Execute bot function
                    action = runner.execute_bot_function(
                        bot_id, visible_objects, move_history
                    )

                    # Apply action to arena
                    if action:
                        arena.set_single_bot_action(bot_id, action)

                        # Update bot's signal if provided in action
                        signal = action.get("signal")
                        if signal is not None and bot_id in arena.bot_data:
                            arena.bot_data[bot_id].signal = signal

            # Handle firing for all bots
            for bot_id in range(arena.BOT_COUNT):
                if (
                    arena._is_bot_alive(bot_id)
                    and hasattr(arena, "fire_command")
                    and arena.fire_command[bot_id]
                ):
                    arena.try_fire_projectile(bot_id)

            # Log state periodically for visualization
            if arena.tick % 12 == 0:  # ~10Hz logging
                arena.log_state()

        # Check for battle end
        is_over, winner, reason = arena.is_battle_over(max_duration)
        if is_over:
            arena.log_state()  # Final state
            break

    elapsed = time.time() - start_time

    if verbose:
        print(f"Battle complete: Team {winner} wins by {reason}")
        print(f"Duration: {arena.time:.1f}s simulated, {elapsed:.2f}s real")

        # Print performance stats
        stats = runner.get_runner_stats()
        print(f"Function executions: {stats['total_executions']}")
        print(f"Timeouts: {stats['total_timeouts']}")
        print(f"Errors: {stats['total_errors']}")
        if stats["total_executions"] > 0:
            print(f"Average execution time: {stats['avg_execution_time'] * 1000:.2f}ms")

        # Print individual bot stats
        print("\nBot performance:")
        for bot_id in range(arena.BOT_COUNT):
            bot_info = runner.get_bot_function_info(bot_id)
            if bot_info:
                personality = llm.get_bot_personality(bot_id)
                print(
                    f"  Bot {bot_id} ({personality}): "
                    f"{bot_info['error_count']} errors, "
                    f"{bot_info['last_execution_time'] * 1000:.2f}ms last exec"
                )

    # Generate battle data
    battle_data = {
        "metadata": {
            "seed": seed,
            "duration": round(arena.time, 2),
            "winner": f"team_{winner}",
            "reason": reason,
            "arena_size": arena.ARENA_SIZE,
            "walls": arena.get_walls(),
            "total_ticks": arena.tick,
            "real_time": round(elapsed, 2),
            "control_system": "python_functions",
            "compilation_success": compilation_success,
            "runner_stats": runner.get_runner_stats(),
        },
        "timeline": arena.battle_log,
        "summary": _generate_python_battle_summary(arena, runner, llm),
    }

    # Clean up
    arena.cleanup()

    return battle_data


def _generate_python_battle_summary(
    arena: Arena, runner: PythonFunctionRunner, llm: PythonLLMController
) -> Dict:
    """Generate comprehensive battle summary for Python function battles."""
    from battle_sim import _generate_battle_summary

    # Get base summary
    base_summary = _generate_battle_summary(arena)

    # Add Python function specific stats
    runner_stats = runner.get_runner_stats()

    # Get bot function info
    bot_function_info = {}
    for bot_id in range(arena.BOT_COUNT):
        info = runner.get_bot_function_info(bot_id)
        if info:
            bot_function_info[bot_id] = {
                "personality": llm.get_bot_personality(bot_id),
                "function_errors": info["error_count"],
                "avg_exec_time_ms": round(info["last_execution_time"] * 1000, 2),
                "version": info["version"],
            }

    # Combine summaries
    python_summary = base_summary.copy()
    python_summary.update(
        {
            "control_system": "python_functions",
            "function_executions": runner_stats["total_executions"],
            "function_timeouts": runner_stats["total_timeouts"],
            "function_errors": runner_stats["total_errors"],
            "avg_function_time_ms": round(runner_stats["avg_execution_time"] * 1000, 2),
            "bot_functions": bot_function_info,
            "numba_enabled": runner_stats["numba_available"],
        }
    )

    return python_summary


def run_interactive_viewer(battle_file: str, use_3d: bool = False, use_godot: bool = False):
    """Launch interactive viewer with a saved Python battle JSON file."""
    if use_godot:
        import importlib.util

        spec = importlib.util.spec_from_file_location("graphicsgodot3d", "graphicsgodot3d.py")
        graphicsgodot3d = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graphicsgodot3d)
        graphicsgodot3d.run_godot_viewer(battle_file)
        return
        
    if use_3d:
        import importlib.util

        spec = importlib.util.spec_from_file_location("graphics3d", "graphics3d.py")
        graphics3d = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graphics3d)
        graphics3d.run_3d_viewer(battle_file)
        return

    from graphics import BattleViewer

    print(f"\n=== Interactive Viewer: {battle_file} ===")

    try:
        with open(battle_file, "r") as f:
            battle_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Battle file '{battle_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{battle_file}'")
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
                "Usage: uv run python run_python_battle.py viewer <battle_file.json> [--3d|--godot3d]"
            )
            print("Examples:")
            print("  uv run python run_python_battle.py viewer battle_output.json")
            print("  uv run python run_python_battle.py viewer battle_output.json --3d")
            print("  uv run python run_python_battle.py viewer battle_output.json --godot3d")
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

    args = parser.parse_args()

    if args.command == "viewer":
        if args.battle_file:
            run_interactive_viewer(args.battle_file, getattr(args, "3d", False), getattr(args, "godot3d", False))
        else:
            print("Error: viewer command requires a battle JSON file")
            print(
                "Usage: uv run python run_python_battle.py viewer <battle_file.json> [--3d|--godot3d]"
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
    )

    # Save results
    output_file = args.output or f"python_battle_{args.seed}.json"
    with open(output_file, "w") as f:
        json.dump(battle_data, f, indent=2)
    if not args.quiet:
        print(f"Battle data saved to {output_file}")

    # Print summary
    if not args.quiet:
        summary = battle_data["summary"]
        metadata = battle_data["metadata"]

        print(f"\n=== PYTHON BATTLE SUMMARY ===")
        print(f"Winner: {metadata['winner']} ({metadata['reason']})")
        print(f"Duration: {metadata['duration']}s")
        print(f"Total shots: {summary['total_shots']}")
        print(f"Hit rate: {summary['hit_rate']:.1%}")
        print(f"MVP: Bot {summary['mvp']['bot_id']} (score: {summary['mvp']['score']})")
        print(f"Function executions: {summary['function_executions']}")
        print(f"Average function time: {summary['avg_function_time_ms']:.2f}ms")
        print(f"\nTo view this battle:")
        print(f"  2D: uv run python run_python_battle.py viewer {output_file}")
        print(f"  3D: uv run python run_python_battle.py viewer {output_file} --3d")
        print(f"  Godot: uv run python run_python_battle.py viewer {output_file} --godot3d")


if __name__ == "__main__":
    main()
