"""
Python Function Runner for Battle Sim
Compiles and executes LLM-generated Python functions via Numba JIT with timeout handling.
"""

import math
import time
import types
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import threading
import signal
from contextlib import contextmanager

# Try to import numba, fall back to no JIT if unavailable
try:
    from numba import jit, types as nb_types

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        """Fallback decorator that does nothing if numba unavailable"""

        def decorator(func):
            return func

        return decorator


@dataclass
class BotFunction:
    """Compiled bot function with metadata."""

    source_code: str
    compiled_func: Callable
    compile_time: float
    version: int
    bot_id: int
    error_count: int = 0
    last_execution_time: float = 0.0


class TimeoutError(Exception):
    """Raised when function execution exceeds timeout."""

    pass


@contextmanager
def timeout_context(duration: float):
    """Context manager for function execution timeout."""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function execution exceeded {duration}s timeout")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(duration * 1000000) // 1000000 + 1)  # Convert to seconds, round up

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# Standard signal vocabulary for bot communication
ALLOWED_SIGNALS = {
    # Status/State signals
    "none": "No specific status",
    "ready": "Ready for action",
    "low_hp": "Health critical (<30 HP)",
    "reloading": "Weapon cooling down",
    # Combat action signals
    "attacking": "Engaging enemy target",
    "firing": "Currently shooting",
    "flanking": "Moving to flank enemy",
    "retreating": "Falling back from combat",
    "advancing": "Moving forward to engage",
    "cover_fire": "Providing suppressive fire",
    # Tactical coordination signals
    "need_backup": "Requesting immediate assistance",
    "enemy_spotted": "Enemy contact established",
    "holding_position": "Maintaining current location",
    "moving_to_cover": "Relocating for protection",
    "watching_flank": "Covering team's side/rear",
    "regrouping": "Moving to rally point",
    # Team coordination signals
    "follow_me": "Request team to follow",
    "wait": "Hold current position",
    "go_go_go": "Execute coordinated advance",
    "spread_out": "Increase team dispersion",
    "focus_fire": "Concentrate fire on target",
    "disengage": "Break contact and withdraw",
}


class PythonFunctionRunner:
    """Manages compilation and execution of LLM-generated Python bot functions."""

    def __init__(self):
        self.bot_functions: Dict[int, BotFunction] = {}
        self.execution_stats = {
            "total_executions": 0,
            "total_timeouts": 0,
            "total_errors": 0,
            "avg_execution_time": 0.0,
        }
        self.max_execution_time = 0.01  # 10ms limit
        self.compile_cache = {}  # Cache compiled functions by source code hash
        self.allowed_signals = list(ALLOWED_SIGNALS.keys())

    def compile_bot_function(
        self, bot_id: int, source_code: str, force_recompile: bool = False
    ) -> bool:
        """
        Compile a Python function for a bot.

        Args:
            bot_id: Bot identifier
            source_code: Python function source code
            force_recompile: Force recompilation even if cached

        Returns:
            True if compilation succeeded, False otherwise
        """
        # Check if we already have this exact function compiled
        code_hash = hash(source_code)
        if not force_recompile and code_hash in self.compile_cache:
            cached_func = self.compile_cache[code_hash]
            version = (
                self.bot_functions.get(
                    bot_id, BotFunction("", None, 0, 0, bot_id)
                ).version
                + 1
            )
            self.bot_functions[bot_id] = BotFunction(
                source_code=source_code,
                compiled_func=cached_func,
                compile_time=0.0,  # Use cached
                version=version,
                bot_id=bot_id,
            )
            return True

        start_time = time.time()

        try:
            # Validate and compile the function
            compiled_func = self._compile_function(source_code)
            if compiled_func is None:
                return False

            compile_time = time.time() - start_time

            # Store compiled function
            version = (
                self.bot_functions.get(
                    bot_id, BotFunction("", None, 0, 0, bot_id)
                ).version
                + 1
            )
            self.bot_functions[bot_id] = BotFunction(
                source_code=source_code,
                compiled_func=compiled_func,
                compile_time=compile_time,
                version=version,
                bot_id=bot_id,
            )

            # Cache for reuse
            self.compile_cache[code_hash] = compiled_func

            return True

        except Exception as e:
            print(f"Failed to compile function for bot {bot_id}: {e}")
            return False

    def _compile_function(self, source_code: str) -> Optional[Callable]:
        """
        Compile Python source code into a callable function.

        Args:
            source_code: Python function source code

        Returns:
            Compiled function or None if compilation failed
        """
        try:
            # Create a restricted namespace for execution
            namespace = {
                "math": math,
                "np": np,
                "__builtins__": {
                    "len": len,
                    "range": range,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "sum": sum,
                    "round": round,
                    "float": float,
                    "int": int,
                    "bool": bool,
                    "str": str,
                    "list": list,
                    "dict": dict,
                    "enumerate": enumerate,
                    "zip": zip,
                    "any": any,  # Allow any() builtin function
                    "all": all,  # Allow all() builtin function as well
                    "__import__": __import__,  # Allow imports for math module
                },
            }

            # Execute the function definition
            exec(source_code, namespace)

            # Find the bot function (should be named 'bot_function')
            if "bot_function" not in namespace:
                raise ValueError("Function must be named 'bot_function'")

            func = namespace["bot_function"]

            # Validate function signature
            self._validate_function_signature(func)

            # Apply JIT compilation if available
            if NUMBA_AVAILABLE:
                try:
                    # Create a JIT-compiled version
                    jit_func = jit(nopython=False, cache=True)(func)
                    return jit_func
                except Exception as jit_error:
                    print(f"JIT compilation failed, using regular Python: {jit_error}")
                    return func
            else:
                return func

        except Exception as e:
            print(f"Function compilation error: {e}")
            return None

    def _validate_function_signature(self, func: Callable):
        """Validate that function has correct signature."""
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if len(params) != 3:
            raise ValueError(
                f"Function must have exactly 3 parameters, got {len(params)}: {params}"
            )

        if (
            params[0] != "visible_objects"
            or params[1] != "move_history"
            or params[2] != "allowed_signals"
        ):
            raise ValueError(
                f"Function parameters must be 'visible_objects', 'move_history', and 'allowed_signals', got: {params}"
            )

    def execute_bot_function(
        self, bot_id: int, visible_objects: List[Dict], move_history: List[Dict]
    ) -> Optional[Dict]:
        """
        Execute a bot's compiled function with timeout protection.

        Args:
            bot_id: Bot identifier
            visible_objects: List of visible objects
            move_history: List of recent moves

        Returns:
            Action dictionary or None if execution failed
        """
        if bot_id not in self.bot_functions:
            return None

        bot_func = self.bot_functions[bot_id]
        start_time = time.time()

        try:
            # Execute function with timeout
            with timeout_context(self.max_execution_time):
                result = bot_func.compiled_func(
                    visible_objects, move_history, self.allowed_signals
                )

            execution_time = time.time() - start_time
            bot_func.last_execution_time = execution_time

            # Update statistics
            self.execution_stats["total_executions"] += 1
            self._update_avg_execution_time(execution_time)

            # Validate and sanitize result
            sanitized_result = self._validate_and_sanitize_action(result)
            return sanitized_result

        except TimeoutError:
            self.execution_stats["total_timeouts"] += 1
            bot_func.error_count += 1
            print(
                f"Bot {bot_id} function execution timed out (>{self.max_execution_time}s)"
            )
            return None

        except Exception as e:
            self.execution_stats["total_errors"] += 1
            bot_func.error_count += 1
            print(f"Bot {bot_id} function execution error: {e}")
            return None

    def _validate_and_sanitize_action(self, action: Any) -> Optional[Dict]:
        """
        Validate and sanitize action output from bot function.

        Args:
            action: Raw action from bot function

        Returns:
            Sanitized action dictionary or None if invalid
        """
        if not isinstance(action, dict):
            return None

        action_type = action.get("action")
        if action_type not in ["move", "fire", "rotate", "dodge"]:
            return None

        sanitized = {"action": action_type}

        # Validate and include signal if provided
        signal = action.get("signal")
        if signal is not None:
            if isinstance(signal, str) and signal.strip() in self.allowed_signals:
                sanitized["signal"] = signal.strip()
            else:
                sanitized["signal"] = "none"  # Default if invalid/unknown signal
        else:
            sanitized["signal"] = "none"  # Default if not provided

        if action_type == "move":
            target_x = action.get("target_x")
            target_y = action.get("target_y")
            if isinstance(target_x, (int, float)) and isinstance(
                target_y, (int, float)
            ):
                sanitized["target_x"] = float(target_x)
                sanitized["target_y"] = float(target_y)
            else:
                return None

        elif action_type == "fire":
            target_x = action.get("target_x")
            target_y = action.get("target_y")
            if isinstance(target_x, (int, float)) and isinstance(
                target_y, (int, float)
            ):
                sanitized["target_x"] = float(target_x)
                sanitized["target_y"] = float(target_y)
            else:
                return None

        elif action_type == "rotate":
            angle = action.get("angle")
            if isinstance(angle, (int, float)):
                sanitized["angle"] = float(angle) % 360  # Normalize to 0-360
            else:
                return None

        elif action_type == "dodge":
            direction = action.get("direction")
            if isinstance(direction, (int, float)):
                sanitized["direction"] = float(direction) % 360  # Normalize to 0-360
            else:
                return None

        return sanitized

    def _update_avg_execution_time(self, execution_time: float):
        """Update running average of execution times."""
        total = self.execution_stats["total_executions"]
        current_avg = self.execution_stats["avg_execution_time"]

        # Running average calculation
        self.execution_stats["avg_execution_time"] = (
            current_avg * (total - 1) + execution_time
        ) / total

    def get_bot_function_info(self, bot_id: int) -> Optional[Dict]:
        """Get information about a bot's compiled function."""
        if bot_id not in self.bot_functions:
            return None

        bot_func = self.bot_functions[bot_id]
        return {
            "bot_id": bot_id,
            "version": bot_func.version,
            "compile_time": bot_func.compile_time,
            "error_count": bot_func.error_count,
            "last_execution_time": bot_func.last_execution_time,
            "source_lines": len(bot_func.source_code.splitlines()),
            "has_jit": NUMBA_AVAILABLE,
        }

    def get_runner_stats(self) -> Dict:
        """Get overall runner statistics."""
        stats = self.execution_stats.copy()
        stats["bot_count"] = len(self.bot_functions)
        stats["cache_size"] = len(self.compile_cache)
        stats["numba_available"] = NUMBA_AVAILABLE
        return stats

    def clear_bot_function(self, bot_id: int):
        """Remove a bot's compiled function."""
        if bot_id in self.bot_functions:
            del self.bot_functions[bot_id]

    def clear_all_functions(self):
        """Clear all compiled functions and cache."""
        self.bot_functions.clear()
        self.compile_cache.clear()
        self.execution_stats = {
            "total_executions": 0,
            "total_timeouts": 0,
            "total_errors": 0,
            "avg_execution_time": 0.0,
        }


# Example bot functions for testing
EXAMPLE_AGGRESSIVE_FUNCTION = '''
def bot_function(visible_objects, move_history, allowed_signals):
    """Aggressive bot that charges at nearest enemy."""
    import math
    
    # Find nearest enemy
    nearest_enemy = None
    min_distance = float('inf')
    
    for obj in visible_objects:
        if obj.get('type') == 'enemy' and obj.get('distance', float('inf')) < min_distance:
            min_distance = obj['distance']
            nearest_enemy = obj
    
    # If enemy found, move towards them and fire
    if nearest_enemy:
        if min_distance < 3.0:
            # Too close, dodge
            return {'action': 'dodge', 'direction': nearest_enemy['angle'] + 90, 'signal': 'retreating'}
        elif min_distance < 8.0:
            # In range, fire
            return {'action': 'fire', 'target_x': nearest_enemy['x'], 'target_y': nearest_enemy['y'], 'signal': 'firing'}
        else:
            # Move closer
            return {'action': 'move', 'target_x': nearest_enemy['x'], 'target_y': nearest_enemy['y'], 'signal': 'attacking'}
    
    # No enemies, search
    return {'action': 'rotate', 'angle': 45.0, 'signal': 'ready'}
'''

EXAMPLE_DEFENSIVE_FUNCTION = '''
def bot_function(visible_objects, move_history, allowed_signals):
    """Defensive bot that keeps distance and fires accurately."""
    import math
    
    # Check for incoming projectiles
    for obj in visible_objects:
        if obj.get('type') == 'projectile' and obj.get('distance', float('inf')) < 2.0:
            # Dodge incoming projectile
            return {'action': 'dodge', 'direction': obj['angle'] + 90, 'signal': 'moving_to_cover'}
    
    # Find enemies
    enemies = [obj for obj in visible_objects if obj.get('type') == 'enemy']
    
    if enemies:
        # Sort by distance
        enemies.sort(key=lambda e: e.get('distance', float('inf')))
        target = enemies[0]
        
        distance = target.get('distance', float('inf'))
        
        if distance < 5.0:
            # Too close, back away
            retreat_angle = (target['angle'] + 180) % 360
            retreat_x = target['x'] + 10 * math.cos(math.radians(retreat_angle))
            retreat_y = target['y'] + 10 * math.sin(math.radians(retreat_angle))
            return {'action': 'move', 'target_x': retreat_x, 'target_y': retreat_y, 'signal': 'retreating'}
        elif distance < 12.0:
            # Good range, fire
            return {'action': 'fire', 'target_x': target['x'], 'target_y': target['y'], 'signal': 'cover_fire'}
        else:
            # Move closer but cautiously
            approach_x = target['x'] + 2 * math.cos(math.radians(target['angle'] + 180))
            approach_y = target['y'] + 2 * math.sin(math.radians(target['angle'] + 180))
            return {'action': 'move', 'target_x': approach_x, 'target_y': approach_y, 'signal': 'advancing'}
    
    # No enemies visible, rotate to search
    return {'action': 'rotate', 'angle': 90.0, 'signal': 'watching_flank'}
'''
