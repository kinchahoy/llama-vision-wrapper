"""
Python Function Runner for Battle Sim
Compiles and executes LLM-generated Python functions via Numba JIT with timeout handling.
"""

import copy
import importlib
import math
import queue
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Optional

import numpy as np

# Allowlist of modules bot functions may import when running in the restricted sandbox
_SANDBOX_IMPORT_ALLOWLIST = {
    "math",
    "random",
    "statistics",
    "itertools",
    "functools",
    "collections",
    "heapq",
    "bisect",
    "numpy",
}

# Try to import numba, fall back to no JIT if unavailable
try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        """Fallback decorator that does nothing if numba unavailable"""

        def decorator(func):
            return func

        return decorator


try:
    from RestrictedPython import compile_restricted
    from RestrictedPython.Guards import guarded_iter_unpack_sequence as _rp_guarded_iter_unpack_sequence
    from RestrictedPython.Guards import guarded_unpack_sequence as _rp_guarded_unpack_sequence

    RESTRICTED_AVAILABLE = True
except ImportError:
    compile_restricted = None  # type: ignore[assignment]
    _rp_guarded_iter_unpack_sequence = None  # type: ignore[assignment]
    _rp_guarded_unpack_sequence = None  # type: ignore[assignment]
    RESTRICTED_AVAILABLE = False


def _no_print(*args, **kwargs):
    """Silence print output from restricted code."""
    return None


def _guarded_getattr(obj, name, default=None):
    if isinstance(name, str) and name.startswith("_"):
        raise AttributeError(name)
    if default is None:
        return getattr(obj, name)
    return getattr(obj, name, default)


def _guarded_getitem(obj, index):
    return obj[index]


def _guarded_getiter(obj):
    return iter(obj)


def _guarded_iter_unpack_sequence(obj, spec, *_):
    if _rp_guarded_iter_unpack_sequence is not None:
        return _rp_guarded_iter_unpack_sequence(obj, spec, _guarded_getiter)
    iterator = iter(obj)
    for _ in range(int(spec)):
        yield next(iterator)


def _guarded_unpack_sequence(obj, spec, *_):
    if _rp_guarded_unpack_sequence is not None:
        return _rp_guarded_unpack_sequence(obj, spec, _guarded_getiter)
    iterator = iter(obj)
    return tuple(next(iterator) for _ in range(int(spec)))


def _write_guard(obj):
    return obj


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Permit imports from a small allowlist of safe modules inside the sandbox."""

    if level != 0:
        raise ImportError("Relative imports are not supported in sandboxed bot functions")

    root_name = name.split(".", 1)[0]
    if root_name not in _SANDBOX_IMPORT_ALLOWLIST:
        raise ImportError(f"Module '{name}' is not permitted in sandboxed bot functions")

    module = importlib.import_module(name)

    # mimic CPython behaviour: if fromlist is empty return the top-level module
    if not fromlist:
        return module

    return module


class ScratchPad:
    """Limited mutable scratch space for bot functions."""

    __slots__ = ("_data", "_limit")

    def __init__(self, limit: int = 16):
        self._data: Dict[str, Any] = {}
        self._limit = max(0, limit)

    def set(self, key: str, value: Any) -> bool:
        if not isinstance(key, str) or not key or len(key) > 32:
            return False
        if not isinstance(value, (int, float, bool, str)) and value is not None:
            return False
        if key not in self._data and len(self._data) >= self._limit:
            return False
        self._data[key] = value
        return True

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def items(self):  # pragma: no cover - inspection helper
        return tuple(self._data.items())

    def __repr__(self) -> str:
        return f"ScratchPad(size={len(self._data)}, limit={self._limit})"


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


def _noop_bot_function(*args: Any, **kwargs: Any) -> Any:
    return None


class PythonFunctionRunner:
    """Manages compilation and execution of LLM-generated Python bot functions."""

    MEMORY_LIMIT = 16
    SCRATCH_LIMIT = 16

    def __init__(self, sandbox_enabled: bool = True):
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
        self.bot_memory: Dict[int, Dict[str, Any]] = {}
        self.sandbox_enabled = sandbox_enabled and RESTRICTED_AVAILABLE

    def _execute_with_timeout(
        self, func: Callable, *args
    ) -> Any:
        """Execute function with a wall-clock timeout using a worker thread."""

        result_queue: queue.Queue = queue.Queue(maxsize=1)

        def target():
            try:
                result = func(*args)
            except Exception as exc:  # noqa: BLE001 - propagate bot errors
                result_queue.put((False, exc))
            else:
                result_queue.put((True, result))

        worker = threading.Thread(target=target, daemon=True)
        worker.start()

        try:
            success, payload = result_queue.get(timeout=self.max_execution_time)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Function execution exceeded {self.max_execution_time}s timeout"
            ) from exc
        finally:
            worker.join(timeout=0)

        if success:
            return payload
        raise payload

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
                    bot_id,
                    BotFunction("", _noop_bot_function, 0.0, 0, bot_id),
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
            self.bot_memory.pop(bot_id, None)
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
                    bot_id,
                    BotFunction("", _noop_bot_function, 0.0, 0, bot_id),
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
            self.bot_memory.pop(bot_id, None)

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
            if self.sandbox_enabled:
                func = self._compile_with_restrictedpython(source_code)
            else:
                func = self._compile_with_legacy_exec(source_code)

            if func is None:
                return None

            self._validate_function_signature(func)

            if NUMBA_AVAILABLE and not self.sandbox_enabled:
                try:
                    return jit(nopython=False, cache=True)(func)
                except Exception as jit_error:
                    print(f"JIT compilation failed, using regular Python: {jit_error}")
                    return func

            return func

        except Exception as e:
            print(f"Function compilation error: {e}")
            return None

    def _compile_with_restrictedpython(self, source_code: str) -> Optional[Callable]:
        if not RESTRICTED_AVAILABLE or compile_restricted is None:
            return None

        byte_code = compile_restricted(source_code, "<bot_function>", "exec")
        restricted_globals = self._build_restricted_globals()
        exec(byte_code, restricted_globals)
        func = restricted_globals.get("bot_function")
        if not callable(func):
            raise ValueError("Function must define callable 'bot_function'")
        return func

    def _compile_with_legacy_exec(self, source_code: str) -> Optional[Callable]:
        namespace: Dict[str, Any] = {
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
            "isinstance": isinstance,
            "enumerate": enumerate,
            "zip": zip,
            "any": any,
            "all": all,
        },
        }

        exec(source_code, namespace)

        func = namespace.get("bot_function")
        if not callable(func):
            raise ValueError("Function must be named 'bot_function'")
        return func

    def _build_restricted_globals(self) -> Dict[str, Any]:
        safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "round": round,
            "float": float,
            "int": int,
            "bool": bool,
            "str": str,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "any": any,
            "all": all,
            "isinstance": isinstance,
            "__import__": _restricted_import,
        }

        return {
            "__builtins__": safe_builtins,
            "math": math,
            "np": np,
            "_print_": _no_print,
            "_getattr_": _guarded_getattr,
            "_getitem_": _guarded_getitem,
            "_getiter_": _guarded_getiter,
            "_iter_unpack_sequence_": _guarded_iter_unpack_sequence,
            "_unpack_sequence_": _guarded_unpack_sequence,
            "_write_": _write_guard,
            "ScratchPad": ScratchPad,
        }

    def _prepare_observation(self, bot_id: int, observation: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(observation, dict):
            raise ValueError("Observation must be a dictionary")

        payload = copy.deepcopy(observation)
        payload["memory"] = copy.deepcopy(self.bot_memory.get(bot_id, {}))
        payload["allowed_signals"] = tuple(self.allowed_signals)
        payload["scratchpad"] = ScratchPad(self.SCRATCH_LIMIT)
        return self._make_read_only(payload)

    def _make_read_only(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            frozen = {key: self._make_read_only(value) for key, value in obj.items()}
            return MappingProxyType(frozen)
        if isinstance(obj, list):
            return tuple(self._make_read_only(value) for value in obj)
        if isinstance(obj, set):
            return frozenset(self._make_read_only(value) for value in obj)
        if isinstance(obj, tuple):
            return tuple(self._make_read_only(value) for value in obj)
        if isinstance(obj, ScratchPad):
            return obj
        return obj

    def _sanitize_memory(self, memory: Any) -> Dict[str, Any]:
        if not isinstance(memory, dict):
            return {}

        sanitized: Dict[str, Any] = {}
        for key, value in memory.items():
            if len(sanitized) >= self.MEMORY_LIMIT:
                break
            if not isinstance(key, str) or not key or len(key) > 32:
                continue

            if isinstance(value, (int, float, bool)) or value is None:
                sanitized[key] = value
            elif isinstance(value, str):
                sanitized[key] = value[:64]

        return sanitized

    def _validate_function_signature(self, func: Callable):
        """Validate that function has correct signature."""
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if len(params) != 1:
            raise ValueError(
                f"Function must accept a single 'observation' parameter, got {len(params)}: {params}"
            )

        if params[0] != "observation":
            raise ValueError(
                "Function parameter must be named 'observation' to receive world state"
            )

    def execute_bot_function(
        self, bot_id: int, observation: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        Execute a bot's compiled function with timeout protection.

        Args:
            bot_id: Bot identifier
            observation: Full observation payload for the bot

        Returns:
            Action dictionary or None if execution failed
        """
        if bot_id not in self.bot_functions:
            return None

        bot_func = self.bot_functions[bot_id]
        start_time = time.time()

        try:
            observation_payload = self._prepare_observation(bot_id, observation)
            result = self._execute_with_timeout(
                bot_func.compiled_func,
                observation_payload,
            )

            execution_time = time.time() - start_time
            bot_func.last_execution_time = execution_time

            # Update statistics
            self.execution_stats["total_executions"] += 1
            self._update_avg_execution_time(execution_time)

            # Validate and sanitize result
            sanitized_result, sanitized_memory = self._validate_and_sanitize_action(result)

            if sanitized_memory:
                self.bot_memory[bot_id] = sanitized_memory
            elif bot_id in self.bot_memory:
                del self.bot_memory[bot_id]

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

    def _validate_and_sanitize_action(
        self, action: Any
    ) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Validate and sanitize action output from bot function.

        Args:
            action: Raw action from bot function

        Returns:
            Tuple of (sanitized action or None, sanitized memory dict)
        """
        sanitized_memory: Dict[str, Any] = {}

        if isinstance(action, dict) and "memory" in action:
            sanitized_memory = self._sanitize_memory(action.get("memory"))
            action = {k: v for k, v in action.items() if k != "memory"}

        if not isinstance(action, dict):
            return None, sanitized_memory

        action_type = action.get("action")
        if action_type not in ["move", "fire", "rotate", "dodge"]:
            return None, sanitized_memory

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
                return None, sanitized_memory

        elif action_type == "fire":
            target_x = action.get("target_x")
            target_y = action.get("target_y")
            if isinstance(target_x, (int, float)) and isinstance(
                target_y, (int, float)
            ):
                sanitized["target_x"] = float(target_x)
                sanitized["target_y"] = float(target_y)
            else:
                return None, sanitized_memory

        elif action_type == "rotate":
            angle = action.get("angle")
            if isinstance(angle, (int, float)):
                sanitized["angle"] = float(angle) % 360  # Normalize to 0-360
            else:
                return None, sanitized_memory

        elif action_type == "dodge":
            direction = action.get("direction")
            if isinstance(direction, (int, float)):
                sanitized["direction"] = float(direction) % 360  # Normalize to 0-360
            else:
                return None, sanitized_memory

        return sanitized, sanitized_memory

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
        stats["sandbox_enabled"] = self.sandbox_enabled
        return stats

    def clear_bot_function(self, bot_id: int):
        """Remove a bot's compiled function."""
        if bot_id in self.bot_functions:
            del self.bot_functions[bot_id]
        self.bot_memory.pop(bot_id, None)

    def clear_all_functions(self):
        """Clear all compiled functions and cache."""
        self.bot_functions.clear()
        self.compile_cache.clear()
        self.bot_memory.clear()
        self.execution_stats = {
            "total_executions": 0,
            "total_timeouts": 0,
            "total_errors": 0,
            "avg_execution_time": 0.0,
        }
