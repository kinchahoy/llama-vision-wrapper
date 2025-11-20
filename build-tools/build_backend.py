"""Hybrid build backend that lets uv_build package the project while ensuring
native llama.cpp artifacts are compiled via CMake (discovered through
scikit-build-core)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from scikit_build_core.cmake import CMake
from scikit_build_core.errors import CMakeNotFoundError
from uv_build import (
    build_editable as _uv_build_editable,
    build_sdist as _uv_build_sdist,
    build_wheel as _uv_build_wheel,
    get_requires_for_build_editable as _uv_get_requires_for_build_editable,
    get_requires_for_build_sdist as _uv_get_requires_for_build_sdist,
    get_requires_for_build_wheel as _uv_get_requires_for_build_wheel,
    prepare_metadata_for_build_editable as _uv_prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel as _uv_prepare_metadata_for_build_wheel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_SRC_DIR = PROJECT_ROOT / "wrapper_src"
PACKAGE_DIR = WRAPPER_SRC_DIR / "llama_insight"
LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
GEN_HELPER_DIR = WRAPPER_SRC_DIR / "gen-helper"

LLAMA_BUILD_DIR = LLAMA_CPP_DIR / "build-uv"
GEN_BUILD_DIR = GEN_HELPER_DIR / "build-uv"

PACKAGED_LIB_DIR = PACKAGE_DIR / "libs"
BUILD_METADATA_FILE = PACKAGED_LIB_DIR / "build-metadata.json"

LIB_EXT = (
    "dll"
    if sys.platform.startswith("win")
    else "dylib"
    if sys.platform == "darwin"
    else "so"
)

BACKEND_FLAGS = {
    "cuda": ["-DGGML_CUDA=ON"],
    "metal": ["-DGGML_METAL=ON"],
    "vulkan": ["-DGGML_VULKAN=ON"],
    "hip": ["-DGGML_HIP=ON"],
    "kleidiai": ["-DGGML_CPU_KLEIDIAI=ON"],
    "cpu": [],
    "custom": [],
}

DEFAULT_BACKEND = "cpu"
DEFAULT_LLAMA_REPO = "https://github.com/ggerganov/llama.cpp.git"


def get_requires_for_build_wheel(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return _uv_get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return _uv_get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return _uv_get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: dict[str, Any] | None = None
) -> str:
    return _uv_prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(
    metadata_directory: str, config_settings: dict[str, Any] | None = None
) -> str:
    return _uv_prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ensure_native_artifacts(config_settings)
    return _uv_build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ensure_native_artifacts(config_settings)
    return _uv_build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _uv_build_sdist(sdist_directory, config_settings)


def _ensure_native_artifacts(config_settings: dict[str, Any] | None) -> None:
    skip_native = _read_bool_setting(
        config_settings,
        "skip-native-build",
        env=("LLAMA_INSIGHT_SKIP_NATIVE_BUILD",),
    )
    if skip_native:
        _log("Skipping native compilation (existing artifacts assumed).")
        _ensure_required_libs_present()
        return

    _ensure_llama_cpp_sources()
    _stage_headers()
    backend = _select_backend(config_settings)
    extra_flags = _collect_extra_flags(config_settings)
    _maybe_restore_cached_artifacts(backend, extra_flags)
    jobs = _determine_jobs()
    dry_run = _read_bool_setting(
        config_settings, "dry-run", env=("LLAMA_INSIGHT_DRY_RUN",)
    )
    if dry_run:
        _log("Dry-run enabled: llama.cpp compilation will be skipped.")

    if not _native_artifacts_fresh(backend, extra_flags):
        _log("Building llama.cpp + helper shared libraries...")
        _build_llama_cpp(backend, extra_flags, jobs, skip_compile=dry_run)
        _build_generation_helper(jobs)
        libs = _stage_built_libraries()
        _write_build_metadata(backend, extra_flags, libs)
        _store_cached_artifacts(backend, extra_flags, libs)
    else:
        _log("Native artifacts already up-to-date; skipping rebuild.")
    _ensure_required_libs_present()


def _native_artifacts_fresh(backend: str, extra_flags: list[str]) -> bool:
    resolved = _resolve_existing_artifacts()
    if not resolved:
        return False
    metadata, libs = resolved
    fingerprint = _fingerprint(backend, extra_flags, libs)
    return metadata == fingerprint


def _build_llama_cpp(
    backend: str, extra_flags: list[str], jobs: int, *, skip_compile: bool = False
) -> None:
    _apply_patch()
    cmake = _find_cmake()

    configure_cmd = [
        cmake,
        "-S",
        str(LLAMA_CPP_DIR),
        "-B",
        str(LLAMA_BUILD_DIR),
        "-DBUILD_SHARED_LIBS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_BENCHMARKS=OFF",
        "-DLLAMA_BUILD_COMMON=ON",
        "-DLLAMA_BUILD_COMMON_DLL=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
        *BACKEND_FLAGS[backend],
        *extra_flags,
    ]
    _run(configure_cmd, cwd=PROJECT_ROOT)
    if skip_compile:
        _log("Dry-run: skipping `cmake --build` for llama.cpp.")
        return
    _run(
        [cmake, "--build", str(LLAMA_BUILD_DIR), "--parallel", str(jobs)],
        cwd=PROJECT_ROOT,
    )


def _build_generation_helper(jobs: int) -> None:
    cmake = _find_cmake()
    configure_cmd = [
        cmake,
        "-S",
        str(GEN_HELPER_DIR),
        "-B",
        str(GEN_BUILD_DIR),
        f"-DLLAMA_CPP_SOURCE_DIR={LLAMA_CPP_DIR}",
        f"-DLLAMA_CPP_BUILD_DIR={LLAMA_BUILD_DIR}",
    ]
    _run(configure_cmd, cwd=PROJECT_ROOT)
    _run(
        [cmake, "--build", str(GEN_BUILD_DIR), "--parallel", str(jobs)],
        cwd=PROJECT_ROOT,
    )


def _stage_built_libraries() -> list[str]:
    libs = _discover_built_libraries()
    if not libs:
        raise FileNotFoundError(
            f"No compiled libraries found matching lib*.{LIB_EXT} in build directories."
        )
    if PACKAGED_LIB_DIR.exists():
        shutil.rmtree(PACKAGED_LIB_DIR)
    PACKAGED_LIB_DIR.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in libs:
        destination = PACKAGED_LIB_DIR / src.name
        shutil.copy2(src, destination)
        copied.append(src.name)
        _log(f"Packaged {src.name}")
    return copied


def _discover_built_libraries() -> list[Path]:
    def _glob_all(directories: tuple[Path, ...], patterns: tuple[str, ...]) -> Iterator[Path]:
        for directory in directories:
            if not directory.exists():
                continue
            for pattern in patterns:
                yield from directory.glob(pattern)

    base_dirs = (LLAMA_BUILD_DIR / "bin", GEN_BUILD_DIR)
    patterns: list[str] = [f"lib*.{LIB_EXT}"]
    if LIB_EXT == "so":
        patterns.append(f"lib*.{LIB_EXT}.*")  # capture SONAME targets like libfoo.so.0
    libs = {path.name: path for path in _glob_all(base_dirs, tuple(patterns))}

    mtmd_prefix = "" if LIB_EXT == "dll" else "lib"
    mtmd_tag = f"{mtmd_prefix}mtmd"
    if not any(name.startswith(mtmd_tag) for name in libs):
        mtmd_patterns = [f"{mtmd_tag}*.{LIB_EXT}"]
        if LIB_EXT == "so":
            mtmd_patterns.append(f"{mtmd_tag}*.{LIB_EXT}.*")
        mtmd_dirs = (
            LLAMA_BUILD_DIR / "bin",
            LLAMA_BUILD_DIR / "tools" / "mtmd",
        )
        libs.update(
            (candidate.name, candidate)
            for candidate in _glob_all(mtmd_dirs, tuple(mtmd_patterns))
        )
    return [libs[name] for name in sorted(libs)]


def _apply_patch() -> None:
    patch_file = PROJECT_ROOT / "patch_llama_common_for_dynamic.patch"
    if not patch_file.exists():
        _log("Patch file not found; skipping patch step.")
        return

    cmd = ["patch", "-p1", "-N", "--silent", "-r", "-", "-i", str(patch_file)]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"Unable to apply llama.cpp patch ({patch_file}).\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    if completed.returncode == 0:
        _log("Applied llama.cpp shared-library patch.")
    else:
        _log("Patch already applied; continuing.")


def _ensure_llama_cpp_sources() -> None:
    if (LLAMA_CPP_DIR / "CMakeLists.txt").exists():
        return

    git_dir = PROJECT_ROOT / ".git"
    gitmodules = PROJECT_ROOT / ".gitmodules"
    if git_dir.exists() and gitmodules.exists():
        _log("Initializing llama.cpp submodule...")
        _run(
            ["git", "submodule", "update", "--init", "--recursive", "llama.cpp"],
            cwd=PROJECT_ROOT,
        )
        if (LLAMA_CPP_DIR / "CMakeLists.txt").exists():
            return

    # Fallback: clone directly into the tree so we can
    # build wheels even when the submodule is not pre-initialized.
    repo_url = os.environ.get("LLAMA_INSIGHT_LLAMA_CPP_URL") or DEFAULT_LLAMA_REPO
    _clone_llama_cpp(repo_url, LLAMA_CPP_DIR)
    if not (LLAMA_CPP_DIR / "CMakeLists.txt").exists():
        raise FileNotFoundError(
            f"llama.cpp checkout missing at {LLAMA_CPP_DIR} after clone.\n"
            "Ensure llama.cpp is available via git submodule or manual checkout."
        )


def _clone_llama_cpp(repo_url: str, destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Cloning llama.cpp from {repo_url} into {destination} ...")
    _run(
        ["git", "clone", "--depth", "1", "--recursive", repo_url, str(destination)],
        cwd=destination.parent,
    )


def _stage_headers() -> None:
    stage_script = PROJECT_ROOT / "build-tools" / "stage_headers.py"
    if not stage_script.exists():
        _log("Header staging script missing; skipping header sync.")
        return
    python = sys.executable or shutil.which("python3") or shutil.which("python")
    if not python:
        raise RuntimeError(
            "Unable to locate a Python interpreter to run stage_headers.py."
        )
    _log("Staging llama.cpp headers into package data...")
    _run([python, str(stage_script)], cwd=PROJECT_ROOT)


def _find_cmake() -> str:
    try:
        cmake = CMake.default_search()
    except CMakeNotFoundError as err:
        raise RuntimeError(
            "CMake is required to build llama.cpp. "
            "Ensure the 'cmake' Python package (with embedded binaries) "
            "is available or CMake is installed system-wide."
        ) from err
    return str(cmake.cmake_path)


def _select_backend(config_settings: dict[str, Any] | None) -> str:
    env_b = os.environ.get("LLAMA_INSIGHT_BACKEND")
    if env_b:
        backend = env_b.lower()
        source = "env"
    else:
        auto = _auto_detect_backend()
        if auto:
            backend = auto.lower()
            source = "autodetect"
        else:
            backend = DEFAULT_BACKEND
            source = "default"
    if backend not in BACKEND_FLAGS:
        valid = ", ".join(sorted(BACKEND_FLAGS))
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of: {valid}")
    _log_backend_summary(backend, source)
    return backend


def _auto_detect_backend() -> str | None:
    plat = sys.platform
    machine = platform.machine().lower()
    _log("Detecting accelerated backend (Metal / CUDA / Vulkan / HIP / KleidiAI)...")
    for detector in _BACKEND_DETECTORS:
        backend = detector(plat, machine)
        if backend:
            return backend
    _log("No accelerated backend detected; defaulting to CPU.")
    return None


def _detect_metal(plat: str, machine: str) -> str | None:
    if (
        plat == "darwin"
        and machine in {"arm64", "aarch64"}
        and _cmd_ok(["xcrun", "-f", "metal"])
    ):
        _log("Auto-selected backend: metal (macOS + Metal toolchain detected)")
        return "metal"
    return None


def _detect_cuda(plat: str, machine: str) -> str | None:  # noqa: ARG001
    if (
        _cmd_ok(["nvcc", "--version"])
        or _cmd_ok(["nvidia-smi", "-L"])
        or Path("/usr/local/cuda").exists()
    ):
        _log("Auto-selected backend: cuda (CUDA toolkit/driver detected)")
        return "cuda"
    return None


def _detect_vulkan(plat: str, machine: str) -> str | None:  # noqa: ARG001
    if (
        _cmd_ok(["pkg-config", "--exists", "vulkan"])
        or _cmd_ok(["vulkaninfo", "--summary"])
        or Path("/usr/include/vulkan/vulkan.h").exists()
    ):
        _log("Auto-selected backend: vulkan (Vulkan SDK detected)")
        return "vulkan"
    return None


def _detect_hip(plat: str, machine: str) -> str | None:  # noqa: ARG001
    hip_paths = [
        os.environ.get("HIP_PATH"),
        os.environ.get("ROCM_PATH"),
        "/opt/rocm",
    ]
    path_exists = any(Path(path).exists() for path in hip_paths if path)
    if (
        path_exists
        or Path("/opt/rocm/bin/hipcc").exists()
        or _cmd_ok(["hipconfig", "-V"])
        or _cmd_ok(["rocminfo"])
    ):
        _log("Auto-selected backend: hip (ROCm toolkit detected)")
        return "hip"
    return None


def _detect_kleidiai(plat: str, machine: str) -> str | None:
    if (
        plat == "linux"
        and machine in {"arm64", "aarch64"}
        and _supports_kleidiai_backend()
    ):
        _log("Auto-selected backend: kleidiai (Arm KleidiAI features detected)")
        return "kleidiai"
    return None


def _supports_kleidiai_backend() -> bool:
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        return False
    try:
        content = cpuinfo_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    features: set[str] = set()
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().lower() not in {"features", "flags"}:
            continue
        tokens = value.replace(",", " ").split()
        for token in tokens:
            if token:
                features.add(token.strip().lower())
    return bool(features & _KLEIDIAI_FEATURE_TOKENS)


_KLEIDIAI_FEATURE_TOKENS = frozenset({
    "asimddp",
    "dotprod",
    "i8mm",
    "matmulint8",
    "sme",
    "sve",
    "sve2",
})


_BACKEND_DETECTORS = (
    _detect_metal,
    _detect_cuda,
    _detect_vulkan,
    _detect_hip,
    _detect_kleidiai,
)


def _collect_extra_flags(config_settings: dict[str, Any] | None) -> list[str]:
    flag_string = _read_setting(config_settings, "extra-flags")
    env_flags = os.environ.get("LLAMA_INSIGHT_EXTRA_CMAKE_FLAGS")
    merged = " ".join(filter(None, [flag_string, env_flags])).strip()
    return merged.split() if merged else []


def _read_setting(config_settings: dict[str, Any] | None, option: str) -> str | None:
    if not config_settings:
        return None
    key = f"llama-insight.{option}"
    value = config_settings.get(key)
    if isinstance(value, list):
        return value[-1]
    return value


def _read_bool_setting(
    config_settings: dict[str, Any] | None,
    option: str,
    *,
    env: tuple[str, ...] = (),
) -> bool:
    raw = _read_setting(config_settings, option)
    if raw is None:
        for env_var in env:
            env_value = os.environ.get(env_var)
            if env_value is not None:
                raw = env_value
                break
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _determine_jobs() -> int:
    env_value = os.environ.get("LLAMA_INSIGHT_JOBS") or os.environ.get("JOBS")
    if env_value:
        return max(1, int(env_value))
    cpu_count = os.cpu_count() or 2
    return max(1, cpu_count - 1)


def _run(
    cmd: list[str],
    cwd: Path | None = None,
    *,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    printable = " ".join(str(part) for part in cmd)
    if not quiet:
        _log(printable)
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    return subprocess.run(cmd, cwd=cwd, check=True, stdout=stdout, stderr=stderr)


def _write_build_metadata(
    backend: str, extra_flags: list[str], libs: list[str]
) -> None:
    PACKAGED_LIB_DIR.mkdir(parents=True, exist_ok=True)
    payload = _fingerprint(backend, extra_flags, libs)
    BUILD_METADATA_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_build_metadata() -> dict[str, Any] | None:
    if not BUILD_METADATA_FILE.exists():
        return None
    try:
        return json.loads(BUILD_METADATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _fingerprint(
    backend: str, extra_flags: list[str], libs: list[str]
) -> dict[str, Any]:
    return {
        "backend": backend,
        "extra_flags": extra_flags,
        "platform": sys.platform,
        "machine": platform.machine(),
        "libs": libs,
    }


def _cache_root() -> Path:
    override = os.environ.get("LLAMA_INSIGHT_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "llama_insight"
    return Path.home() / ".cache" / "llama_insight"


def _artifact_cache_dir() -> Path:
    cache = _cache_root() / "artifacts"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _artifact_cache_key(backend: str, extra_flags: list[str]) -> str:
    payload = json.dumps(
        {
            "backend": backend,
            "extra_flags": extra_flags,
            "platform": sys.platform,
            "machine": platform.machine(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_slot(backend: str, extra_flags: list[str]) -> Path:
    return _artifact_cache_dir() / _artifact_cache_key(backend, extra_flags)


def _maybe_restore_cached_artifacts(backend: str, extra_flags: list[str]) -> bool:
    try:
        slot = _cache_slot(backend, extra_flags)
        libs_dir = slot / "libs"
        metadata_src = slot / BUILD_METADATA_FILE.name
        if not libs_dir.exists() or not metadata_src.exists():
            return False
        lib_candidates = sorted(p for p in libs_dir.iterdir() if p.is_file())
        if not lib_candidates:
            return False
        if PACKAGED_LIB_DIR.exists():
            shutil.rmtree(PACKAGED_LIB_DIR)
        PACKAGED_LIB_DIR.mkdir(parents=True, exist_ok=True)
        for candidate in lib_candidates:
            shutil.copy2(candidate, PACKAGED_LIB_DIR / candidate.name)
        shutil.copy2(metadata_src, BUILD_METADATA_FILE)
        _log("Restored cached native artifacts; skipping rebuild if unchanged.")
        return True
    except Exception as err:
        _log(f"Native artifact cache restore failed: {err}")
        return False


def _store_cached_artifacts(
    backend: str, extra_flags: list[str], libs: list[str]
) -> None:
    try:
        slot = _cache_slot(backend, extra_flags)
        if slot.exists():
            shutil.rmtree(slot)
        libs_dir = slot / "libs"
        libs_dir.mkdir(parents=True, exist_ok=True)
        for lib_name in libs:
            source = PACKAGED_LIB_DIR / lib_name
            if source.exists():
                shutil.copy2(source, libs_dir / lib_name)
        if BUILD_METADATA_FILE.exists():
            shutil.copy2(BUILD_METADATA_FILE, slot / BUILD_METADATA_FILE.name)
        _log(f"Cached native artifacts for backend '{backend}'.")
    except Exception as err:
        _log(f"Unable to write native artifact cache: {err}")


def _resolve_existing_artifacts(
    raise_errors: bool = False,
) -> tuple[dict[str, Any], list[str]] | None:
    def _fail(message: str) -> None | tuple[dict[str, Any], list[str]]:
        if raise_errors:
            raise FileNotFoundError(message)
        return None

    if not PACKAGED_LIB_DIR.exists():
        return _fail(f"Packaged library directory missing: {PACKAGED_LIB_DIR}")
    metadata = _read_build_metadata()
    if not metadata:
        return _fail("Build metadata missing; native libraries were not generated.")
    libs = metadata.get("libs") or []
    if not libs:
        return _fail("Build metadata missing library listing; rebuild required.")
    missing = [lib for lib in libs if not (PACKAGED_LIB_DIR / lib).exists()]
    if missing:
        return _fail("Missing required libraries after build: " + ", ".join(missing))
    return metadata, libs


def _ensure_required_libs_present() -> None:
    _resolve_existing_artifacts(raise_errors=True)


def _log(message: str) -> None:
    # Print concise progress messages; flush so quiet frontends show activity.
    print(f"[llama_insight.build] {message}")
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _cmd_ok(cmd: list[str]) -> bool:
    try:
        _run(cmd, quiet=True)
        return True
    except Exception:
        return False


def _log_backend_summary(backend: str, source: str) -> None:
    selectors = "|".join(sorted(BACKEND_FLAGS))
    _log(
        f"Backend selected: {backend} (source: {source}). "
        f"Override with LLAMA_INSIGHT_BACKEND=<{selectors}>."
    )
