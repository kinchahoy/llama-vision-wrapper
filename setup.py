#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typer[all]",
#     "sh",
# ]
# ///

"""
Llama-Vision-Wrapper interactive setup
--------------------------------------
• Enum prompt / CLI for backend (cuda | metal | vulkan | none | custom)
• sh one-liners with live output
• Auto-detected -j, numbered banners, --dry-run
• Patch step is batch-mode and now treats exit-code 1 as “already applied”
"""

from enum import Enum
from pathlib import Path
import os, typer, sh

# ─────────── paths ───────────
HERE        = Path(__file__).resolve().parent
PATCH_FILE  = HERE / "patch_llama_common_for_dynamic.patch"
LLAMA_SRC   = HERE / "llama.cpp"
LLAMA_BUILD = LLAMA_SRC / "build"
GEN_SRC     = HERE / "gen-helper"
GEN_BUILD   = GEN_SRC / "build"

# ────────── backend ──────────
class Backend(str, Enum):
    cuda   = "cuda"
    metal  = "metal"
    vulkan = "vulkan"
    hip    = "hip"
    kleidiai = "kleidiai"
    none   = "none"
    custom = "custom"

BACKEND_FLAGS = {
    Backend.cuda.value   : ["-DGGML_CUDA=ON"],
    Backend.metal.value  : ["-DGGML_METAL=ON"],
    Backend.vulkan.value : ["-DGGML_VULKAN=ON"],
    Backend.hip.value : ["-DGGML_HIP=ON"],
    Backend.kleidiai.value : ["-DGGML_CPU_KLEIDIAI=ON"],
    Backend.none.value   : [],
}

# ───────── utilities ─────────
step_no = 0
def banner(msg: str) -> None:
    global step_no
    step_no += 1
    typer.secho(f"\n=== Step {step_no}: {msg} ===",
                fg=typer.colors.BRIGHT_BLUE, bold=True)

def run(cmd: list[str], *, dry=False) -> None:
    """Echo & execute cmd with sh; abort on non-zero exit."""
    typer.echo(f"👉 {' '.join(map(str, cmd))}")
    if dry:
        return
    try:
        sh.Command(cmd[0])(*cmd[1:], _fg=True)
    except sh.ErrorReturnCode as e:
        typer.secho(f"❌ exited {e.exit_code}", fg=typer.colors.RED)
        raise typer.Exit(e.exit_code)

# ─────────── CLI app ─────────
app = typer.Typer(add_help_option=False)

@app.command()
def setup(
    backend: Backend = typer.Option(
        Backend.cuda, "--backend", "-b",
        prompt=True, help="cuda | metal | vulkan | none | custom",
    ),
    extra_flags: str = typer.Option(
        "", "--extra-flags",
        help="Extra -D flags (use only with --backend custom)",
    ),
    jobs: int | None = typer.Option(
        None, "--jobs", "-j",
        help="Parallel build jobs (default: $JOBS or CPUs-1)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print commands without executing",
    ),
):
    os.chdir(HERE)
    typer.secho("=== Llama-Vision-Wrapper Setup ===",
                fg=typer.colors.GREEN, bold=True)

    if backend is Backend.custom and not extra_flags:
        extra_flags = typer.prompt("Enter custom CMake -D flags", default="")

    jobs = jobs or int(os.getenv("JOBS", (os.cpu_count() or 2) - 1))

    # ── 1. patch ──────────────────────────────────────────────────────────
    banner("Apply llama.cpp patch (batch mode)")
    if not PATCH_FILE.exists():
        typer.secho("Patch file missing!", fg=typer.colors.RED); raise typer.Exit(1)

    # Accept exit-code 0 (applied) or 1 (already applied / reversed)
    typer.echo(f"👉 patch -p1 -N --silent --batch -r - -i {PATCH_FILE}")
    if not dry_run:
        sh.patch("-p1", "-N", "--silent", "--batch", "-r", "-",
                 "-i", str(PATCH_FILE), _ok_code=[0, 1], _fg=True)

    # ── 2. configure ──────────────────────────────────────────────────────
    banner("Configure llama.cpp")
    LLAMA_BUILD.mkdir(parents=True, exist_ok=True)
    cmake_flags = (BACKEND_FLAGS[backend.value]
                   if backend is not Backend.custom
                   else extra_flags.split())
    run(["cmake", str(LLAMA_SRC), "-B", str(LLAMA_BUILD),
         "-DBUILD_SHARED_LIBS=ON", *cmake_flags], dry=dry_run)

    # ── 3. build llama.cpp ────────────────────────────────────────────────
    banner(f"Build llama.cpp (-j{jobs})")
    run(["cmake", "--build", str(LLAMA_BUILD), "-j", str(jobs)], dry=dry_run)

    # ── 4. build gen-helper ───────────────────────────────────────────────
    banner("Build gen-helper")
    GEN_BUILD.mkdir(parents=True, exist_ok=True)
    run(["cmake", str(GEN_SRC), "-B", str(GEN_BUILD)], dry=dry_run)
    run(["cmake", "--build", str(GEN_BUILD), "-j", str(jobs)], dry=dry_run)

    # ── 5. sync venv ──────────────────────────────────────────────────────
    banner("Sync virtual-env via uv")
    run(["uv", "sync"], dry=dry_run)

    # ── 6. optional examples ─────────────────────────────────────────────
    banner("Run example (optional)")
    ex = typer.prompt("Which example? (cppyy / cython / skip)", default="skip").lower()
    if ex == "cppyy":
        run(["uv", "run", "cppyy-src/cppyy-mtmd.py"], dry=dry_run)
    elif ex == "cython":
        run(["uv", "run", "cython-src/setup.py", "build_ext", "--inplace"], dry=dry_run)
        run(["uv", "run", "cython/cython-mtmd.py"], dry=dry_run)

    typer.secho("\n✅  Setup finished successfully!",
                fg=typer.colors.GREEN, bold=True)

# ─────────────────────────────
if __name__ == "__main__":
    app()
