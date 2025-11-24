from __future__ import annotations

import shutil
from pathlib import Path

HEADER_SUFFIXES = {
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
    ".inc",
    ".inl",
    ".metal",
    ".cuh",
}


def _copy_headers_only(src: Path, dst: Path) -> None:
    """Replicate `src` under `dst` but only copy allowed header files."""
    copied = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in HEADER_SUFFIXES:
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    print(f"[stage-headers] Copied {copied} header files from {src} -> {dst}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    llama_cpp_dir = repo_root / "llama.cpp"
    package_dir = repo_root / "wrapper_src" / "llama_insight"
    headers_root = package_dir / "_headers"
    llama_headers_root = headers_root / "llama.cpp"

    include_dirs = [
        llama_cpp_dir / "include",
        llama_cpp_dir / "ggml" / "include",
        llama_cpp_dir / "common",
        llama_cpp_dir / "tools" / "mtmd",
    ]

    if not llama_cpp_dir.exists():
        raise SystemExit(f"llama.cpp checkout not found at {llama_cpp_dir}")

    # Reset headers tree
    if headers_root.exists():
        shutil.rmtree(headers_root)
    llama_headers_root.mkdir(parents=True, exist_ok=True)

    for src in include_dirs:
        if not src.exists():
            raise SystemExit(f"Missing include directory: {src}")
        rel = src.relative_to(llama_cpp_dir)
        dst = llama_headers_root / rel
        dst.mkdir(parents=True, exist_ok=True)
        print(f"[stage-headers] Copying headers {src} -> {dst}")
        _copy_headers_only(src, dst)

    # Also expose the helper header next to the llama.cpp headers.
    gen_helper_header = repo_root / "wrapper_src" / "gen-helper" / "generation_helper.h"
    if gen_helper_header.exists():
        dst = headers_root / "generation_helper.h"
        print(f"[stage-headers] Copy {gen_helper_header} -> {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gen_helper_header, dst)
    else:
        print(f"[stage-headers] Warning: helper header not found: {gen_helper_header}")

    print(f"[stage-headers] Header staging complete under {headers_root}")


if __name__ == "__main__":
    main()
