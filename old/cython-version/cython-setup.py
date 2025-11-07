from __future__ import annotations

from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

# Resolve important paths relative to repo root (independent of cwd).
ROOT_DIR = Path(__file__).resolve().parents[2]
LLAMA_CPP_DIR = ROOT_DIR / "llama.cpp"
GEN_HELPER_DIR = ROOT_DIR / "wrapper_src" / "gen-helper"
GEN_HELPER_BUILD_DIR = GEN_HELPER_DIR / "build"

# Include directories
include_dirs = [
    LLAMA_CPP_DIR / "include",
    LLAMA_CPP_DIR / "ggml" / "include",
    LLAMA_CPP_DIR / "common",
    LLAMA_CPP_DIR / "tools" / "mtmd",
    GEN_HELPER_DIR,
    LLAMA_CPP_DIR / "ggml" / "src",  # For ggml.h
]

# Library directories
library_dirs = [
    LLAMA_CPP_DIR / "build" / "bin",
    GEN_HELPER_BUILD_DIR,
]

# Libraries to link against (Linux .so files)
libraries = ["ggml-base", "ggml-cpu", "ggml", "llama", "mtmd", "generation_helper"]

# Define the extension
extensions = [
    Extension(
        "llama_mtmd_cython_wrapper",
        [ROOT_DIR / "old" / "cython-version" / "llama_mtmd_cython_wrapper.pyx"],
        include_dirs=[str(path) for path in include_dirs],
        library_dirs=[str(path) for path in library_dirs],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-std=c++17"],
        extra_link_args=["-Wl,-rpath," + str(lib_dir) for lib_dir in library_dirs],
    )
]

# Setup
setup(
    name="llama_mtmd_cython_wrapper",
    ext_modules=cythonize(extensions),
)
