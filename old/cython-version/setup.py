from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Base directory for the project
# Handle both cases: running from cython-src/ and from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "cython-src":
    # Running from cython-src directory
    BASE_DIR = os.path.dirname(script_dir)
else:
    # Running from project root (e.g., via uv run cython-src/setup.py)
    BASE_DIR = os.getcwd()
LLAMA_CPP_DIR = f"{BASE_DIR}/llama.cpp"

# Include directories
include_dirs = [
    f"{LLAMA_CPP_DIR}/include",
    f"{LLAMA_CPP_DIR}/ggml/include",
    f"{LLAMA_CPP_DIR}/common",
    f"{LLAMA_CPP_DIR}/tools/mtmd",
    f"{BASE_DIR}/gen-helper",
    f"{LLAMA_CPP_DIR}/ggml/src",  # For ggml.h
]

# Library directories
library_dirs = [
    f"{LLAMA_CPP_DIR}/build/bin",
    f"{BASE_DIR}/gen-helper/build",
]

# Libraries to link against (Linux .so files)
libraries = ["ggml-base", "ggml-cpu", "ggml", "llama", "mtmd", "generation_helper"]

# Define the extension
extensions = [
    Extension(
        "llama_mtmd_cython_wrapper",
        ["llama_mtmd_cython_wrapper.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language="c++",
        extra_compile_args=["-std=c++17"],
        extra_link_args=["-Wl,-rpath," + lib_dir for lib_dir in library_dirs],
    )
]

# Setup
setup(
    name="llama_mtmd_cython_wrapper",
    ext_modules=cythonize(extensions),
)
