from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Base directory for the project
BASE_DIR = "/Users/raistlin/code/llama-mtmd-py"
LLAMA_CPP_DIR = f"{BASE_DIR}/llama.cpp"

# Include directories
include_dirs = [
    f"{LLAMA_CPP_DIR}/include",
    f"{LLAMA_CPP_DIR}/ggml/include",
    f"{LLAMA_CPP_DIR}/common",
    f"{LLAMA_CPP_DIR}/examples/llava",
]

# Library directories
library_dirs = [
    f"{LLAMA_CPP_DIR}/build/bin",
    f"{BASE_DIR}/build",
]

# Libraries to link against
libraries = [
    "ggml-base", "ggml-blas", "ggml-cpu", "ggml-metal", "ggml",
    "llama", "llava_shared", "mtmd_shared", "generation_helper"
]

# Define the extension
extensions = [
    Extension(
        "llama_mtmd_wrapper",
        ["llama_mtmd_wrapper.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language="c++",
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-Wl,-rpath," + lib_dir for lib_dir in library_dirs],
    )
]

# Setup
setup(
    name="llama_mtmd_wrapper",
    ext_modules=cythonize(extensions),
)
