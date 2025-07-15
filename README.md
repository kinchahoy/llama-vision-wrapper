# Llama Vision Wrapper Setup Guide

This guide provides quick steps to set up the `llama.cpp` submodule within this project and apply a necessary patch.

## 1. Clone the Repository (if you haven't already)

If you haven't cloned this parent repository, do so first:

```bash
git clone <URL_OF_THIS_REPO>
cd <NAME_OF_THIS_REPO_DIRECTORY>

## 2. Initialize and Download the llama.cpp Submodule

The llama.cpp component is included as a Git submodule. You need to initialize and update it to download its contents.

git submodule update --init --recursive

## 3. Apply the `llama.cpp` Patch

A patch is provided to modify the `llama.cpp` build configuration. This patch is necessary to build `libcommon` as a shared library, which is required by the Python wrappers in this project.

Ensure you are in the root directory of this repository and run:

```bash
patch -p1 < patch_llama_common_for_dynamic.patch
```

## 4. Build `llama.cpp` Shared Libraries

Navigate into the `llama.cpp` submodule and build it with the `BUILD_SHARED_LIBS` flag enabled.

```bash
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build
cd ..
```

## 5. Build the Helper Library

This project uses a small C++ helper (`generation_helper`) to simplify token generation. Build it as a shared library from the project root:

```bash
mkdir -p build
cd build
cmake ..
make
cd ..
```
The required libraries will now be available in `llama.cpp/build/bin/` and `build/`.

## 6. Set up Python Environment

This project uses `pyproject.toml` to declare its Python dependencies.

1.  **Create a virtual environment and activate it.** If you are using `uv`:
    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  **Install dependencies.** This will install `cppyy` and `Cython` into your virtual environment.
    ```bash
    uv pip install -e .
    ```

## 7. Run the Examples

Before running, ensure you have downloaded the required models (e.g., a Gemma-3 GGUF model and the corresponding MMPROJ file) and updated the hardcoded paths at the top of the scripts:
- `cython/cython-mtmd.py`
- `cpppy/ccpyy-mtmd.py`

There are two example implementations available:

### Option A: Run with `cppyy` (Dynamic Bindings)

This approach is simpler for development as it loads C++ code at runtime.
```bash
python cpppy/ccpyy-mtmd.py
```

### Option B: Run with Cython (Compiled Extension)

This approach offers the best performance by compiling the wrapper into a native Python extension.

1.  **Build the Cython extension:**
    ```bash
    cd cython/
    python setup.py build_ext --inplace
    cd ..
    ```

2.  **Run the script:**
    ```bash
    python cython/cython-mtmd.py
    ```

