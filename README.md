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

2.  **Install dependencies.** This will install `cppyy`, `Cython`, and `huggingface-hub` into your virtual environment.
    ```bash
    uv pip install -e .
    ```

## 7. Run the Examples

The scripts will automatically download the necessary models from Hugging Face Hub. The default model is `ggml-org/SmolVLM2-2.2B-Instruct-GGUF`.

You can customize the model, image, and prompt using command-line arguments. For example:
```bash
python cpppy/ccpyy-mtmd.py --image my_image.png --prompt "USER: What is in this picture?\n<__image__>\nASSISTANT:"
```

To use a different model from Hugging Face Hub, use the `-hf` (or `--repo-id`), `-m` (`--model`), and `--mmproj` arguments:
```bash
python cpppy/ccpyy-mtmd.py \
  -hf "another-org/another-model-GGUF" \
  -m "model-file.gguf" \
  --mmproj "mmproj-file.gguf"
```

There are two example implementations available:

### Option A: Run with `cppyy` (Dynamic Bindings)

This approach is simpler for development as it loads C++ code at runtime.
```bash
# Run with default model and test image
python cpppy/ccpyy-mtmd.py

# Run with a specific image
python cpppy/ccpyy-mtmd.py --image path/to/your/image.jpg
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
    # Run with default model and test image
    python cython/cython-mtmd.py

    # Run with a specific image and prompt
    python cython/cython-mtmd.py --image path/to/your/image.jpg --prompt "USER: What's this?\n<__image__>\nASSISTANT:"
    ```

## 8. Build Installable Artifacts with `uv`

Once the libraries above are built, you can produce redistributable artifacts and install them elsewhere without cloning the repo.

1.  Build the wheel and source distribution:
    ```bash
    uv build
    ```
    The artifacts land in `dist/llama_insight-<version>.whl` and `.tar.gz`.
ls
2.  Install the wheel into another environment or project:
    ```bash
    uv pip install dist/llama_insight-0.1.0-py3-none-any.whl
    # or directly from the repo root:
    uv pip install .
    ```

3.  After installing outside this repository, point the package at a directory that contains your compiled `llama.cpp` and helper libraries:
    ```bash
    export LLAMA_INSIGHT_ROOT=/path/to/llama-vision-wrapper
    ```
    This ensures `llama_insight.core` can find `llama.cpp`, the helper build artifacts, and the shared libraries expected by `cppyy`.

### Native build automation (Option 3)

`uv build` now uses a custom backend (`build-tools/build_backend.py`) that keeps `uv_build` in charge of packaging while delegating the native compilation work to scikit-build-core’s CMake discovery helpers. During every wheel/editable build the backend:

- Applies `patch_llama_common_for_dynamic.patch` to the vendored `llama.cpp`.
- Configures and compiles `llama.cpp` plus `wrapper_src/gen-helper` with CMake.
- Copies the produced shared objects into `wrapper_src/llama_insight/libs/`.
- Ensures the `llama.cpp` sources exist: if the submodule hasn’t been initialized yet it runs `git submodule update --init --recursive llama.cpp`, or clones from `LLAMA_INSIGHT_LLAMA_CPP_URL` when provided.

Header files for `cppyy` live under `wrapper_src/llama_insight/_headers/` and the backend automatically refreshes them on each build by invoking `build-tools/stage_headers.py`, so you don’t need a separate staging step.

Backend selection defaults to an auto-detect pass (Metal → CUDA → Vulkan → HIP → KleidiAI → CPU). Set `LLAMA_INSIGHT_BACKEND` to force a choice.

Native builds cache their compiled `.so`/`.dylib` files under `~/.cache/llama_insight/artifacts` (override via `LLAMA_INSIGHT_CACHE_DIR`), so repeated `uv build` runs with the same backend/flags can skip recompilation.

Controls (env vars):

| Setting | Purpose |
| --- | --- |
| `LLAMA_INSIGHT_BACKEND` | Override the backend (`cpu`, `cuda`, `metal`, `vulkan`, `hip`, `kleidiai`, `custom`). If unset, the installer auto-detects Metal → CUDA → Vulkan → HIP → KleidiAI → CPU. |
| `LLAMA_INSIGHT_EXTRA_CMAKE_FLAGS` | Append custom `-D` flags for atypical builds. |
| `LLAMA_INSIGHT_JOBS` / `JOBS` | Override the CMake parallelism. |
| `LLAMA_INSIGHT_SKIP_NATIVE_BUILD=1` | Skip the native compilation (useful for docs/tests); leaves any existing libs untouched. |
| `LLAMA_INSIGHT_LLAMA_CPP_URL` | Provide a git URL to clone `llama.cpp` when the submodule isn’t available. |

On install (`uv pip install .` or `pip install dist/llama_insight-*.whl`), the backend recompiles the native pieces on the target machine and embeds them inside the wheel, so `llama_insight.core` automatically loads the packaged `.so/.dylib` files without requiring a pre-built checkout. Set `LLAMA_INSIGHT_ROOT` only when you intentionally want to override the packaged artifacts with your own build tree.

Notes on logging and long installs
- The installer prints concise progress messages like `[llama_insight.build] Building llama.cpp ...` and flushes output, so even with uv’s minimal progress UI you’ll see periodic updates.
- You can run with `UV_VERBOSE=1` (or `-v` flags on some commands) for more verbosity from underlying tools; our backend always emits brief step logs regardless.

Backend selection at install time
- During `uv pip install ./llama_insight-<ver>.tar.gz`, the installer logs the backend in use, e.g. `Backend selected: metal (source: env)` when `LLAMA_INSIGHT_BACKEND=metal` is set.
- To force a backend or tweak settings inline with uv:
  - Force backend:
    - `LLAMA_INSIGHT_BACKEND=metal uv pip install ./llama_insight-0.1.0.tar.gz`
  - Extra CMake flags:
    - `LLAMA_INSIGHT_EXTRA_CMAKE_FLAGS='-DGGML_VULKAN=ON' uv pip install ./...`
  - Parallel jobs:
    - `LLAMA_INSIGHT_JOBS=8 uv pip install ./llama_insight-0.1.0.tar.gz`
  - Alternate llama.cpp fork:
    - `LLAMA_INSIGHT_LLAMA_CPP_URL=https://github.com/ggerganov/llama.cpp.git uv pip install ./...`
