# Llama Vision Wrapper

Lightweight bridge around `llama.cpp` that supports multimodal prompts (text + images), caches media embeddings, and exposes Python helpers via `cppyy`.

## Quick start

1. Clone and pull the submodule:
   ```bash
   git clone <URL_OF_THIS_REPO>
   cd <NAME_OF_THIS_REPO_DIRECTORY>
   git submodule update --init --recursive
   ```
2. Patch `llama.cpp` once after updating the submodule:
   ```bash
   patch -p1 < patch_llama_common_for_dynamic.patch
   ```
3. Build native libs (shared `llama.cpp` and the helper library):
   ```bash
   cd llama.cpp
   cmake -B build -DBUILD_SHARED_LIBS=ON
   cmake --build build
   cd ..
   cmake -B build
   cmake --build build
   ```
4. Set up Python with `uv`:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```
5. Run a demo (downloads the default model `ggml-org/SmolVLM2-2.2B-Instruct-GGUF` on first use):
   ```bash
   python usecases/infer_from_encoded.py --image test-images/debug.jpg
   ```

## Repo layout

- `usecases/` – runnable demos (e.g., `infer_from_encoded.py` caches media embeddings across runs).
- `wrapper_src/llama_insight/` – Python wrapper that loads the shared libs, stages headers for `cppyy`, and provides helpers like `start_session`.
- `wrapper_src/gen-helper/` – C++ helper (`generation_helper.cpp/.h`) that adds media embedding I/O and a thin generation loop; built as a shared lib consumed by `cppyy`.
- `llama.cpp/` – upstream submodule; shared libs produced under `llama.cpp/build/bin/`.
- `build-tools/` – packaging/build helpers used by `uv build` to patch `llama.cpp`, stage headers, and bundle artifacts.
- `embeddings/` – on-disk cache for media embeddings created by the usecase scripts.
- `test-images/` – sample assets used by demos.

## Packaging notes

- `uv build` or `uv pip install .` will trigger the custom backend in `build-tools/` to patch and compile `llama.cpp` plus `wrapper_src/gen-helper`, then bundle the shared libs for loading via `cppyy`.
- Set `LLAMA_INSIGHT_BACKEND` to force a backend (cpu/metal/cuda/vulkan/hip/kleidiai/custom) or `LLAMA_INSIGHT_EXTRA_CMAKE_FLAGS` to append CMake options.
- If you want to reuse an external build tree instead of packaged libs, point `LLAMA_INSIGHT_ROOT` to that directory.
