Notes for agents
----------------

- Always use `uv` for Python environment and dependency workflows in this repo.
- Use a wheel-only local run loop:
  - `uv build`
  - `uv sync`
  - `uv pip install --reinstall --no-deps dist/llama_insight-*.whl`
  - `uv run usecases/generate_simple.py` (or another script in `usecases/`)
- Do not run `uv add dist/*.whl` in this repo. `uv` rejects it as a self-dependency.
- Repo layout quick reference:
  - `old/` - Contains old and out of date code. Never read, edit or update this code unless explicitly directed.
  - `usecases/` – runnable demos such as `infer_from_encoded.py` (media embedding cache demo).
  - `src/llama_insight/` – Python wrapper/loading logic, staged headers for `cppyy`, session helpers.
  - `src/gen-helper/` – C++ helper (`generation_helper.cpp/.h`) providing generation loop and media embedding save/load; built as a shared library.
  - `llama.cpp/` – upstream submodule; build shared libs into `llama.cpp/build/bin/`.
  - `build-tools/` – packaging/build helpers used by `uv build` to patch/build bundle artifacts.
    - `stage_headers.py` ensures headers are packaged correctly into the final python wheel
    - `build_backend.py` ensures llama.cpp builds, and copies the required files into the python wheel
  - `dist/` – wheel/sdist artifacts produced by `uv build`.
  - `embeddings/` – optional on-disk cache for media embeddings created by demos.
  - `test-images/` – sample assets for demos.
  - `bootstrap-arm.sh` – helper for first-time ARM64 bootstrap of `cppyy-cling`.

- The goal of this project is to provide a simple installable Python wheel that supports a wide variety of backends and deep integrations into llama.cpp as a Python library. `llama.cpp` must be built, then shared libraries are embedded into the wheel.
