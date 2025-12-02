Notes for agents
----------------

- Always use `uv` for Python environment and dependency workflows in this repo.
- Repo layout quick reference:
  - `usecases/` – runnable demos such as `infer_from_encoded.py` (media embedding cache demo).
  - `wrapper_src/llama_insight/` – Python wrapper/loading logic, staged headers for `cppyy`, session helpers.
  - `wrapper_src/gen-helper/` – C++ helper (`generation_helper.cpp/.h`) providing generation loop and media embedding save/load; built as a shared library.
  - `llama.cpp/` – upstream submodule; build shared libs into `llama.cpp/build/bin/`.
  - `build-tools/` – packaging/build helpers used by `uv build` to patch/build bundle artifacts.
  - `embeddings/` – on-disk cache for media embeddings created by demos.
  - `test-images/` – sample assets for demos.
