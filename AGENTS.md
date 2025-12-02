Notes for agents
----------------

- Always use `uv` for Python environment and dependency workflows in this repo.
- Repo layout quick reference:
  - `old/` - Contains old and out of date code. Never read, edit or update this code unless explicitly directed.
  - `usecases/` – runnable demos such as `infer_from_encoded.py` (media embedding cache demo).
  - `wrapper_src/llama_insight/` – Python wrapper/loading logic, staged headers for `cppyy`, session helpers.
  - `wrapper_src/gen-helper/` – C++ helper (`generation_helper.cpp/.h`) providing generation loop and media embedding save/load; built as a shared library.
  - `llama.cpp/` – upstream submodule; build shared libs into `llama.cpp/build/bin/`.
  - `build-tools/` – packaging/build helpers used by `uv build` to patch/build bundle artifacts.
    - `stage_headers.py` ensures headers are packaged correctly into the final python wheel
    - `build_backend.py` ensures llama.cpp builds, and copies the required files into the python wheel
  - `embeddings/` – on-disk cache for media embeddings created by demos.
  - `test-images/` – sample assets for demos.

- The goal of this project is to provide a simple installable python wheel that can be built to support a wide variety of backends and allow specific deep integrations into llama.cpp as a python library. Llama.cpp needs to be build, then shared library files need to be embedded into the dist python wheel