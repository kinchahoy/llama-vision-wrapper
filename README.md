# llama-vision-wrapper

High-level llama.cpp multimodal bindings powered by cppyy. Builds a device-specific wheel that bundles llama.cpp shared libraries and the generation helper.

## Quick Start

```bash
uv build
uv sync
uv pip install --reinstall --no-deps dist/llama_insight-*.whl
uv run usecases/generate_simple.py
```

This repo intentionally uses a wheel-only run path for local testing. Do not run `uv add dist/*.whl` in this repository: `uv` treats that as a self-dependency because the wheel name matches the current project.

### ARM64 Linux (RK3588 / OrangePi 5 Pro)

`cppyy-cling` has no pre-built aarch64 wheel and must be compiled from source (1–3 hours on first run). If the uv cache already has a built wheel, plain `uv sync` works. On a fully cold machine (no cache), run the bootstrap first to be safe:

```bash
./bootstrap-arm.sh  # pre-builds cppyy-cling; skip if uv cache is warm
uv sync
```

After the first successful sync, `uv sync` and `uv run` work normally.

---

## Building the wheel

```bash
uv build
```

This will:
- Ensure llama.cpp sources exist (submodule or manual checkout)
- Apply `patch_llama_common_for_dynamic.patch`
- Stage headers into `src/llama_insight/_headers/`
- Build llama.cpp + gen-helper shared libraries
- Package everything into `src/llama_insight/libs/`

### Backend selection

Priority order: config setting → env var → auto-detect → default (`cpu`)

```bash
uv build --config-setting llama-insight.backend=cpu
uv build --config-setting llama-insight.backend=cuda
uv build --config-setting llama-insight.backend=metal    # macOS
uv build --config-setting llama-insight.backend=vulkan
uv build --config-setting llama-insight.backend=kleidiai # ARM
```

### Extra CMake flags

```bash
uv build --config-setting 'llama-insight.extra-flags=-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS'
```

### Other config settings

| Setting | Effect |
|---------|--------|
| `llama-insight.skip-native-build=true` | Reuse already-built libs, skip compilation |
| `llama-insight.dry-run=true` | Print build steps without compiling |

### Environment variable equivalents

`LLAMA_INSIGHT_BACKEND`, `LLAMA_INSIGHT_EXTRA_CMAKE_FLAGS`, `LLAMA_INSIGHT_SKIP_NATIVE_BUILD`, `LLAMA_INSIGHT_DRY_RUN`, `LLAMA_INSIGHT_JOBS` / `JOBS`
