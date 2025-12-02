"""cppyy-based wrapper classes that bridge to llama.cpp."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cppyy

PACKAGE_DIR = Path(__file__).resolve().parent
HEADERS_ROOT = PACKAGE_DIR / "_headers"


def _resolve_project_root() -> Path:
    env_root = os.environ.get("LLAMA_INSIGHT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "llama.cpp").exists():
            return candidate

    embedded = PACKAGE_DIR / ".deps"
    if (embedded / "llama.cpp").exists():
        return embedded

    return current.parents[1]


BASE_DIR = _resolve_project_root()
LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"

PACKAGED_LIB_DIR = PACKAGE_DIR / "libs"

if PACKAGED_LIB_DIR.exists():
    LLAMA_CPP_LIBS_DIR = PACKAGED_LIB_DIR
    HELPER_LIB_DIR = PACKAGED_LIB_DIR
else:
    LLAMA_CPP_LIBS_DIR = LLAMA_CPP_DIR / "build" / "bin"
    HELPER_LIB_DIR = BASE_DIR / "wrapper_src" / "gen-helper" / "build"

if sys.platform == "darwin":
    LIB_EXT = "dylib"
elif sys.platform.startswith("win"):
    LIB_EXT = "dll"
else:
    LIB_EXT = "so"

LIB_PREFIX = "" if sys.platform.startswith("win") else "lib"


def _shared_lib_name(base: str) -> str:
    return f"{LIB_PREFIX}{base}.{LIB_EXT}"


_GENERATION_HELPER_LIB = _shared_lib_name("generation_helper")

if HEADERS_ROOT.exists():
    INCLUDE_DIRS = [
        HEADERS_ROOT / "llama.cpp" / "include",
        HEADERS_ROOT / "llama.cpp" / "ggml" / "include",
        HEADERS_ROOT / "llama.cpp" / "common",
        HEADERS_ROOT / "llama.cpp" / "tools" / "mtmd",
        HEADERS_ROOT,
    ]
    HEADER_PATHS = {
        "llama.h": HEADERS_ROOT / "llama.cpp" / "include" / "llama.h",
        "common.h": HEADERS_ROOT / "llama.cpp" / "common" / "common.h",
        "sampling.h": HEADERS_ROOT / "llama.cpp" / "common" / "sampling.h",
        "mtmd.h": HEADERS_ROOT / "llama.cpp" / "tools" / "mtmd" / "mtmd.h",
        "mtmd-helper.h": HEADERS_ROOT
        / "llama.cpp"
        / "tools"
        / "mtmd"
        / "mtmd-helper.h",
        "generation_helper.h": HEADERS_ROOT / "generation_helper.h",
    }
else:
    INCLUDE_DIRS = [
        LLAMA_CPP_DIR / "include",
        LLAMA_CPP_DIR / "ggml" / "include",
        LLAMA_CPP_DIR / "common",
        LLAMA_CPP_DIR / "tools" / "mtmd",
        BASE_DIR / "wrapper_src" / "gen-helper",
    ]
    HEADER_PATHS = {
        "llama.h": LLAMA_CPP_DIR / "include" / "llama.h",
        "common.h": LLAMA_CPP_DIR / "common" / "common.h",
        "sampling.h": LLAMA_CPP_DIR / "common" / "sampling.h",
        "mtmd.h": LLAMA_CPP_DIR / "tools" / "mtmd" / "mtmd.h",
        "mtmd-helper.h": LLAMA_CPP_DIR / "tools" / "mtmd" / "mtmd-helper.h",
        "generation_helper.h": BASE_DIR
        / "wrapper_src"
        / "gen-helper"
        / "generation_helper.h",
    }

_initialized = False
gbl = None


def _collect_required_libs() -> list[str]:
    libs = [
        _shared_lib_name("ggml-base"),
        _shared_lib_name("ggml-cpu"),
    ]
    libs.extend(_discover_backend_libs(LLAMA_CPP_LIBS_DIR))
    libs.extend(
        [
            _shared_lib_name("ggml"),
            _shared_lib_name("llama"),
            _shared_lib_name("mtmd"),
            _shared_lib_name("common"),
            _GENERATION_HELPER_LIB,
        ]
    )
    return libs


def _discover_backend_libs(lib_dir: Path) -> list[str]:
    if not lib_dir.exists():
        return []
    skipped = {
        _shared_lib_name("ggml-base"),
        _shared_lib_name("ggml-cpu"),
        _shared_lib_name("ggml"),
    }
    pattern = f"{LIB_PREFIX}ggml-*.{LIB_EXT}" if LIB_PREFIX else f"ggml-*.{LIB_EXT}"
    libs = []
    for candidate in sorted(lib_dir.glob(pattern)):
        if candidate.name in skipped:
            continue
        libs.append(candidate.name)
    return libs


def _include_header(header: str) -> None:
    candidate = HEADER_PATHS.get(header)
    if candidate and candidate.exists():
        cppyy.include(str(candidate))
    else:
        cppyy.include(header)


def initialize():
    """Initialize cppyy and load all required libraries."""
    global _initialized, gbl
    if _initialized:
        return gbl

    for inc_path in INCLUDE_DIRS:
        if inc_path.exists():
            cppyy.add_include_path(str(inc_path))

    for lib_name in _collect_required_libs():
        lib_dir = (
            HELPER_LIB_DIR if lib_name == _GENERATION_HELPER_LIB else LLAMA_CPP_LIBS_DIR
        )
        lib_path = lib_dir / lib_name
        if not lib_path.exists():
            raise FileNotFoundError(f"Required library not found: {lib_path}")
        cppyy.load_library(str(lib_path))

    for header in [
        "vector",
        "string",
        "llama.h",
        "common.h",
        "sampling.h",
        "mtmd.h",
        "mtmd-helper.h",
        "generation_helper.h",
    ]:
        _include_header(header)

    gbl = cppyy.gbl
    _initialized = True
    return gbl


class LlamaBackend:
    """Context manager that owns llama backend initialization."""

    def __init__(self) -> None:
        self.gbl = initialize()

    def __enter__(self):
        self.gbl.llama_backend_init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gbl.llama_backend_free()


class ModelLoader:
    """Helpers for loading GGUF models and contexts."""

    def __init__(self, gbl):
        self.gbl = gbl

    def load_model(self, model_path: str, gpu_layers: int = 0):
        params = self.gbl.llama_model_default_params()
        params.n_gpu_layers = gpu_layers
        model = self.gbl.llama_load_model_from_file(model_path.encode("utf-8"), params)
        if not model:
            raise RuntimeError(f"Failed to load model: {model_path}")
        return model

    def create_context(
        self,
        model,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_threads: int = 8,
        n_seq_max: int = 1,
    ):
        params = self.gbl.llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_batch
        params.n_seq_max = max(1, n_seq_max)
        params.n_threads = params.n_threads_batch = n_threads
        params.log_level = self.gbl.GGML_LOG_LEVEL_ERROR

        ctx = self.gbl.llama_new_context_with_model(model, params)
        if not ctx:
            raise RuntimeError("Failed to create context")
        return ctx

    def load_multimodal(
        self, mmproj_path: str, model, use_gpu: bool = False, n_threads: int = 8
    ):
        params = self.gbl.mtmd_context_params_default()
        params.use_gpu = use_gpu
        params.n_threads = n_threads
        params.verbosity = self.gbl.GGML_LOG_LEVEL_ERROR

        ctx_mtmd = self.gbl.mtmd_init_from_file(
            mmproj_path.encode("utf-8"), model, params
        )
        if not ctx_mtmd:
            raise RuntimeError(f"Failed to load multimodal projector: {mmproj_path}")
        return ctx_mtmd


class MultimodalProcessor:
    """Utility class that marshals data into the multimodal pipeline."""

    def __init__(self, gbl):
        self.gbl = gbl
        self.loaded_embedding = None
        self.last_logits_index = 0

    def load_image(self, ctx_mtmd, image_path: str):
        bitmap = self.gbl.mtmd_helper_bitmap_init_from_file(
            ctx_mtmd, image_path.encode("utf-8")
        )
        if not bitmap:
            raise RuntimeError(f"Failed to load image: {image_path}")
        return bitmap

    def tokenize_prompt(self, ctx_mtmd, prompt: str, bitmaps):
        prompt_cstr = prompt.encode("utf-8")

        input_text = self.gbl.mtmd_input_text()
        input_text.text = prompt_cstr
        input_text.add_special = True
        input_text.parse_special = True

        if not isinstance(bitmaps, list):
            bitmaps = [bitmaps]

        bitmaps_vec = self.gbl.std.vector["const mtmd_bitmap*"]()
        for bitmap in bitmaps:
            bitmaps_vec.push_back(bitmap)

        chunks = self.gbl.mtmd_input_chunks_init()
        if not chunks:
            raise RuntimeError("Failed to initialize chunks")

        if (
            self.gbl.mtmd_tokenize(
                ctx_mtmd, chunks, input_text, bitmaps_vec.data(), bitmaps_vec.size()
            )
            != 0
        ):
            raise RuntimeError("Failed to tokenize multimodal prompt")

        return chunks

    def process_chunks(
        self, ctx, ctx_mtmd, chunks, n_batch: int = 512, seq_id: int = 0
    ):
        n_past = 0
        n_chunks = self.gbl.mtmd_input_chunks_size(chunks)
        seq_identifier = self.gbl.llama_seq_id(seq_id)
        self.last_logits_index = 0

        for i in range(n_chunks):
            chunk = self.gbl.mtmd_input_chunks_get(chunks, i)
            chunk_type = self.gbl.mtmd_input_chunk_get_type(chunk)
            is_last = i == n_chunks - 1

            if chunk_type == self.gbl.MTMD_INPUT_CHUNK_TYPE_TEXT:
                n_past = self._eval_text_chunk(
                    ctx, chunk, n_past, n_batch, is_last, seq_identifier=seq_identifier
                )
            elif chunk_type in (
                self.gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE,
                self.gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO,
            ):
                self.encode_media_chunk(ctx_mtmd, chunk)
                n_past = self._decode_media_chunk(
                    ctx, ctx_mtmd, chunk, n_past, n_batch, seq_identifier=seq_identifier
                )

        return n_past

    def encode_media_chunk(self, ctx_mtmd, chunk):
        if self.gbl.mtmd_encode_chunk(ctx_mtmd, chunk) != 0:
            raise RuntimeError("Failed to encode media chunk")

    def _decode_media_chunk(
        self,
        ctx,
        ctx_mtmd,
        chunk,
        n_past: int,
        n_batch: int,
        seq_identifier=None,
    ):
        embd = (
            self.loaded_embedding.data()
            if self.loaded_embedding
            else self.gbl.mtmd_get_output_embd(ctx_mtmd)
        )
        seq_id = seq_identifier or self.gbl.llama_seq_id(0)

        new_n_past_array = self.gbl.std.array[self.gbl.llama_pos, 1]()
        new_n_past_array[0] = self.gbl.llama_pos(n_past)

        if (
            self.gbl.mtmd_helper_decode_image_chunk(
                ctx_mtmd,
                ctx,
                chunk,
                embd,
                n_past,
                seq_id,
                n_batch,
                new_n_past_array.data(),
            )
            != 0
        ):
            raise RuntimeError("Failed to decode media chunk")

        return int(new_n_past_array[0])

    def decode_media_chunks_batched(
        self,
        ctx,
        ctx_mtmd,
        chunks,
        n_pasts,
        seq_ids,
        n_batch: int,
    ):
        if not chunks:
            return []

        decode_fn = None
        try:
            decode_fn = self.gbl.decode_media_chunks_batch
        except AttributeError:
            try:
                decode_fn = self.gbl.mtmd_helper_decode_image_chunks_batch
            except AttributeError:
                raise RuntimeError(
                    "Batched media decode helper not found in loaded libraries"
                )

        chunk_vec = self.gbl.std.vector["const mtmd_input_chunk *"]()
        for chunk in chunks:
            chunk_vec.push_back(chunk)

        n_past_vec = self.gbl.std.vector[self.gbl.llama_pos]()
        for pos in n_pasts:
            n_past_vec.push_back(self.gbl.llama_pos(pos))

        seq_vec = self.gbl.std.vector[self.gbl.llama_seq_id]()
        for seq_id in seq_ids:
            seq_vec.push_back(seq_id)

        result_vec = self.gbl.std.vector[self.gbl.llama_pos]()
        result_vec.resize(len(chunks))

        ret = decode_fn(
            ctx_mtmd,
            ctx,
            chunk_vec.data(),
            n_past_vec.data(),
            seq_vec.data(),
            len(chunks),
            n_batch,
            result_vec.data(),
        )
        if ret != 0:
            raise RuntimeError("Failed to decode batched media chunks")

        return [int(result_vec[i]) for i in range(len(chunks))]

    def _eval_text_chunk(
        self,
        ctx,
        chunk,
        n_past: int,
        n_batch: int,
        logits_for_last: bool = False,
        seq_identifier=None,
    ):
        seq_id = seq_identifier or self.gbl.llama_seq_id(0)

        # Prefer the native helper to cut Python<->C churn during prefill.
        try:
            prefill_res = self.gbl.eval_text_chunk_cpp(
                ctx, chunk, n_past, n_batch, logits_for_last, seq_id
            )
            if int(prefill_res.status) != 0:
                raise RuntimeError("Failed to decode text chunk")
            if logits_for_last:
                self.last_logits_index = int(prefill_res.last_logits_index)
            return int(prefill_res.final_n_past)
        except AttributeError:
            pass

        n_tokens_out = self.gbl.std.vector["size_t"](1)
        tokens = self.gbl.mtmd_input_chunk_get_tokens_text(chunk, n_tokens_out.data())
        n_tokens = n_tokens_out[0]

        batch = self.gbl.llama_batch_init(n_batch, 0, 1)
        token_idx = 0
        while token_idx < n_tokens:
            batch.n_tokens = 0
            while token_idx < n_tokens and batch.n_tokens < n_batch:
                j = batch.n_tokens
                batch.token[j] = tokens[token_idx]
                batch.pos[j] = n_past
                batch.n_seq_id[j] = 1
                batch.seq_id[j][0] = seq_id
                batch.logits[j] = False
                batch.n_tokens += 1
                n_past += 1
                token_idx += 1

            if logits_for_last and token_idx == n_tokens:
                batch.logits[batch.n_tokens - 1] = True
                self.last_logits_index = batch.n_tokens - 1

            if self.gbl.llama_decode(ctx, batch) != 0:
                self.gbl.llama_batch_free(batch)
                raise RuntimeError("Failed to decode text chunk")

        self.gbl.llama_batch_free(batch)
        return n_past


class TextGenerator:
    """Sampling helpers built on top of llama.cpp common sampler."""

    def __init__(self, gbl):
        self.gbl = gbl

    def create_sampler(
        self,
        model,
        temp: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        ctx_size: int = 2048,
    ):
        params = self.gbl.common_params_sampling()
        params.temp = temp
        params.top_k = top_k
        params.top_p = top_p
        params.penalty_repeat = repeat_penalty
        params.penalty_last_n = ctx_size
        params.grammar = ""

        sampler = self.gbl.common_sampler_init(model, params)
        if not sampler:
            raise RuntimeError("Failed to create sampler")
        return sampler

    def generate(
        self, sampler, ctx, model, n_past: int, n_ctx: int, max_tokens: int = 256
    ):
        self.gbl.common_sampler_reset(sampler)
        seq_id_vec = self.gbl.std.vector[self.gbl.llama_seq_id](
            [self.gbl.llama_seq_id(0)]
        )
        result = self.gbl.generate_tokens_cpp(
            sampler, ctx, model, n_past, n_ctx, max_tokens, seq_id_vec
        )
        return result


__all__ = [
    "LlamaBackend",
    "ModelLoader",
    "MultimodalProcessor",
    "TextGenerator",
    "initialize",
]
