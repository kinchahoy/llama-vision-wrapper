"""
A Python wrapper for llama.cpp using cppyy, providing a high-level interface
for model loading, context management, and multimodal operations.
"""
import cppyy
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

# Configuration
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # Go up from cpppy/ to project root
LLAMA_CPP_DIR = f"{BASE_DIR}/llama.cpp"
INCLUDE_DIRS = [
    f"{LLAMA_CPP_DIR}/include",
    f"{LLAMA_CPP_DIR}/ggml/include",
    f"{LLAMA_CPP_DIR}/common",
    f"{LLAMA_CPP_DIR}/tools/mtmd",
    f"{BASE_DIR}/gen-helper",
]
LLAMA_CPP_LIBS_DIR = f"{LLAMA_CPP_DIR}/build/bin"
HELPER_LIB_DIR = f"{BASE_DIR}/gen-helper/build"

# Dynamics libraries are different across linux and macos
if sys.platform == "linux":
    LIB_EXT = "so"
elif sys.platform == "darwin":
    LIB_EXT = "dylib"
else:
    raise RuntimeError("Unsupported platform")

# Libraries to load (Linux .so files instead of macOS .dylib)
# Required libraries
REQUIRED_LIB_NAMES = [
    f"libggml-base.{LIB_EXT}",
    f"libggml-cpu.{LIB_EXT}",
    f"libggml.{LIB_EXT}",
    f"libllama.{LIB_EXT}",
    f"libmtmd.{LIB_EXT}",
    f"libgeneration_helper.{LIB_EXT}",
]

# Optional libraries (will continue if not found)
OPTIONAL_LIB_NAMES = [
    f"libggml-blas.{LIB_EXT}",
    f"libggml-metal.{LIB_EXT}",
]

# Global list to collect timing statistics
all_timings = []


@dataclass
class TimingStats:
    """Stores timing statistics for a processing step."""

    name: str
    duration: float
    tokens: Optional[int] = None

    def __str__(self) -> str:
        """Format timing statistics as a string."""
        result = f"{self.name}: {self.duration:.2f}s"
        if self.tokens is not None:
            tokens_per_sec = (
                self.tokens / self.duration if self.duration > 0 else float("inf")
            )
            result += f" ({self.tokens} tokens, {tokens_per_sec:.2f} tokens/s)"
        return result


@contextmanager
def timed_operation(name: str, tokens: Optional[int] = None):
    """Context manager to time an operation and return statistics."""
    start_time = time.time()
    timing_ctx = TimingStats(name=name, duration=0, tokens=tokens)
    try:
        yield timing_ctx
    finally:
        timing_ctx.duration = time.time() - start_time
        all_timings.append(timing_ctx)
        print(f"⏱️ {timing_ctx}")


def initialize_cppyy():
    """Initializes cppyy by setting up include paths and loading libraries."""
    with timed_operation("cppyy setup"):
        # Setup cppyy
        for inc_path in INCLUDE_DIRS:
            if os.path.exists(inc_path):
                cppyy.add_include_path(inc_path)
            else:
                print(f"Warning: Include path does not exist: {inc_path}")

        print("Loading libraries...")

        # Load required libraries
        for lib_name in REQUIRED_LIB_NAMES:
            lib_dir = (
                HELPER_LIB_DIR
                if lib_name == f"libgeneration_helper.{LIB_EXT}"
                else LLAMA_CPP_LIBS_DIR
            )
            lib_path = os.path.join(lib_dir, lib_name)
            if not os.path.exists(lib_path):
                raise FileNotFoundError(
                    f"Required library '{lib_name}' not found at {lib_path}"
                )
            cppyy.load_library(lib_path)
            print(f"  Loaded: {lib_name}")

        # Load optional libraries
        for lib_name in OPTIONAL_LIB_NAMES:
            lib_dir = LLAMA_CPP_LIBS_DIR
            lib_path = os.path.join(lib_dir, lib_name)
            if os.path.exists(lib_path):
                try:
                    cppyy.load_library(lib_path)
                    print(f"  Loaded (optional): {lib_name}")
                except Exception as e:
                    print(
                        f"  Warning: Failed to load optional library {lib_name}: {e}"
                    )
            else:
                print(f"  Skipped (not found): {lib_name}")

        # Include headers
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
            cppyy.include(header)


# Initialize cppyy on module import
initialize_cppyy()
gbl = cppyy.gbl


# Resource manager class
class ResourceManager:
    """Combined resource manager for all llama.cpp resources"""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.resources = {}
        self.timings = []
        self.loaded_embedding = None

    def log(self, msg):
        if self.verbose:
            print(msg)

    def __enter__(self):
        with timed_operation("Backend initialization"):
            self.log("Initializing llama backend...")
            cppyy.gbl.llama_backend_init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with timed_operation("Backend cleanup"):
            self.log("Cleaning up resources...")
            cppyy.gbl.llama_backend_free()
        return False

    def load_model(self, model_path, gpu_layers=0):
        self.log(f"Loading model from {model_path}...")
        params = cppyy.gbl.llama_model_default_params()
        params.n_gpu_layers = gpu_layers

        with timed_operation("Model loading"):
            model = cppyy.gbl.llama_load_model_from_file(
                model_path.encode("utf-8"), params
            )
            if not model:
                raise RuntimeError(f"Failed to load model from {model_path}")

        self.log(f"Model loaded (GPU layers: {gpu_layers})")
        return model

    def create_context(self, model, n_ctx, n_batch, n_threads, verbose_cpp=False):
        self.log("Creating LLaMA context...")
        params = cppyy.gbl.llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_batch
        params.n_threads = params.n_threads_batch = n_threads
        params.log_level = (
            cppyy.gbl.GGML_LOG_LEVEL_INFO
            if verbose_cpp
            else cppyy.gbl.GGML_LOG_LEVEL_ERROR
        )

        with timed_operation("Context creation"):
            ctx = cppyy.gbl.llama_new_context_with_model(model, params)
            if not ctx:
                raise RuntimeError("Failed to create LLaMA context")

        self.log(f"LLaMA context created (n_ctx: {cppyy.gbl.llama_n_ctx(ctx)})")

        # Check for single-threaded decoding, which can be a major performance bottleneck.
        if n_threads == 1:
            self.log(
                "[WARN] Decoding with 1 thread, performance may be poor. "
                "Adjust --n-threads for a significant speed up."
            )

        return ctx

    def load_mtmd(self, mmproj_path, model, use_gpu, n_threads, verbose_cpp=False):
        self.log(f"Loading multimodal context from {mmproj_path}...")

        # Check if file exists and is readable
        if not os.path.exists(mmproj_path):
            raise RuntimeError(
                f"Multimodal projector file does not exist: {mmproj_path}"
            )

        file_size = os.path.getsize(mmproj_path)
        self.log(f"Projector file size: {file_size} bytes")

        params = cppyy.gbl.mtmd_context_params_default()
        params.use_gpu = use_gpu
        params.n_threads = n_threads
        params.verbosity = (
            cppyy.gbl.GGML_LOG_LEVEL_INFO
            if verbose_cpp
            else cppyy.gbl.GGML_LOG_LEVEL_ERROR
        )

        with timed_operation("Multimodal projector loading"):
            ctx_mtmd = cppyy.gbl.mtmd_init_from_file(
                mmproj_path.encode("utf-8"), model, params
            )
            if not ctx_mtmd:
                raise RuntimeError(
                    f"Failed to load multimodal projector from {mmproj_path}"
                )

        self.log(f"Multimodal context loaded (GPU: {use_gpu})")
        return ctx_mtmd

    def create_sampler(self, model, temp, top_k, top_p, repeat_penalty, ctx_size):
        self.log("Initializing sampler...")
        params = cppyy.gbl.common_params_sampling()
        params.temp = temp
        params.top_k = top_k
        params.top_p = top_p
        params.penalty_repeat = repeat_penalty
        params.penalty_last_n = ctx_size
        params.grammar = ""

        with timed_operation("Sampler initialization"):
            sampler = cppyy.gbl.common_sampler_init(model, params)
            if not sampler:
                raise RuntimeError("Failed to initialize common_sampler")

        self.log("Sampler initialized.")
        return sampler

    def load_image(self, ctx_mtmd, image_path):
        with timed_operation("Image loading") as timing_ctx:
            bitmap = gbl.mtmd_helper_bitmap_init_from_file(
                ctx_mtmd, image_path.encode("utf-8")
            )
            if not bitmap:
                raise RuntimeError(f"Failed to load image {image_path}")
        return bitmap, timing_ctx.duration

    def tokenize_prompt(self, ctx_mtmd, prompt, bitmaps):
        # Keep prompt string alive
        prompt_cstr = prompt.encode("utf-8")

        input_text = gbl.mtmd_input_text()
        input_text.text = prompt_cstr
        input_text.add_special = True
        input_text.parse_special = True

        bitmaps_ptr_vec = gbl.std.vector["const mtmd_bitmap*"]()
        # Handle both single bitmap and list of bitmaps
        if not isinstance(bitmaps, list):
            bitmaps = [bitmaps]
        for bitmap in bitmaps:
            bitmaps_ptr_vec.push_back(bitmap)

        chunks = gbl.mtmd_input_chunks_init()
        if not chunks:
            raise RuntimeError(
                "Failed to initialize mtmd_input_chunks: mtmd_input_chunks_init returned null"
            )

        with timed_operation("Tokenization"):
            if gbl.mtmd_tokenize(
                ctx_mtmd,
                chunks,
                input_text,
                bitmaps_ptr_vec.data(),
                bitmaps_ptr_vec.size(),
            ) != 0:
                raise RuntimeError("Failed mtmd_tokenize")
        return chunks

    def encode_image_chunk(self, ctx_mtmd, chunk):
        if gbl.mtmd_input_chunk_get_type(chunk) not in [
            gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE,
            gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO,
        ]:
            raise ValueError("Chunk is not an image or audio chunk")

        with timed_operation("Media encoding"):
            if gbl.mtmd_encode_chunk(ctx_mtmd, chunk) != 0:
                raise RuntimeError("Failed to encode media chunk")

    def save_encoded_chunk(self, ctx_mtmd, model, chunk, file_path):
        with timed_operation("Saving media embedding"):
            image_tokens = gbl.mtmd_input_chunk_get_tokens_image(chunk)
            if not image_tokens:
                raise ValueError("Chunk does not contain image tokens")

            nx = gbl.mtmd_image_tokens_get_nx(image_tokens)
            ny = gbl.mtmd_image_tokens_get_ny(image_tokens)
            use_mrope_pos = gbl.mtmd_decode_use_mrope(ctx_mtmd)

            if self.loaded_embedding:
                embedding_ptr = self.loaded_embedding.data()
                embedding_size = self.loaded_embedding.size()
            else:
                embedding_ptr = gbl.mtmd_get_output_embd(ctx_mtmd)
                n_tokens = gbl.mtmd_image_tokens_get_n_tokens(image_tokens)
                n_embd = gbl.llama_model_n_embd(model)
                embedding_size = n_tokens * n_embd

            if not embedding_ptr:
                raise RuntimeError("Failed to get embedding pointer")

            if not gbl.save_media_embedding(
                file_path, nx, ny, use_mrope_pos, embedding_size, embedding_ptr
            ):
                raise RuntimeError(f"Failed to save embedding to {file_path}")

    def load_encoded_chunk(self, ctx_mtmd, model, chunk, file_path):
        with timed_operation("Loading media embedding"):
            image_tokens = gbl.mtmd_input_chunk_get_tokens_image(chunk)
            if not image_tokens:
                raise ValueError("Chunk does not contain image tokens")

            self.loaded_embedding = gbl.std.vector["float"]()
            load_result = gbl.load_media_embedding(
                file_path, self.loaded_embedding
            )

            if not load_result.success:
                raise RuntimeError(f"Failed to load embedding from {file_path}")

            # Verify that the loaded embedding is compatible with the current model context
            use_mrope_pos_from_ctx = gbl.mtmd_decode_use_mrope(ctx_mtmd)
            if load_result.use_mrope_pos != use_mrope_pos_from_ctx:
                raise RuntimeError(
                    f"M-RoPE mismatch: embedding file expects use_mrope_pos={load_result.use_mrope_pos}, "
                    f"but model context is configured with use_mrope_pos={use_mrope_pos_from_ctx}."
                )

            image_tokens.nx = load_result.nx
            image_tokens.ny = load_result.ny

            # Verify embedding size after loading
            n_tokens = gbl.mtmd_image_tokens_get_n_tokens(image_tokens)
            n_embd = gbl.llama_model_n_embd(model)
            expected_size = n_tokens * n_embd

            if self.loaded_embedding.size() != expected_size:
                raise IOError(
                    f"Embedding file has wrong size. Expected {expected_size} floats, got {self.loaded_embedding.size()}"
                )

    def eval_text_chunk(
        self, ctx, chunk, n_past, n_batch, logits_for_last_token=False
    ):
        n_tokens_out = gbl.std.vector["size_t"](1)
        tokens = gbl.mtmd_input_chunk_get_tokens_text(chunk, n_tokens_out.data())
        n_tokens = n_tokens_out[0]
        seq_id = gbl.llama_seq_id(0)

        text_batch = gbl.llama_batch_init(n_batch, 0, 1)

        token_idx = 0
        while token_idx < n_tokens:
            text_batch.n_tokens = 0
            while token_idx < n_tokens and text_batch.n_tokens < n_batch:
                j = text_batch.n_tokens
                text_batch.token[j] = tokens[token_idx]
                text_batch.pos[j] = n_past
                text_batch.n_seq_id[j] = 1
                text_batch.seq_id[j][0] = seq_id
                text_batch.logits[j] = False
                text_batch.n_tokens += 1
                n_past += 1
                token_idx += 1

            is_last_token_in_chunk = token_idx == n_tokens
            if logits_for_last_token and is_last_token_in_chunk:
                text_batch.logits[text_batch.n_tokens - 1] = True

            if gbl.llama_decode(ctx, text_batch) != 0:
                gbl.llama_batch_free(text_batch)
                raise RuntimeError("Failed to decode text")

        gbl.llama_batch_free(text_batch)
        return n_past

    def decode_image_chunk(self, ctx, ctx_mtmd, chunk, n_past, n_batch):
        if self.loaded_embedding:
            embd = self.loaded_embedding.data()
        else:
            embd = gbl.mtmd_get_output_embd(ctx_mtmd)
        seq_id = gbl.llama_seq_id(0)

        new_n_past_out_array = gbl.std.array[gbl.llama_pos, 1]()
        new_n_past_out_array[0] = gbl.llama_pos(n_past)

        if (
            gbl.mtmd_helper_decode_image_chunk(
                ctx_mtmd,
                ctx,
                chunk,
                embd,
                n_past,
                seq_id,
                n_batch,
                new_n_past_out_array.data(),
            )
            != 0
        ):
            raise RuntimeError("Failed to decode media chunk")

        return int(new_n_past_out_array[0])

    def generate(self, sampler, ctx, model, n_past, n_ctx, max_new_tokens):
        gbl.common_sampler_reset(sampler)
        seq_id_vec = gbl.std.vector[gbl.llama_seq_id]([gbl.llama_seq_id(0)])

        with timed_operation("Token generation") as timing_ctx:
            cpp_result = gbl.generate_tokens_cpp(
                sampler, ctx, model, n_past, n_ctx, max_new_tokens, seq_id_vec
            )
            timing_ctx.tokens = cpp_result.total_tokens_generated

        return cpp_result, timing_ctx
