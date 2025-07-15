"""Multimodal generation using mtmd library with Huggingface GGUF models and image input."""

import cppyy
import os
import time
import sys
import argparse
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager
from huggingface_hub import hf_hub_download

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

# Libraries to load (Linux .so files instead of macOS .dylib)
# Required libraries
REQUIRED_LIB_NAMES = [
    "libggml-base.so",
    "libggml-cpu.so",
    "libggml.so",
    "libllama.so",
    "libmtmd.so",
    "libgeneration_helper.so",
]

# Optional libraries (will continue if not found)
OPTIONAL_LIB_NAMES = [
    "libggml-blas.so",
    "libggml-metal.so",
]


# Runtime parameters
N_CTX = 2048
N_THREADS = 8
N_GPU_LAYERS = 0
N_BATCH = 512
MAX_NEW_TOKENS = 256

# Sampling parameters
TEMP = 0.8
TOP_K = 40
TOP_P = 0.95
REPEAT_PENALTY = 1.1


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


# Resource manager class
class ResourceManager:
    """Combined resource manager for all llama.cpp resources"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.resources = {}
        self.timings = []

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

    def create_context(self, model, n_ctx, n_batch, n_threads):
        self.log("Creating LLaMA context...")
        params = cppyy.gbl.llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_batch
        params.n_threads = params.n_threads_batch = n_threads
        params.log_level = cppyy.gbl.GGML_LOG_LEVEL_ERROR

        with timed_operation("Context creation"):
            ctx = cppyy.gbl.llama_new_context_with_model(model, params)
            if not ctx:
                raise RuntimeError("Failed to create LLaMA context")

        self.log(f"LLaMA context created (n_ctx: {cppyy.gbl.llama_n_ctx(ctx)})")
        return ctx

    def load_mtmd(self, mmproj_path, model, use_gpu, n_threads):
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
        )  # Increase verbosity for debugging

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


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run multimodal generation with cppyy."
    )
    parser.add_argument(
        "--repo-id",
        "-hf",
        type=str,
        default="ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
        help="Hugging Face repository ID for GGUF models.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        help="Model file name in the repository.",
    )
    parser.add_argument(
        "--mmproj",
        type=str,
        default="mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
        help="Multimodal projector file name in the repository.",
    )
    parser.add_argument(
        "--image", type=str, default="debug.jpg", help="Path to the input image."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="USER: Describe this image.\n<__image__>\nASSISTANT:",
        help="The prompt for the model.",
    )
    args = parser.parse_args()

    try:
        print("--- Initializing ---")

        print("--- Downloading models from Hugging Face Hub ---")
        with timed_operation("Model download"):
            model_path = hf_hub_download(repo_id=args.repo_id, filename=args.model)
        with timed_operation("MMPROJ download"):
            mmproj_path = hf_hub_download(repo_id=args.repo_id, filename=args.mmproj)
        print(f"Model path: {model_path}")
        print(f"MMPROJ path: {mmproj_path}")

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
                    if lib_name == "libgeneration_helper.so"
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
                "llama.h",
                "common.h",
                "sampling.h",
                "mtmd.h",
                "mtmd-helper.h",
                "generation_helper.h",
            ]:
                cppyy.include(header)
            gbl = cppyy.gbl

        # Initialize resources
        with ResourceManager() as rm:
            # Load model and create contexts
            model = rm.load_model(model_path, N_GPU_LAYERS)
            ctx = rm.create_context(model, N_CTX, N_BATCH, N_THREADS)
            ctx_mtmd = rm.load_mtmd(mmproj_path, model, N_GPU_LAYERS > 0, N_THREADS)
            sampler = rm.create_sampler(
                model, TEMP, TOP_K, TOP_P, REPEAT_PENALTY, gbl.llama_n_ctx(ctx)
            )

            # Load image
            print("Loading image...")
            with timed_operation("Image loading"):
                bitmap = gbl.mtmd_helper_bitmap_init_from_file(
                    ctx_mtmd, args.image.encode("utf-8")
                )
                if not bitmap:
                    raise RuntimeError(f"Failed to load image {args.image}")
                print(
                    f"Image loaded: {gbl.mtmd_bitmap_get_nx(bitmap)}x{gbl.mtmd_bitmap_get_ny(bitmap)}"
                )

            # Prepare and evaluate multimodal input
            print("Evaluating multimodal input...")

            # Setup input
            prompt_bytes = args.prompt.encode("utf-8")
            input_text = cppyy.gbl.mtmd_input_text()
            try:
                input_text.text = prompt_bytes
                input_text.add_special = True
                input_text.parse_special = True

                bitmaps_ptr_vec = gbl.std.vector["const mtmd_bitmap*"]()
                bitmaps_ptr_vec.push_back(bitmap)
                chunks = gbl.mtmd_input_chunks_init()

                # Tokenize and evaluate
                with timed_operation("Tokenization"):
                    if (
                        gbl.mtmd_tokenize(
                            ctx_mtmd,
                            chunks,
                            input_text,
                            bitmaps_ptr_vec.data(),
                            bitmaps_ptr_vec.size(),
                        )
                        != 0
                    ):
                        raise RuntimeError("Failed mtmd_tokenize")
            except:
                raise RuntimeError("Failed to initialize mtmd_input_text")

            n_past = 0
            n_past_out = gbl.llama_pos(0)
            seq_id = gbl.llama_seq_id(0)

            # Process prompt tokens
            prompt_tokens = gbl.mtmd_helper_get_n_tokens(chunks)
            with timed_operation("Prompt evaluation", tokens=prompt_tokens):
                if (
                    gbl.mtmd_helper_eval_chunks(
                        ctx_mtmd,
                        ctx,
                        chunks,
                        n_past,
                        seq_id,
                        N_BATCH,
                        False,
                        cppyy.addressof(n_past_out),
                    )
                    != 0
                ):
                    raise RuntimeError("Failed mtmd_helper_eval_chunks")

            # Update KV cache position
            n_past = n_past_out
            print(f"KV cache position (n_past): {n_past}")

            # Generate response
            print(f"\n--- Generating Response ({MAX_NEW_TOKENS} tokens max) ---")
            print(f"{args.prompt}", end="", flush=True)

            # Call C++ generation function
            seq_id_vec = gbl.std.vector[gbl.llama_seq_id]([gbl.llama_seq_id(0)])

            # Generate tokens using C++ helper function
            with timed_operation("Token generation") as timing_ctx:
                cpp_result = gbl.generate_tokens_cpp(
                    sampler, ctx, model, n_past, N_CTX, MAX_NEW_TOKENS, seq_id_vec
                )

                # Print results
                print(f"{cpp_result.generated_text}")

                # Update timing with token count
                timing_ctx.tokens = cpp_result.total_tokens_generated

            print(f"Final KV cache position (n_past): {cpp_result.final_n_past}")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)


print("\n--- End of script ---")
print("\nSummary of timing statistics:")
print("=" * 50)
for i, stat in enumerate(all_timings, 1):
    print(f"{i}. {stat}")
print("=" * 50)
print(f"Total operations: {len(all_timings)}")
total_time = sum(stat.duration for stat in all_timings)
print(f"Total measured time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
