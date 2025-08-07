"""Multimodal generation using mtmd library with Gemma-3 model and image input (Cython version)."""

import os
import time
import sys
import argparse
import json
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager
from huggingface_hub import hf_hub_download

# Configuration
# Handle both cases: running from cython-src/ and from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "cython-src":
    # Running from cython-src directory
    BASE_DIR = os.path.dirname(script_dir)
else:
    # Running from project root (e.g., via uv run cython-src/cython-mtmd.py)
    BASE_DIR = os.getcwd()
LLAMA_CPP_DIR = f"{BASE_DIR}/llama.cpp"

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

# Global benchmark results
benchmark_results = {
    "version": "cython",
    "model_loading_time": 0.0,
    "prompt_processing_time": 0.0,
    "image_processing_time": 0.0,
    "token_generation_rate": 0.0,
    "total_tokens_generated": 0,
    "final_output": "",
    "total_time": 0.0,
    "timestamp": "",
    "model_info": {},
    "parameters": {}
}


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


# Import our Cython module
try:
    import pyximport

    pyximport.install()
    from llama_mtmd_wrapper import (
        initialize_backend,
        cleanup_backend,
        load_model,
        create_context,
        load_mtmd_context,
        create_sampler,
        load_image,
        tokenize_input,
        evaluate_input,
        generate_tokens,
    )
except ImportError as e:
    print(f"Error importing Cython module: {e}")
    print("Make sure you've built the Cython extension with:")
    print("python setup.py build_ext --inplace")
    sys.exit(1)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run multimodal generation with Cython wrapper."
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

        # Store model info and parameters for benchmark
        benchmark_results["model_info"] = {
            "repo_id": args.repo_id,
            "model_file": args.model,
            "mmproj_file": args.mmproj,
            "model_path": model_path,
            "mmproj_path": mmproj_path
        }
        benchmark_results["parameters"] = {
            "n_ctx": N_CTX,
            "n_batch": N_BATCH,
            "n_threads": N_THREADS,
            "n_gpu_layers": N_GPU_LAYERS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temp": TEMP,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "repeat_penalty": REPEAT_PENALTY
        }
        benchmark_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with timed_operation("Backend initialization"):
            initialize_backend()

        # Load model and create contexts
        start_model_load = time.time()
        with timed_operation("Model loading"):
            model = load_model(model_path, N_GPU_LAYERS)

        with timed_operation("Context creation"):
            ctx = create_context(model, N_CTX, N_BATCH, N_THREADS)

        with timed_operation("Multimodal projector loading"):
            ctx_mtmd = load_mtmd_context(
                mmproj_path, model, N_GPU_LAYERS > 0, N_THREADS
            )

        with timed_operation("Sampler initialization"):
            sampler = create_sampler(model, TEMP, TOP_K, TOP_P, REPEAT_PENALTY, N_CTX)
        benchmark_results["model_loading_time"] = time.time() - start_model_load

        # Load image
        print("Loading image...")
        start_image_load = time.time()
        with timed_operation("Image loading"):
            bitmap_info = load_image(ctx_mtmd, args.image)
            print(f"Image loaded: {bitmap_info['width']}x{bitmap_info['height']}")
        benchmark_results["image_processing_time"] = time.time() - start_image_load

        # Prepare and evaluate multimodal input
        print("Evaluating multimodal input...")

        # Tokenize input
        with timed_operation("Tokenization"):
            chunks_info = tokenize_input(ctx_mtmd, args.prompt, bitmap_info["handle"])

        # Process prompt tokens
        prompt_tokens = chunks_info["n_tokens"]
        start_prompt_eval = time.time()
        with timed_operation("Prompt evaluation", tokens=prompt_tokens):
            n_past = evaluate_input(ctx_mtmd, ctx, chunks_info["handle"], N_BATCH)
        benchmark_results["prompt_processing_time"] = time.time() - start_prompt_eval

        print(f"KV cache position (n_past): {n_past}")

        # Generate response
        print(f"\n--- Generating Response ({MAX_NEW_TOKENS} tokens max) ---")
        print(f"{args.prompt}", end="", flush=True)

        # Generate tokens
        with timed_operation("Token generation") as timing_ctx:
            result = generate_tokens(sampler, ctx, model, n_past, N_CTX, MAX_NEW_TOKENS)

            # Print results
            print(f"{result['text']}")

            # Update timing with token count
            timing_ctx.tokens = result["n_tokens"]

            # Store benchmark results
            benchmark_results["total_tokens_generated"] = result["n_tokens"]
            benchmark_results["final_output"] = result["text"]
            if timing_ctx.duration > 0:
                benchmark_results["token_generation_rate"] = result["n_tokens"] / timing_ctx.duration

        print(f"Final KV cache position (n_past): {result['final_n_past']}")

        # Cleanup
        with timed_operation("Backend cleanup"):
            cleanup_backend()

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

# Save benchmark results
benchmark_results["total_time"] = total_time
benchmark_filename = "benchmark.json"
try:
    with open(benchmark_filename, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nBenchmark results saved to {benchmark_filename}")
except Exception as e:
    print(f"Warning: Failed to save benchmark results: {e}")


if __name__ == "__main__":
    main()
