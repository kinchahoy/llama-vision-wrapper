"""Multimodal generation using mtmd library with Gemma-3 model and image input (Cython version)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from llama_insight import (
        Config,
        Timer,
        add_common_args,
        download_models,
    )
except ImportError:  # pragma: no cover - local editable install
    sys.path.append(str(PROJECT_ROOT / "wrapper_src"))
    from llama_insight import (  # type: ignore  # noqa
        Config,
        Timer,
        add_common_args,
        download_models,
    )


# Import our Cython module
try:
    import pyximport

    pyximport.install()
    from llama_mtmd_cython_wrapper import (
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
    add_common_args(parser)
    parser.add_argument(
        "--image", type=str, default="debug.jpg", help="Path to the input image."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="USER: Describe this image.\n<__image__>\nASSISTANT:",
        help="The prompt for the model.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmarking and save results to benchmark.json.",
    )
    args = parser.parse_args()
    config = Config.from_args(args)
    timer = Timer()
    benchmark_results = None
    if args.benchmark:
        benchmark_results = {
            "version": "cython",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": {
                "repo_id": config.repo_id,
                "model_file": config.model,
                "mmproj_file": config.mmproj,
            },
            "parameters": {
                "n_ctx": config.n_ctx,
                "n_batch": config.n_batch,
                "n_threads": config.n_threads,
                "n_gpu_layers": config.n_gpu_layers,
                "max_new_tokens": config.max_new_tokens,
                "temp": config.temp,
                "top_k": config.top_k,
                "top_p": config.top_p,
                "repeat_penalty": config.repeat_penalty,
            },
        }

    try:
        print("--- Initializing ---")

        print("--- Downloading models from Hugging Face Hub ---")
        model_path, mmproj_path = download_models(config)
        if benchmark_results is not None:
            benchmark_results["model_info"].update(
                {
                    "model_path": model_path,
                    "mmproj_path": mmproj_path,
                }
            )

        with timer.time_operation("Backend initialization"):
            initialize_backend()

        # Load model and create contexts
        start_model_load = time.time()
        with timer.time_operation("Model loading"):
            model = load_model(model_path, config.n_gpu_layers)

        with timer.time_operation("Context creation"):
            ctx = create_context(model, config.n_ctx, config.n_batch, config.n_threads)

        with timer.time_operation("Multimodal projector loading"):
            ctx_mtmd = load_mtmd_context(
                mmproj_path, model, config.n_gpu_layers > 0, config.n_threads
            )

        with timer.time_operation("Sampler initialization"):
            sampler = create_sampler(
                model,
                config.temp,
                config.top_k,
                config.top_p,
                config.repeat_penalty,
                config.n_ctx,
            )
        if benchmark_results is not None:
            benchmark_results["model_loading_time"] = time.time() - start_model_load

        # Load image
        print("Loading image...")
        start_image_load = time.time()
        image_path = str(Path(args.image).expanduser())
        with timer.time_operation("Image loading"):
            bitmap_info = load_image(ctx_mtmd, image_path)
            print(f"Image loaded: {bitmap_info['width']}x{bitmap_info['height']}")
        if benchmark_results is not None:
            benchmark_results["image_processing_time"] = time.time() - start_image_load

        # Prepare and evaluate multimodal input
        print("Evaluating multimodal input...")

        # Tokenize input
        with timer.time_operation("Tokenization"):
            chunks_info = tokenize_input(ctx_mtmd, args.prompt, bitmap_info["handle"])

        # Process prompt tokens
        prompt_tokens = chunks_info["n_tokens"]
        start_prompt_eval = time.time()
        with timer.time_operation("Prompt evaluation", tokens=prompt_tokens):
            n_past = evaluate_input(
                ctx_mtmd, ctx, chunks_info["handle"], config.n_batch
            )
        if benchmark_results is not None:
            benchmark_results["prompt_processing_time"] = (
                time.time() - start_prompt_eval
            )

        print(f"KV cache position (n_past): {n_past}")

        # Generate response
        print(f"\n--- Generating Response ({config.max_new_tokens} tokens max) ---")
        print(f"{args.prompt}", end="", flush=True)

        # Generate tokens
        with timer.time_operation("Token generation"):
            result = generate_tokens(
                sampler, ctx, model, n_past, config.n_ctx, config.max_new_tokens
            )
        # Assign token count to the most recent timing entry
        if timer.timings:
            name, duration, _ = timer.timings[-1]
            timer.timings[-1] = (name, duration, int(result["n_tokens"]))

        print(f"{result['text']}")

        if benchmark_results is not None:
            benchmark_results["total_tokens_generated"] = result["n_tokens"]
            benchmark_results["final_output"] = result["text"]
            last_duration = timer.timings[-1][1] if timer.timings else 0
            if last_duration > 0:
                benchmark_results["token_generation_rate"] = (
                    result["n_tokens"] / last_duration
                )

        print(f"Final KV cache position (n_past): {result['final_n_past']}")

        # Cleanup
        with timer.time_operation("Backend cleanup"):
            cleanup_backend()

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)

    print("\n--- End of script ---")

    if benchmark_results is not None:
        print("\nSummary of timing statistics:")
        print("=" * 50)
        for i, (name, duration, tokens) in enumerate(timer.timings, 1):
            if tokens:
                rate = tokens / duration if duration > 0 else float("inf")
                print(
                    f"{i}. {name}: {duration:.2f}s ({tokens} tokens, {rate:.2f} tok/s)"
                )
            else:
                print(f"{i}. {name}: {duration:.2f}s")
        print("=" * 50)
        print(f"Total operations: {len(timer.timings)}")
        total_time = sum(duration for _, duration, _ in timer.timings)
        print(f"Total measured time: {total_time:.2f}s")

        # Save benchmark results
        benchmark_results["total_time"] = int(total_time)
        benchmark_filename = "benchmark.json"
        try:
            with open(benchmark_filename, "w") as f:
                json.dump(benchmark_results, f, indent=2)
            print(f"\nBenchmark results saved to {benchmark_filename}")
        except Exception as e:
            print(f"Warning: Failed to save benchmark results: {e}")


if __name__ == "__main__":
    main()
