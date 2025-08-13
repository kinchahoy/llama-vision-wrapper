"""
Example script to run multimodal inference and benchmarking using the llama_cpp_wrapper.
"""
import time
import sys
import argparse
import json
from huggingface_hub import hf_hub_download

from llama_cpp_wrapper import ResourceManager, gbl, timed_operation, all_timings

# Runtime parameters
N_CTX = 2048
N_BATCH = 512
MAX_NEW_TOKENS = 256

# Sampling parameters
TEMP = 0.8
TOP_K = 40
TOP_P = 0.95
REPEAT_PENALTY = 1.1

# Global benchmark results
benchmark_results = {
    "version": "cppyy",
    "model_loading_time": 0.0,
    "prompt_processing_time": 0.0,
    "image_processing_time": 0.0,
    "token_generation_rate": 0.0,
    "total_tokens_generated": 0,
    "final_output": "",
    "total_time": 0,
    "timestamp": "",
    "model_info": {},
    "parameters": {},
}


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
    parser.add_argument(
        "--n-gpu-layers",
        "-ngl",
        type=int,
        default=0,
        dest="n_gpu_layers",
        help="Number of layers to offload to GPU.",
    )
    parser.add_argument(
        "-t",
        type=int,
        default=8,
        dest="n_threads",
        help="Number of threads to use for computation.",
    )
    parser.add_argument(
        "--verbose-cpp",
        action="store_true",
        help="Enable verbose logging from the C++ backend.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmarking and save results to benchmark.json.",
    )
    parser.add_argument(
        "--save-embedding",
        type=str,
        default=None,
        help="Path to save the media embedding to a file.",
    )
    parser.add_argument(
        "--load-embedding",
        type=str,
        default=None,
        help="Path to load a media embedding from a file, ignoring the --image argument for encoding.",
    )
    args = parser.parse_args()

    try:
        print("--- Initializing ---")
        print("Threads set to :", args.n_threads)

        print("--- Downloading models from Hugging Face Hub ---")
        with timed_operation("Model download"):
            model_path = hf_hub_download(repo_id=args.repo_id, filename=args.model)
        with timed_operation("MMPROJ download"):
            mmproj_path = hf_hub_download(repo_id=args.repo_id, filename=args.mmproj)
        print(f"Model path: {model_path}")
        print(f"MMPROJ path: {mmproj_path}")

        if args.benchmark:
            # Store model info and parameters for benchmark
            benchmark_results["model_info"] = {
                "repo_id": args.repo_id,
                "model_file": args.model,
                "mmproj_file": args.mmproj,
                "model_path": model_path,
                "mmproj_path": mmproj_path,
            }
            benchmark_results["parameters"] = {
                "n_ctx": N_CTX,
                "n_batch": N_BATCH,
                "n_threads": args.n_threads,
                "n_gpu_layers": args.n_gpu_layers,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temp": TEMP,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "repeat_penalty": REPEAT_PENALTY,
            }
            benchmark_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # cppyy setup is handled by importing llama_cpp_wrapper

        # Initialize resources
        with ResourceManager(verbose=args.verbose_cpp) as rm:
            # Load model and create contexts
            start_model_load = time.time()
            model = rm.load_model(model_path, args.n_gpu_layers)
            ctx = rm.create_context(
                model, N_CTX, N_BATCH, args.n_threads, args.verbose_cpp
            )
            ctx_mtmd = rm.load_mtmd(
                mmproj_path,
                model,
                args.n_gpu_layers > 0,
                args.n_threads,
                args.verbose_cpp,
            )
            sampler = rm.create_sampler(
                model, TEMP, TOP_K, TOP_P, REPEAT_PENALTY, gbl.llama_n_ctx(ctx)
            )
            if args.benchmark:
                benchmark_results["model_loading_time"] = (
                    time.time() - start_model_load
                )

            # Load image
            print("Loading image...")
            bitmap, image_load_duration = rm.load_image(ctx_mtmd, args.image)
            print(
                f"Image loaded: {gbl.mtmd_bitmap_get_nx(bitmap)}x{gbl.mtmd_bitmap_get_ny(bitmap)}"
            )
            if args.benchmark:
                benchmark_results["image_processing_time"] = image_load_duration

            # Prepare and evaluate multimodal input
            print("Tokenizing multimodal input...")
            chunks = rm.tokenize_prompt(ctx_mtmd, args.prompt, bitmap)

            # Find and encode image chunk
            image_chunk = None
            for i in range(gbl.mtmd_input_chunks_size(chunks)):
                chunk = gbl.mtmd_input_chunks_get(chunks, i)
                if gbl.mtmd_input_chunk_get_type(chunk) in [
                    gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE,
                    gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO,
                ]:
                    image_chunk = chunk
                    break  # Assuming one media chunk for now

            if image_chunk:
                if args.load_embedding:
                    print(f"Loading media embedding from {args.load_embedding}...")
                    rm.load_encoded_chunk(
                        ctx_mtmd, image_chunk, args.load_embedding
                    )
                else:
                    print("Encoding media chunk...")
                    rm.encode_image_chunk(ctx_mtmd, image_chunk)

                if args.save_embedding:
                    print(f"Saving media embedding to {args.save_embedding}...")
                    rm.save_encoded_chunk(
                        ctx_mtmd, image_chunk, args.save_embedding
                    )

            print("Evaluating multimodal input...")
            n_past, prompt_eval_duration = rm.eval_chunks(
                ctx, ctx_mtmd, chunks, N_BATCH
            )
            if args.benchmark:
                benchmark_results["prompt_processing_time"] = prompt_eval_duration

            print(f"KV cache position (n_past): {n_past}")

            # Generate response
            print(f"\n--- Generating Response ({MAX_NEW_TOKENS} tokens max) ---")
            print(f"{args.prompt}", end="", flush=True)

            cpp_result, gen_timing_ctx = rm.generate(
                sampler, ctx, model, n_past, N_CTX, MAX_NEW_TOKENS
            )

            # Print results
            print(f"{cpp_result.generated_text}")

            # Store benchmark results
            if args.benchmark:
                benchmark_results[
                    "total_tokens_generated"
                ] = cpp_result.total_tokens_generated
                benchmark_results["final_output"] = str(cpp_result.generated_text)
                if gen_timing_ctx.duration > 0:
                    benchmark_results["token_generation_rate"] = (
                        cpp_result.total_tokens_generated / gen_timing_ctx.duration
                    )

            print(f"Final KV cache position (n_past): {cpp_result.final_n_past}")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)

    print("\n--- End of script ---")

    if args.benchmark:
        print("\nSummary of timing statistics:")
        print("=" * 50)
        for i, stat in enumerate(all_timings, 1):
            print(f"{i}. {stat}")
        print("=" * 50)
        print(f"Total operations: {len(all_timings)}")
        total_time = sum(stat.duration for stat in all_timings)
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
