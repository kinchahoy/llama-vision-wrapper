"""
A simple example to run multimodal inference using the llama_cpp_wrapper.
This script demonstrates the basic steps to load a model, process an image and prompt,
and generate a response.
"""
import sys
import argparse
import os
from huggingface_hub import hf_hub_download

# Adjust path to import from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llama_cpp_wrapper import ResourceManager, gbl

# Default runtime parameters
N_CTX = 2048
N_BATCH = 512
MAX_NEW_TOKENS = 256

# Default sampling parameters
TEMP = 0.8
TOP_K = 40
TOP_P = 0.95
REPEAT_PENALTY = 1.1

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="A simple example of multimodal generation with cppyy."
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
        default="USER: Describe this image.\n<__media__>\nASSISTANT:",
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
        "--temp",
        type=float,
        default=TEMP,
        help="Temperature for sampling.",
    )
    args = parser.parse_args()

    try:
        print("--- Initializing ---")
        print(f"Threads set to: {args.n_threads}")

        print("--- Downloading models from Hugging Face Hub ---")
        model_path = hf_hub_download(repo_id=args.repo_id, filename=args.model)
        mmproj_path = hf_hub_download(repo_id=args.repo_id, filename=args.mmproj)
        print(f"Model path: {model_path}")
        print(f"MMPROJ path: {mmproj_path}")

        # Initialize resources
        with ResourceManager(verbose=args.verbose_cpp) as rm:
            # Load model and create contexts
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
                model, args.temp, TOP_K, TOP_P, REPEAT_PENALTY, gbl.llama_n_ctx(ctx)
            )

            # Load image
            print("Loading image...")
            bitmap, _ = rm.load_image(ctx_mtmd, args.image)
            print(f"Image loaded: {gbl.mtmd_bitmap_get_nx(bitmap)}x{gbl.mtmd_bitmap_get_ny(bitmap)}")

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
                print("Encoding media chunk...")
                rm.encode_image_chunk(ctx_mtmd, image_chunk)

            print("Evaluating multimodal input...")
            n_past = 0
            n_chunks = gbl.mtmd_input_chunks_size(chunks)
            for i in range(n_chunks):
                chunk = gbl.mtmd_input_chunks_get(chunks, i)
                chunk_type = gbl.mtmd_input_chunk_get_type(chunk)
                is_last_chunk = i == n_chunks - 1

                if chunk_type == gbl.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    n_past = rm.eval_text_chunk(
                        ctx, chunk, n_past, N_BATCH, is_last_chunk
                    )
                elif chunk_type in [
                    gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE,
                    gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO,
                ]:
                    n_past = rm.decode_image_chunk(
                        ctx, ctx_mtmd, chunk, n_past, N_BATCH
                    )
                else:
                    raise RuntimeError(f"Unsupported chunk type: {chunk_type}")

            print(f"KV cache position (n_past): {n_past}")

            # Generate response
            print(f"\n--- Generating Response ({MAX_NEW_TOKENS} tokens max) ---")
            print(f"{args.prompt}", end="", flush=True)

            cpp_result, _ = rm.generate(
                sampler, ctx, model, n_past, N_CTX, MAX_NEW_TOKENS
            )

            # Print results
            print(f"{cpp_result.generated_text}")
            print(f"\nFinal KV cache position (n_past): {cpp_result.final_n_past}")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)

    print("\n--- End of script ---")

if __name__ == "__main__":
    main()
