# Simplified test for multimodal generation using mtmd library
# Assumes Gemma-3 model and a single image input.

import cppyy
import os
import time
import sys
# Removed cppyy.numba_ext, argparse, numba - not used in simplified version

# --- 1. Configuration ---
LLAMA_CPP_SOURCE_DIRS = [
    "/Users/raistlin/code/llama-mtmd-py/llama.cpp/include",
    "/Users/raistlin/code/llama-mtmd-py/llama.cpp/ggml/include",  # Core ggml headers
    "/Users/raistlin/code/llama-mtmd-py/llama.cpp/common",  # Common headers (IMPORTANT for sampling)
    "/Users/raistlin/code/llama-mtmd-py/llama.cpp/examples/llava",  # For mtmd.h
]
# Define the two build directories
LLAMA_CPP_LIBS_DIR = (
    "/Users/raistlin/code/llama-mtmd-py/llama.cpp/build/bin"  # Main llama.cpp libs
)
HELPER_LIB_DIR = "/Users/raistlin/code/llama-mtmd-py/build"  # Custom helper lib

LIB_NAMES = [
    # Libraries from llama.cpp build
    "libggml-base.dylib",
    "libggml-blas.dylib",
    "libggml-cpu.dylib",
    "libggml-metal.dylib",
    "libggml.dylib",
    "libllama.dylib",
    #   "libcommon.dylib",
    "libllava_shared.dylib",
    "libmtmd_shared.dylib",
    "libllama.dylib",
    "libllava_shared.dylib",  # Contains clip dependencies for mtmd
    "libmtmd_shared.dylib",
    # Our custom helper library (will be loaded from HELPER_LIB_DIR)
    "libgeneration_helper.dylib",
]

# --- Model & Image Paths ---
MODEL_PATH = "/Users/raistlin/models/gguf/gemma-3-4b-it-q4_0.gguf"
MMPROJ_PATH = "/Users/raistlin/models/gguf/mmproj-model-f16-4B.gguf"
IMAGE_PATH = "/Users/raistlin/code/llama-mtmd-py/test.jpg"

# --- Parameters ---
N_CTX = 2048
N_THREADS = 8  # Threads for computation
N_GPU_LAYERS = 0  # GPU offload (0 = CPU only)
N_BATCH = 512  # Batch size for prompt processing

# Sampling Parameters
TEMP = 0.8
TOP_K = 40
TOP_P = 0.95
REPEAT_PENALTY = 1.1
# Consider adding mirostat options if desired:
# SAMPLING_PARAMS.mirostat = 0 # 0 = disabled, 1 = Mirostat V1, 2 = Mirostat V2
# SAMPLING_PARAMS.mirostat_tau = 5.0
# SAMPLING_PARAMS.mirostat_eta = 0.1

# --- Prompt ---
# Gemma-3 format: User message, image marker, Assistant prompt start
# The image marker will be replaced by mtmd_tokenize
PROMPT = "USER: Describe this image.\n<__image__>\nASSISTANT:"

# --- Generation Limits ---
MAX_NEW_TOKENS = 256

# --- 2. cppyy Setup & Initialization ---
# Define variables for cleanup in finally block
gbl = None
model = None
ctx_mtmd = None
ctx = None
sampler = None
bitmap = None # Keep bitmap ref for potential cleanup if needed later

# Define generation state variables in the main scope for the callback
full_generated_text = ""
total_generated_tokens = 0

try:
    print("--- Initializing ---")
    # Add include paths
    for inc_path in LLAMA_CPP_SOURCE_DIRS:
        cppyy.add_include_path(inc_path)

    # Load libraries
    print("Loading libraries...")
    for lib_name in LIB_NAMES:
        lib_dir = HELPER_LIB_DIR if lib_name == "libgeneration_helper.dylib" else LLAMA_CPP_LIBS_DIR
        lib_path = os.path.join(lib_dir, lib_name)
        if not os.path.exists(lib_path):
             raise FileNotFoundError(f"Library '{lib_name}' not found at {lib_path}")
        cppyy.load_library(lib_path)

    # Include headers
    print("Including C++ headers...")
    cppyy.include("vector")
    cppyy.include("llama.h")
    cppyy.include("common.h")
    cppyy.include("sampling.h")
    cppyy.include("mtmd.h")
    cppyy.include("generation_helper.h")
    gbl = cppyy.gbl # Assign gbl only after includes are successful

    # Init backend
    print("Initializing llama backend...")
    gbl.llama_backend_init()

    # --- 3. Load Model & Contexts ---
    print("Loading models and contexts...")
    # Model
    model_params = gbl.llama_model_default_params()
    model_params.n_gpu_layers = N_GPU_LAYERS
    model = gbl.llama_load_model_from_file(MODEL_PATH.encode("utf-8"), model_params)
    if not model: raise RuntimeError(f"Failed to load model from {MODEL_PATH}")
    print(f"Text model loaded (GPU layers: {model_params.n_gpu_layers})")

    # Multimodal Context
    mtmd_params = gbl.mtmd_context_params()
    mtmd_params.use_gpu = N_GPU_LAYERS > 0
    mtmd_params.n_threads = N_THREADS
    mtmd_params.verbosity = gbl.GGML_LOG_LEVEL_ERROR # Less verbose
    ctx_mtmd = gbl.mtmd_init_from_file(MMPROJ_PATH.encode("utf-8"), model, mtmd_params)
    if not ctx_mtmd: raise RuntimeError(f"Failed to load multimodal projector from {MMPROJ_PATH}")
    print(f"Multimodal context loaded (GPU: {mtmd_params.use_gpu})")

    # LLaMA Context
    ctx_params = gbl.llama_context_default_params()
    ctx_params.n_ctx = N_CTX
    ctx_params.n_batch = N_BATCH
    ctx_params.n_threads = N_THREADS
    ctx_params.n_threads_batch = N_THREADS
    ctx_params.log_level = gbl.GGML_LOG_LEVEL_ERROR # Less verbose
    model_n_ctx_train = gbl.llama_n_ctx_train(model)
    if N_CTX > model_n_ctx_train:
        print(f"Warning: N_CTX ({N_CTX}) > model training context ({model_n_ctx_train}).")
    ctx = gbl.llama_new_context_with_model(model, ctx_params)
    if not ctx: raise RuntimeError("Failed to create LLaMA context")
    print(f"LLaMA context created (n_ctx: {gbl.llama_n_ctx(ctx)})")

    # --- 4. Load Image ---
    print("Loading image...")
    bitmap = gbl.mtmd_bitmap()
    ret = gbl.mtmd_helper_bitmap_init_from_file(IMAGE_PATH.encode("utf-8"), bitmap)
    if ret != 0: raise RuntimeError(f"Failed to load image {IMAGE_PATH} (code: {ret})")
    print(f"Image loaded: {bitmap.nx}x{bitmap.ny}")

    # --- 5. Prepare and Evaluate Multimodal Input ---
    print("Evaluating multimodal input...")
    eval_start_time = time.time()

    input_text = gbl.mtmd_input_text()
    input_text.text = PROMPT
    input_text.add_special = True
    input_text.parse_special = True

    bitmaps_vec = gbl.std.vector[gbl.mtmd_bitmap]()
    bitmaps_vec.push_back(bitmap)

    chunks = gbl.mtmd_input_chunks()

    ret = gbl.mtmd_tokenize(ctx_mtmd, chunks, input_text, bitmaps_vec)
    if ret != 0: raise RuntimeError(f"Failed mtmd_tokenize (code: {ret})")

    n_past = 0
    seq_id = gbl.llama_seq_id(0)
    ret = gbl.mtmd_helper_eval(ctx_mtmd, ctx, chunks, n_past, seq_id, N_BATCH)
    if ret != 0: raise RuntimeError(f"Failed mtmd_helper_eval (code: {ret})")

    prompt_tokens = gbl.mtmd_helper_get_n_tokens(chunks)
    n_past += prompt_tokens
    eval_end_time = time.time()
    eval_duration = eval_end_time - eval_start_time
    eval_speed = prompt_tokens / eval_duration if eval_duration > 0 else float('inf')
    print(f"Input evaluated ({prompt_tokens} tokens) in {eval_duration:.2f} s ({eval_speed:.2f} tokens/s)")
    print(f"KV cache position (n_past): {n_past}")

    # --- 6. Setup Sampler ---
    print("Setting up sampler...")
    sampling_params = gbl.common_params_sampling()
    sampling_params.temp = TEMP
    sampling_params.top_k = TOP_K
    sampling_params.top_p = TOP_P
    sampling_params.penalty_repeat = REPEAT_PENALTY
    sampling_params.penalty_last_n = gbl.llama_n_ctx(ctx)
    sampling_params.grammar = "" # No grammar

    sampler = gbl.common_sampler_init(model, sampling_params)
    if not sampler: raise RuntimeError("Failed to initialize common_sampler")
    print("Sampler initialized.")

    # --- 7. Generation Loop ---
    # Variables are now defined outside the try block
    def generation_callback(chunk_bytes, n_tokens_in_chunk):
        """Callback function called by C++ with generated text chunks."""
        global full_generated_text, total_generated_tokens # Use global for module-level variables
        chunk_str = chunk_bytes # cppyy handles conversion
        print(f"{chunk_str}", end="", flush=True)
        full_generated_text += chunk_str
        total_generated_tokens += n_tokens_in_chunk

    print(f"\n--- Generating Response ({MAX_NEW_TOKENS} tokens max) ---")
    print(f"{PROMPT}", end="", flush=True)

    gen_start_time = time.time()
    seq_id_vec = gbl.std.vector[gbl.llama_seq_id]([gbl.llama_seq_id(0)])
    initial_n_past = n_past
    CALLBACK_TOKEN_THRESHOLD = 50 # How often to call back

    cpp_result = gbl.generate_tokens_cpp(
        sampler, ctx, model, initial_n_past, N_CTX, MAX_NEW_TOKENS,
        seq_id_vec, generation_callback, CALLBACK_TOKEN_THRESHOLD
    )
    print("\n--- Generation Complete ---") # Newline after streaming

    gen_end_time = time.time()
    gen_duration = gen_end_time - gen_start_time
    # Use token count from callback for performance calculation
    gen_speed = total_generated_tokens / gen_duration if gen_duration > 0 else float('inf')

    # Update n_past from C++ result
    n_past = cpp_result.final_n_past
    # Verify token count consistency (optional)
    if cpp_result.total_tokens_generated != total_generated_tokens:
         print(f"\nWarning: C++ reported {cpp_result.total_tokens_generated} tokens, callback got {total_generated_tokens}")

    print(f"Generated {total_generated_tokens} tokens in {gen_duration:.2f} s ({gen_speed:.2f} tokens/s)")
    print(f"Final KV cache position (n_past): {n_past}")
    # print(f"Full response:\n{PROMPT}{full_generated_text}") # Uncomment to see full text at end

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"{type(e).__name__}: {e}")
    # Potentially add more specific error handling or logging here
    sys.exit(1) # Exit after printing error

finally:
    # --- 8. Cleanup ---
    print("\n--- Cleaning up ---")
    if sampler:
        print("Freeing sampler...")
        gbl.common_sampler_free(sampler)
    # Bitmap cleanup is likely handled by mtmd library when ctx_mtmd is freed,
    # or potentially when bitmaps_vec goes out of scope if it manages ownership.
    # Explicit free was removed previously to avoid double-free. Keep commented.
    # if bitmap:
    #     print("Freeing bitmap...")
    #     gbl.mtmd_bitmap_free(cppyy.addressof(bitmap))
    if ctx:
        print("Freeing LLaMA context...")
        gbl.llama_free(ctx)
    if ctx_mtmd:
        print("Freeing multimodal context...")
        gbl.mtmd_free(ctx_mtmd)
    if model:
        print("Freeing LLaMA model...")
        gbl.llama_free_model(model)
    if gbl: # Check if gbl was successfully assigned (headers included)
        print("Freeing LLaMA backend...")
        gbl.llama_backend_free()
    print("Resources freed.")
