# Simplified test for multimodal generation using mtmd library
# Assumes Gemma-3 model and a single image input.

import cppyy
import cppyy.numba_ext
import os
import time
import sys
import argparse  # Added for potential future CLI arguments
from numba import njit  # Added for numba acceleration

# --- 1. Configuration ---
# TODO: Consider using environment variables or argparse for these paths
LLAMA_CPP_SOURCE_DIRS = [
    "/Users/raistlin/code/llama-mtmd-py/llama.cpp/include",  # Core llama.h
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
# TODO: Replace with your actual paths
MODEL_PATH = "/Users/raistlin/models/gguf/gemma-3-4b-it-q4_0.gguf"
MMPROJ_PATH = (
    "/Users/raistlin/models/gguf/mmproj-model-f16-4B.gguf"  # Multimodal projector
)
IMAGE_PATH = "/Users/raistlin/code/llama-mtmd-py/test.jpg"  # Path to your test image

# --- Parameters ---
N_CTX = 2048  # Context size (adjust as needed)
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

# --- 2. cppyy Setup ---
try:
    print("--- cppyy Setup ---")
    print("Adding include paths:")
    for inc_path in LLAMA_CPP_SOURCE_DIRS:
        print(f"- {inc_path}")
        cppyy.add_include_path(inc_path)

    print("\nLoading libraries:")
    for lib_name in LIB_NAMES:
        # Determine the correct directory based on the library name
        if lib_name == "libgeneration_helper.dylib":
            lib_dir = HELPER_LIB_DIR
        else:
            lib_dir = LLAMA_CPP_LIBS_DIR

        lib_path = os.path.join(lib_dir, lib_name)

        if not os.path.exists(lib_path):
            print(f"Error: Library '{lib_name}' not found at expected path: {lib_path}")
            print("Please ensure LLAMA_CPP_LIBS_DIR and HELPER_LIB_DIR are correct,")
            print("and that both llama.cpp and the helper library were built.")
            sys.exit(1)

        print(f"- Loading: {lib_path}")
        cppyy.load_library(lib_path)
    print("Libraries loaded successfully.")

    # Include necessary headers
    print("Including C++ headers...")
    # Standard libraries often included implicitly, but good to be explicit if needed
    # cppyy.include("stdlib.h")
    # cppyy.include("stdio.h")
    # cppyy.include("string.h")
    cppyy.include("vector")
    cppyy.include("llama.h")
    cppyy.include("common.h")  # common helpers
    cppyy.include("sampling.h")  # common_sampler
    cppyy.include("mtmd.h")  # multimodal library
    cppyy.include("generation_helper.h")  # Our generation helper
    print("Headers included.")

except Exception as e:
    print(f"\n--- Error during cppyy setup ---")
    print(e)
    print("\nPossible issues:")
    print(
        "- Check LLAMA_CPP_SOURCE_DIRS: Ensure they point to the correct 'include' and 'common' directories of your llama.cpp source."
    )
    print(
        "- Check LLAMA_CPP_LIBS_DIR and HELPER_LIB_DIR: Ensure they point to the correct build output directories."
    )
    print(
        "- Build Configuration: Make sure llama.cpp was built as a *shared* library (e.g., cmake .. -DBUILD_SHARED_LIBS=ON)."
    )
    print(
        "- Header Dependencies: Ensure 'common.h' and 'sampling.h' exist and are accessible."
    )
    sys.exit(1)

# Access C++ functions/structs via cppyy.gbl
gbl = cppyy.gbl

# --- 3. Initialization ---
print("\n--- Initializing llama.cpp backend ---")
# llama_backend_init now takes args for numa config, but defaults are often fine
# gbl.llama_backend_init(False) # False = use NUMA if available
gbl.llama_backend_init()  # Simpler call, uses defaults
print("Backend initialized.")

# --- 4. Load Model and Context ---
print("\n--- Loading Model ---")
model_params = gbl.llama_model_default_params()
model_params.n_gpu_layers = N_GPU_LAYERS

print(f"Loading model from: {MODEL_PATH}")
model = gbl.llama_load_model_from_file(MODEL_PATH.encode("utf-8"), model_params)
if not model:
    print(f"Error: Failed to load model from {MODEL_PATH}")
    gbl.llama_backend_free()
    sys.exit(1)
print(f"Text model loaded successfully. GPU layers = {model_params.n_gpu_layers}")

# --- 4b. Initialize Multimodal Context ---
print("\n--- Initializing Multimodal Context ---")
mtmd_params = gbl.mtmd_context_params()
mtmd_params.use_gpu = (
    N_GPU_LAYERS > 0
)  # Allow MMPROJ to use GPU if LLaMA layers are offloaded
mtmd_params.n_threads = N_THREADS
mtmd_params.verbosity = (
    gbl.GGML_LOG_LEVEL_INFO
)  # Or GGML_LOG_LEVEL_DEBUG for more detail

print(f"Loading multimodal projector from: {MMPROJ_PATH} (GPU: {mtmd_params.use_gpu})")
# Use unique_ptr for automatic cleanup
ctx_mtmd = gbl.mtmd_init_from_file(MMPROJ_PATH.encode("utf-8"), model, mtmd_params)
if not ctx_mtmd:
    print(f"Error: Failed to load multimodal projector from {MMPROJ_PATH}")
    gbl.llama_free_model(model)
    gbl.llama_backend_free()
    sys.exit(1)
print("Multimodal context initialized.")


# --- 4c. Create LLaMA Context ---
print("\n--- Creating LLaMA Context ---")
ctx_params = gbl.llama_context_default_params()
ctx_params.n_ctx = N_CTX
# Use a reasonable batch size for decoding, mtmd_helper_eval handles internal batching
ctx_params.n_batch = N_BATCH
ctx_params.n_threads = N_THREADS
ctx_params.n_threads_batch = N_THREADS  # Often same as n_threads
# Add logging callback to see backend messages
ctx_params.log_level = gbl.GGML_LOG_LEVEL_INFO  # Or GGML_LOG_LEVEL_DEBUG
# ctx_params.log_callback = gbl.llama_log_callback(your_python_log_function) # Optional: Define a Python callback

# Check if context size exceeds model's trained context
model_n_ctx_train = gbl.llama_n_ctx_train(model)
if N_CTX > model_n_ctx_train:
    print(
        f"Warning: N_CTX ({N_CTX}) > model training context ({model_n_ctx_train}). May lead to degraded performance."
    )
    # ctx_params.n_ctx = model_n_ctx_train # Option: Clamp to model max

ctx = gbl.llama_new_context_with_model(model, ctx_params)
if not ctx:
    print("Error: Failed to create context")
    gbl.llama_free_model(model)
    gbl.llama_backend_free()
    sys.exit(1)
print(
    f"LLaMA context created (n_ctx = {gbl.llama_n_ctx(ctx)}, n_batch = {gbl.llama_n_batch(ctx)}, n_threads = {N_THREADS})"
)


# --- 5. Load Image ---
print("\n--- Loading Image ---")
bitmap = gbl.mtmd_bitmap()
ret = gbl.mtmd_helper_bitmap_init_from_file(IMAGE_PATH.encode("utf-8"), bitmap)
if ret != 0:
    print(
        f"Error: Failed to load image from {IMAGE_PATH} (mtmd_helper_bitmap_init_from_file returned {ret})"
    )
    gbl.mtmd_free(ctx_mtmd)
    gbl.llama_free(ctx)
    gbl.llama_free_model(model)
    gbl.llama_backend_free()
    sys.exit(1)
print(f"Image loaded: {bitmap.nx}x{bitmap.ny}")

# --- 6. Prepare and Evaluate Multimodal Input ---
print("\n--- Preparing and Evaluating Multimodal Input ---")
start_time = time.time()

# Prepare input text structure
input_text = gbl.mtmd_input_text()
input_text.text = PROMPT
input_text.add_special = True  # Add BOS if needed by model/template
input_text.parse_special = True  # Parse special tokens like <__image__>

# Prepare image bitmap vector (only one image in this case)
bitmaps_vec = gbl.std.vector[gbl.mtmd_bitmap]()
bitmaps_vec.push_back(bitmap)  # Add the loaded bitmap

# Prepare output chunk vector
chunks = gbl.mtmd_input_chunks()

# Tokenize text and image marker
print("Tokenizing multimodal input...")
ret = gbl.mtmd_tokenize(ctx_mtmd, chunks, input_text, bitmaps_vec)
if ret != 0:
    print(f"Error: Failed to tokenize multimodal input (mtmd_tokenize returned {ret})")
    # Add cleanup
    sys.exit(1)
print(f"Input tokenized into {len(chunks)} chunks.")

# Evaluate the chunks (text and image)
print("Evaluating multimodal input chunks...")
n_past = 0  # Initial position in KV cache
seq_id = gbl.llama_seq_id(0)  # Simple case: sequence ID 0

# mtmd_helper_eval handles decoding text and image chunks appropriately
ret = gbl.mtmd_helper_eval(ctx_mtmd, ctx, chunks, n_past, seq_id, N_BATCH)
if ret != 0:
    print(
        f"Error: Failed to evaluate multimodal input (mtmd_helper_eval returned {ret})"
    )
    # Add cleanup
    sys.exit(1)

# Update n_past based on the number of tokens processed by mtmd_helper_eval
n_past += gbl.mtmd_helper_get_n_tokens(chunks)

end_time = time.time()
print(f"Multimodal input evaluation complete in {end_time - start_time:.2f} s.")
print(f"KV cache position (n_past): {n_past}")

# We don't need the bitmap data anymore after mtmd_tokenize/eval
# bitmap.data.clear() # Optional: free image buffer memory if large

# --- 7. Setup Sampler ---
print("\n--- Setting up Sampler ---")
sampling_params = gbl.common_params_sampling()  # Instantiate the struct (Correct)
# Apply custom settings
sampling_params.temp = TEMP
sampling_params.top_k = TOP_K
sampling_params.top_p = TOP_P
sampling_params.penalty_repeat = REPEAT_PENALTY
# Ensure the context size is known to the sampler for repeat penalty
sampling_params.penalty_last_n = gbl.llama_n_ctx(
    ctx
)  # Use full context for penalty history
sampling_params.grammar = ""  # Explicitly ensure no grammar is used

# --- Debug: Print sampling params before init ---
print("--- Debug: Sampling Parameters before common_sampler_init ---")
print(f"  temp: {sampling_params.temp}")
print(f"  top_k: {sampling_params.top_k}")
print(f"  top_p: {sampling_params.top_p}")
print(f"  penalty_repeat: {sampling_params.penalty_repeat}")
print(f"  penalty_last_n: {sampling_params.penalty_last_n}")
print(f"  mirostat: {sampling_params.mirostat}")
print(f"  grammar empty: {not bool(sampling_params.grammar)}")
print("----------------------------------------------------------")

# Initialize the sampler
try:
    # --- Debug: Check model pointer before init ---
    if not model:
        print("Error: 'model' pointer is null just before common_sampler_init!")
        raise RuntimeError("'model' pointer became null unexpectedly")
    print("Debug: 'model' pointer seems valid before common_sampler_init.")

    # Use the factory function common_sampler_init instead of a constructor
    print("Debug: Calling common_sampler_init...")
    sampler = gbl.common_sampler_init(model, sampling_params)
    print(f"Debug: common_sampler_init returned: {sampler}")  # See if it's None/nullptr
    if not sampler:
        raise RuntimeError("common_sampler_init returned nullptr")
    print("common_sampler initialized successfully.")
except Exception as e:
    print(f"Error initializing common_sampler: {e}")
    print("Check the constructor signature in your llama.cpp common/sampling.h")
    # ... (cleanup)


# --- IMPORTS ---
# (Keep all your existing imports: cppyy, os, time, sys, numba, etc.)
import cppyy
import cppyy.numba_ext  # Make sure this is imported BEFORE the @njit function
import os
import time
import sys
import argparse
from numba import (
    njit,
    types,
)  # Import types for potential signature specification (advanced)
# [...] Your existing configuration and setup code up to section 8


# --- 8. Generation Loop ---

# Define the Python callback function
# It must match the signature: void (*PythonCallbackFunc)(const char* chunk, int n_tokens_in_chunk)
# cppyy handles the marshalling of bytes -> Python string (usually utf-8)
# We keep track of the full text and token count outside the callback
full_generated_text = ""
total_generated_tokens = 0
def generation_callback(chunk_bytes, n_tokens_in_chunk):
    """Callback function called by C++ with generated text chunks."""
    global full_generated_text, total_generated_tokens
    # Decode the byte string received from C++
    chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
    print(f"{chunk_str}", end="", flush=True) # Print intermediate results
    full_generated_text += chunk_str
    total_generated_tokens += n_tokens_in_chunk

print("\n--- Generating Response ---")
print(f"{PROMPT}", end="", flush=True)

start_time_gen = time.time()

# Define the sequence ID (usually 0 for simple generation)
seq_id = gbl.llama_seq_id(0)
# Create the C++ vector containing the sequence ID(s)
seq_id_vec = gbl.std.vector[gbl.llama_seq_id]([seq_id])

# Store initial n_past
initial_n_past = n_past

# --- Call the C++ generation function with the callback ---
CALLBACK_TOKEN_THRESHOLD = 50 # How often to call back (in tokens)
try:
    print(f"\nCalling C++ generation function (callback threshold: {CALLBACK_TOKEN_THRESHOLD} tokens)...")
    # Pass the Python callback function directly to cppyy
    cpp_result = gbl.generate_tokens_cpp(
        sampler,
        ctx,
        model,
        initial_n_past,
        N_CTX,
        MAX_NEW_TOKENS,
        seq_id_vec,           # Pass the vector directly (by value)
        generation_callback,  # Pass the Python callback function
        CALLBACK_TOKEN_THRESHOLD
    )
    # The C++ function now blocks until generation is complete,
    # but calls the callback periodically.
    print("\nC++ function finished.") # Newline after streaming output

    # --- Process the final result from C++ ---
    # The text itself was handled by the callback
    n_past = cpp_result.final_n_past # Update final n_past
    # Verify token count if needed (should match total_generated_tokens)
    if cpp_result.total_tokens_generated != total_generated_tokens:
         print(f"\nWarning: C++ reported {cpp_result.total_tokens_generated} tokens, callback accumulated {total_generated_tokens}")
    # Use the count from the callback for consistency
    generated_token_count = total_generated_tokens


except Exception as e:
    print(f"\n--- C++ Generation Failed or Callback Error ---")
    print(f"Error: {e}")
    # Add cleanup if necessary
    sys.exit(1)


end_time_gen = time.time()
print(f"\n--- Generation Complete ---")
# Use the total count accumulated by the callback
print(
    f"Generated {total_generated_tokens} tokens in {end_time_gen - start_time_gen:.2f} s."
)
# print(f"Full response:\n{PROMPT}{full_generated_text}") # Uncomment if needed

# --- 9. Cleanup ---
print("\n--- Cleaning up ---")
# Free sampler if it was initialized
# Check if 'sampler' exists and is not None (or equivalent null pointer for cppyy)
if 'sampler' in locals() and sampler:
    print("Freeing sampler...")
    gbl.common_sampler_free(sampler)

# Free bitmap if it was initialized
# Check if 'bitmap' exists and potentially if its data is valid (optional)
if 'bitmap' in locals() and bitmap:
     print("Freeing bitmap...")
     gbl.mtmd_bitmap_free(cppyy.addressof(bitmap)) # Pass by pointer

# No need to free gen_batch anymore
if ctx:
    print("Freeing LLaMA context...")
    gbl.llama_free(ctx)
if ctx_mtmd:
    print("Freeing multimodal context...")
    gbl.mtmd_free(ctx_mtmd)  # Free multimodal context
if model:
    print("Freeing LLaMA model...")
    gbl.llama_free_model(model)

print("Freeing LLaMA backend...")
gbl.llama_backend_free()
print("Resources freed.")
