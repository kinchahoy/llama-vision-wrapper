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


# --- Context Managers for C Resource Cleanup ---
class LlamaBackendManager:
    """Manages llama_backend_init() and llama_backend_free()."""
    def __init__(self, verbose=True):
        self.verbose = verbose
        if self.verbose: print("Initializing llama backend...")

    def __enter__(self):
        cppyy.gbl.llama_backend_init()
        if self.verbose: print("Llama backend initialized.")
        return self # Or return cppyy.gbl if preferred

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose: print("Freeing llama backend...")
        cppyy.gbl.llama_backend_free()
        if self.verbose: print("Llama backend freed.")
        # Return False to propagate exceptions, True to suppress
        return False

class LlamaModelManager:
    """Manages llama_load_model_from_file() and llama_free_model()."""
    def __init__(self, model_path, model_params, verbose=True):
        self.model_path = model_path
        self.model_params = model_params
        self.verbose = verbose
        self.model_ptr = None
        if self.verbose: print(f"Loading model from {model_path}...")

    def __enter__(self):
        self.model_ptr = cppyy.gbl.llama_load_model_from_file(
            self.model_path.encode("utf-8"), self.model_params
        )
        if not self.model_ptr:
            raise RuntimeError(f"Failed to load model from {self.model_path}")
        if self.verbose: print(f"Model loaded (GPU layers: {self.model_params.n_gpu_layers})")
        return self.model_ptr # Return the raw pointer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model_ptr:
            if self.verbose: print("Freeing model...")
            cppyy.gbl.llama_free_model(self.model_ptr)
            if self.verbose: print("Model freed.")
        return False

class MtmdContextManager:
    """Manages mtmd_init_from_file() and mtmd_free()."""
    def __init__(self, mmproj_path, model_ptr, mtmd_params, verbose=True):
        self.mmproj_path = mmproj_path
        self.model_ptr = model_ptr
        self.mtmd_params = mtmd_params
        self.verbose = verbose
        self.ctx_mtmd_ptr = None
        if self.verbose: print(f"Loading multimodal context from {mmproj_path}...")

    def __enter__(self):
        self.ctx_mtmd_ptr = cppyy.gbl.mtmd_init_from_file(
            self.mmproj_path.encode("utf-8"), self.model_ptr, self.mtmd_params
        )
        if not self.ctx_mtmd_ptr:
            raise RuntimeError(f"Failed to load multimodal projector from {self.mmproj_path}")
        if self.verbose: print(f"Multimodal context loaded (GPU: {self.mtmd_params.use_gpu})")
        return self.ctx_mtmd_ptr # Return the raw pointer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ctx_mtmd_ptr:
            if self.verbose: print("Freeing multimodal context...")
            cppyy.gbl.mtmd_free(self.ctx_mtmd_ptr)
            if self.verbose: print("Multimodal context freed.")
        return False

class LlamaContextManager:
    """Manages llama_new_context_with_model() and llama_free()."""
    def __init__(self, model_ptr, ctx_params, verbose=True):
        self.model_ptr = model_ptr
        self.ctx_params = ctx_params
        self.verbose = verbose
        self.ctx_ptr = None
        if self.verbose: print("Creating LLaMA context...")

    def __enter__(self):
        self.ctx_ptr = cppyy.gbl.llama_new_context_with_model(self.model_ptr, self.ctx_params)
        if not self.ctx_ptr:
            raise RuntimeError("Failed to create LLaMA context")
        if self.verbose: print(f"LLaMA context created (n_ctx: {cppyy.gbl.llama_n_ctx(self.ctx_ptr)})")
        return self.ctx_ptr # Return the raw pointer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ctx_ptr:
            if self.verbose: print("Freeing LLaMA context...")
            cppyy.gbl.llama_free(self.ctx_ptr)
            if self.verbose: print("LLaMA context freed.")
        return False

class SamplerManager:
    """Manages common_sampler_init() and common_sampler_free()."""
    def __init__(self, model_ptr, sampling_params, verbose=True):
        self.model_ptr = model_ptr
        self.sampling_params = sampling_params
        self.verbose = verbose
        self.sampler_ptr = None
        if self.verbose: print("Initializing sampler...")

    def __enter__(self):
        self.sampler_ptr = cppyy.gbl.common_sampler_init(self.model_ptr, self.sampling_params)
        if not self.sampler_ptr:
            raise RuntimeError("Failed to initialize common_sampler")
        if self.verbose: print("Sampler initialized.")
        return self.sampler_ptr # Return the raw pointer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sampler_ptr:
            if self.verbose: print("Freeing sampler...")
            cppyy.gbl.common_sampler_free(self.sampler_ptr)
            if self.verbose: print("Sampler freed.")
        return False
# --- [End Context Manager Classes] ---


# --- 2. cppyy Setup & Initialization ---
# Define variables for cleanup in finally block # <-- This comment is now obsolete
gbl = None
# model = None # <-- No longer needed here, managed by context manager
ctx_mtmd = None
ctx = None
sampler = None
bitmap = None  # Keep bitmap ref for potential cleanup if needed later

# Define generation state variables
full_generated_text = ""
total_generated_tokens = 0

# Define callback function
def generation_callback(chunk_bytes, n_tokens_in_chunk):
    """Callback function called by C++ with generated text chunks."""
    global full_generated_text, total_generated_tokens
    chunk_str = chunk_bytes
    print(f"{chunk_str}", end="", flush=True)
    full_generated_text += chunk_str
    total_generated_tokens += n_tokens_in_chunk

# --- Main Execution Logic ---
try:
    print("--- Initializing ---")
    # Add include paths
    for inc_path in LLAMA_CPP_SOURCE_DIRS:
        cppyy.add_include_path(inc_path)

    # Load libraries
    print("Loading libraries...")
    for lib_name in LIB_NAMES:
        lib_dir = (
            HELPER_LIB_DIR
            if lib_name == "libgeneration_helper.dylib"
            else LLAMA_CPP_LIBS_DIR
        )
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
    gbl = cppyy.gbl # Assign gbl after includes

    # --- Use Context Managers for Resource Management ---
    # The 'verbose=False' argument can be added to managers to reduce print output
    with LlamaBackendManager(verbose=True):
        # Model Params
        model_params = gbl.llama_model_default_params()
        model_params.n_gpu_layers = N_GPU_LAYERS

        with LlamaModelManager(MODEL_PATH, model_params) as model:
            # Multimodal Context Params
            mtmd_params = gbl.mtmd_context_params()
            mtmd_params.use_gpu = N_GPU_LAYERS > 0
            mtmd_params.n_threads = N_THREADS
            mtmd_params.verbosity = gbl.GGML_LOG_LEVEL_ERROR

            with MtmdContextManager(MMPROJ_PATH, model, mtmd_params) as ctx_mtmd:
                # LLaMA Context Params
                ctx_params = gbl.llama_context_default_params()
                ctx_params.n_ctx = N_CTX
                ctx_params.n_batch = N_BATCH
                ctx_params.n_threads = N_THREADS
                ctx_params.n_threads_batch = N_THREADS
                ctx_params.log_level = gbl.GGML_LOG_LEVEL_ERROR
                model_n_ctx_train = gbl.llama_n_ctx_train(model)
                if N_CTX > model_n_ctx_train:
                    print(f"Warning: N_CTX ({N_CTX}) > model training context ({model_n_ctx_train}).")

                with LlamaContextManager(model, ctx_params) as ctx:
                    # Sampling Params
                    sampling_params = gbl.common_params_sampling()
                    sampling_params.temp = TEMP
                    sampling_params.top_k = TOP_K
                    sampling_params.top_p = TOP_P
                    sampling_params.penalty_repeat = REPEAT_PENALTY
                    sampling_params.penalty_last_n = gbl.llama_n_ctx(ctx) # Use actual context size
                    sampling_params.grammar = ""

                    with SamplerManager(model, sampling_params) as sampler:
                        # --- All resources are now acquired and managed ---

                        # --- 4. Load Image ---
                        print("Loading image...")
                        bitmap = gbl.mtmd_bitmap() # Create bitmap object
                        ret = gbl.mtmd_helper_bitmap_init_from_file(IMAGE_PATH.encode("utf-8"), bitmap)
                        if ret != 0:
                            raise RuntimeError(f"Failed to load image {IMAGE_PATH} (code: {ret})")
                        print(f"Image loaded: {bitmap.nx}x{bitmap.ny}")
                        # No need to manage bitmap with 'with', it's used and implicitly handled

                        # --- 5. Prepare and Evaluate Multimodal Input ---
                        print("Evaluating multimodal input...")
                        eval_start_time = time.time()

                        input_text = gbl.mtmd_input_text()
                        input_text.text = PROMPT
                        input_text.add_special = True
                        input_text.parse_special = True

                        bitmaps_vec = gbl.std.vector[gbl.mtmd_bitmap]()
                        bitmaps_vec.push_back(bitmap) # Pass the loaded bitmap

                        chunks = gbl.mtmd_input_chunks()

                        ret = gbl.mtmd_tokenize(ctx_mtmd, chunks, input_text, bitmaps_vec)
                        if ret != 0:
                            raise RuntimeError(f"Failed mtmd_tokenize (code: {ret})")

                        n_past = 0
                        seq_id = gbl.llama_seq_id(0)
                        ret = gbl.mtmd_helper_eval(ctx_mtmd, ctx, chunks, n_past, seq_id, N_BATCH)
                        if ret != 0:
                            raise RuntimeError(f"Failed mtmd_helper_eval (code: {ret})")

                        prompt_tokens = gbl.mtmd_helper_get_n_tokens(chunks)
                        n_past += prompt_tokens
                        eval_end_time = time.time()
                        eval_duration = eval_end_time - eval_start_time
                        eval_speed = prompt_tokens / eval_duration if eval_duration > 0 else float("inf")
                        print(f"Input evaluated ({prompt_tokens} tokens) in {eval_duration:.2f} s ({eval_speed:.2f} tokens/s)")
                        print(f"KV cache position (n_past): {n_past}")

                        # --- 7. Generation Loop ---
                        print(f"\n--- Generating Response ({MAX_NEW_TOKENS} tokens max) ---")
                        print(f"{PROMPT}", end="", flush=True)

                        gen_start_time = time.time()
                        seq_id_vec = gbl.std.vector[gbl.llama_seq_id]([gbl.llama_seq_id(0)])
                        initial_n_past = n_past
                        CALLBACK_TOKEN_THRESHOLD = 50

                        cpp_result = gbl.generate_tokens_cpp(
                            sampler, ctx, model, initial_n_past, N_CTX,
                            MAX_NEW_TOKENS, seq_id_vec, generation_callback,
                            CALLBACK_TOKEN_THRESHOLD
                        )
                        print("\n--- Generation Complete ---")

                        gen_end_time = time.time()
                        gen_duration = gen_end_time - gen_start_time
                        gen_speed = total_generated_tokens / gen_duration if gen_duration > 0 else float("inf")
                        n_past = cpp_result.final_n_past
                        if cpp_result.total_tokens_generated != total_generated_tokens:
                            print(f"\nWarning: C++ reported {cpp_result.total_tokens_generated} tokens, callback got {total_generated_tokens}")

                        print(f"Generated {total_generated_tokens} tokens in {gen_duration:.2f} s ({gen_speed:.2f} tokens/s)")
                        print(f"Final KV cache position (n_past): {n_past}")

                    # Sampler automatically freed here by SamplerManager.__exit__
                # LlamaContext automatically freed here by LlamaContextManager.__exit__
            # MtmdContext automatically freed here by MtmdContextManager.__exit__
        # Model automatically freed here by LlamaModelManager.__exit__
    # Backend automatically freed here by LlamaBackendManager.__exit__

    print("\n--- Resource cleanup automatically handled by context managers ---")

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"{type(e).__name__}: {e}")
    # Consider more specific error handling or logging
    sys.exit(1)

# NO finally block needed for resource cleanup anymore!

print("\n--- End of script ---")
