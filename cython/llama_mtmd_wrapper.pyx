# distutils: language = c++
# cython: language_level=3

"""
Cython wrapper for llama.cpp and mtmd functionality.
This module provides Python bindings for the C++ code.
"""

from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport int32_t, uint8_t, uint32_t
from libc.stdlib cimport malloc, free
from cpython.ref cimport PyObject
from libc.stddef cimport size_t

# External C++ declarations
cdef extern from "llama.h":
    ctypedef struct llama_model:
        pass
    
    ctypedef struct llama_context:
        pass
    
    ctypedef struct llama_model_params:
        int n_gpu_layers
    
    ctypedef struct llama_context_params:
        int n_ctx
        int n_batch
        int n_threads
        int n_threads_batch
    
    ctypedef int llama_seq_id
    
    llama_model_params llama_model_default_params()
    llama_context_params llama_context_default_params()
    void llama_backend_init()
    void llama_backend_free()
    llama_model* llama_model_load_from_file(const char* path, llama_model_params params)
    llama_context* llama_init_from_model(llama_model* model, llama_context_params params)
    void llama_model_free(llama_model* model)
    void llama_free(llama_context* ctx)
    int llama_n_ctx(llama_context* ctx)
    int llama_n_ctx_train(llama_model* model)

cdef extern from "common.h":
    ctypedef struct common_params_sampling:
        float temp
        int top_k
        float top_p
        float penalty_repeat
        int penalty_last_n
        string grammar
    
    ctypedef struct common_sampler:
        pass
    
    common_sampler* common_sampler_init(llama_model* model, common_params_sampling params)
    void common_sampler_free(common_sampler* sampler)

cdef extern from "ggml.h":
    ctypedef enum ggml_log_level:
        GGML_LOG_LEVEL_ERROR = 2

cdef extern from "mtmd.h":
    ctypedef struct mtmd_context:
        pass
    
    ctypedef struct mtmd_context_params:
        int use_gpu
        int n_threads
        ggml_log_level verbosity
    
    ctypedef struct mtmd_bitmap:
        pass

    ctypedef struct mtmd_input_text:
        const char* text
        int add_special
        int parse_special
    
    ctypedef struct mtmd_input_chunks:
        pass
    
    mtmd_context* mtmd_init_from_file(const char* path, llama_model* model, mtmd_context_params params)
    void mtmd_free(mtmd_context* ctx_mtmd)
    uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap* bitmap)
    uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap* bitmap)
    void mtmd_bitmap_free(mtmd_bitmap* bitmap)
    mtmd_input_chunks* mtmd_input_chunks_init()
    void mtmd_input_chunks_free(mtmd_input_chunks* chunks)
    int mtmd_tokenize(mtmd_context* ctx_mtmd, mtmd_input_chunks* chunks, const mtmd_input_text* input_text, const mtmd_bitmap** bitmaps, size_t n_bitmaps)

cdef extern from "mtmd-helper.h":
    mtmd_bitmap* mtmd_helper_bitmap_init_from_file(mtmd_context* ctx, const char* fname)
    size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks* chunks)
    int32_t mtmd_helper_eval_chunks(mtmd_context* ctx, llama_context* lctx, const mtmd_input_chunks* chunks, int n_past, llama_seq_id seq_id, int n_batch, bint logits_last, int* new_n_past)

cdef extern from "generation_helper.h":
    ctypedef struct GenerationResult:
        int final_n_past
        int total_tokens_generated
        string generated_text
    
    GenerationResult generate_tokens_cpp(
        common_sampler* sampler,
        llama_context* ctx,
        llama_model* model,
        int32_t n_past_start,
        int32_t n_ctx,
        int32_t max_new_tokens,
        vector[llama_seq_id] seq_ids
    )

# Global variables to keep references
cdef llama_model* g_model = NULL
cdef llama_context* g_ctx = NULL
cdef mtmd_context* g_ctx_mtmd = NULL
cdef common_sampler* g_sampler = NULL
cdef mtmd_bitmap* g_bitmap = NULL
cdef mtmd_input_chunks* g_chunks = NULL

# Python-accessible functions
def initialize_backend():
    """Initialize the llama backend."""
    llama_backend_init()

def cleanup_backend():
    """Clean up the llama backend."""
    global g_model, g_ctx, g_ctx_mtmd, g_sampler, g_bitmap, g_chunks
    
    if g_sampler != NULL:
        common_sampler_free(g_sampler)
        g_sampler = NULL
    
    if g_bitmap != NULL:
        mtmd_bitmap_free(g_bitmap)
        g_bitmap = NULL

    if g_chunks != NULL:
        mtmd_input_chunks_free(g_chunks)
        g_chunks = NULL
    
    if g_ctx != NULL:
        llama_free(g_ctx)
        g_ctx = NULL
    
    if g_ctx_mtmd != NULL:
        mtmd_free(g_ctx_mtmd)
        g_ctx_mtmd = NULL
    
    if g_model != NULL:
        llama_model_free(g_model)
        g_model = NULL
    
    llama_backend_free()

def load_model(model_path, n_gpu_layers):
    """Load a model from a file."""
    global g_model
    
    cdef llama_model_params params = llama_model_default_params()
    params.n_gpu_layers = n_gpu_layers
    
    cdef bytes model_path_bytes = model_path.encode('utf-8')
    g_model = llama_model_load_from_file(model_path_bytes, params)
    
    if g_model == NULL:
        raise RuntimeError(f"Failed to load model from {model_path}")
    
    return <unsigned long>g_model

def create_context(model_handle, n_ctx, n_batch, n_threads):
    """Create a llama context."""
    global g_ctx, g_model
    
    # Convert Python object back to C++ pointer
    cdef llama_model* model = <llama_model*><unsigned long>model_handle
    
    cdef llama_context_params params = llama_context_default_params()
    params.n_ctx = n_ctx
    params.n_batch = n_batch
    params.n_threads = n_threads
    params.n_threads_batch = n_threads
    
    g_ctx = llama_init_from_model(model, params)
    
    if g_ctx == NULL:
        raise RuntimeError("Failed to create LLaMA context")
    
    return <unsigned long>g_ctx

def load_mtmd_context(mmproj_path, model_handle, use_gpu, n_threads):
    """Load a multimodal context."""
    global g_ctx_mtmd
    
    # Convert Python object back to C++ pointer
    cdef llama_model* model = <llama_model*><unsigned long>model_handle
    
    cdef mtmd_context_params params
    params.use_gpu = 1 if use_gpu else 0
    params.n_threads = n_threads
    params.verbosity = GGML_LOG_LEVEL_ERROR
    
    cdef bytes mmproj_path_bytes = mmproj_path.encode('utf-8')
    g_ctx_mtmd = mtmd_init_from_file(mmproj_path_bytes, model, params)
    
    if g_ctx_mtmd == NULL:
        raise RuntimeError(f"Failed to load multimodal projector from {mmproj_path}")
    
    return <unsigned long>g_ctx_mtmd

def create_sampler(model_handle, temp, top_k, top_p, repeat_penalty, ctx_size):
    """Create a sampler."""
    global g_sampler
    
    # Convert Python object back to C++ pointer
    cdef llama_model* model = <llama_model*><unsigned long>model_handle
    
    cdef common_params_sampling params
    params.temp = temp
    params.top_k = top_k
    params.top_p = top_p
    params.penalty_repeat = repeat_penalty
    params.penalty_last_n = ctx_size
    params.grammar = b""
    
    g_sampler = common_sampler_init(model, params)
    
    if g_sampler == NULL:
        raise RuntimeError("Failed to initialize common_sampler")
    
    return <unsigned long>g_sampler

def load_image(ctx_mtmd_handle, image_path):
    """Load an image from a file."""
    global g_bitmap, g_ctx_mtmd

    cdef mtmd_context* ctx_mtmd = <mtmd_context*><unsigned long>ctx_mtmd_handle
    cdef bytes image_path_bytes = image_path.encode('utf-8')
    g_bitmap = mtmd_helper_bitmap_init_from_file(ctx_mtmd, image_path_bytes)

    if g_bitmap == NULL:
        raise RuntimeError(f"Failed to load image {image_path}")

    return {
        'handle': <unsigned long>g_bitmap,
        'width': mtmd_bitmap_get_nx(g_bitmap),
        'height': mtmd_bitmap_get_ny(g_bitmap)
    }

def tokenize_input(ctx_mtmd_handle, prompt, bitmap_handle):
    """Tokenize the input text and image."""
    global g_chunks
    
    # Convert Python objects back to C++ pointers
    cdef mtmd_context* ctx_mtmd = <mtmd_context*><unsigned long>ctx_mtmd_handle
    cdef mtmd_bitmap* bitmap = <mtmd_bitmap*><unsigned long>bitmap_handle
    
    # Create input text
    cdef mtmd_input_text input_text
    cdef bytes prompt_bytes = prompt.encode('utf-8')
    input_text.text = prompt_bytes
    input_text.add_special = 1
    input_text.parse_special = 1
    
    # Create bitmap array
    cdef const mtmd_bitmap* bitmaps[1]
    bitmaps[0] = bitmap
    
    # Create chunks
    g_chunks = mtmd_input_chunks_init()
    if g_chunks == NULL:
        raise MemoryError("Failed to allocate memory for mtmd_input_chunks")
    
    # Tokenize
    cdef int ret = mtmd_tokenize(ctx_mtmd, g_chunks, &input_text, bitmaps, 1)
    
    if ret != 0:
        raise RuntimeError(f"Failed mtmd_tokenize (code: {ret})")
    
    # Get token count
    cdef int n_tokens = mtmd_helper_get_n_tokens(g_chunks)
    
    return {
        'handle': <unsigned long>g_chunks,
        'n_tokens': n_tokens
    }

def evaluate_input(ctx_mtmd_handle, ctx_handle, chunks_handle, n_batch):
    """Evaluate the tokenized input."""
    # Convert Python objects back to C++ pointers
    cdef mtmd_context* ctx_mtmd = <mtmd_context*><unsigned long>ctx_mtmd_handle
    cdef llama_context* ctx = <llama_context*><unsigned long>ctx_handle
    cdef mtmd_input_chunks* chunks = <mtmd_input_chunks*><unsigned long>chunks_handle

    cdef int n_past = 0
    cdef llama_seq_id seq_id = 0
    cdef int new_n_past = 0

    # Evaluate - pass by reference
    cdef int ret = mtmd_helper_eval_chunks(ctx_mtmd, ctx, chunks, n_past, seq_id, n_batch, True, &new_n_past)

    if ret != 0:
        raise RuntimeError(f"Failed mtmd_helper_eval_chunks (code: {ret})")

    return new_n_past

def generate_tokens(sampler_handle, ctx_handle, model_handle, n_past, n_ctx, max_new_tokens):
    """Generate tokens using the model."""
    # Convert Python objects back to C++ pointers
    cdef common_sampler* sampler = <common_sampler*><unsigned long>sampler_handle
    cdef llama_context* ctx = <llama_context*><unsigned long>ctx_handle
    cdef llama_model* model = <llama_model*><unsigned long>model_handle
    
    # Create sequence ID vector
    cdef vector[llama_seq_id] seq_ids
    seq_ids.push_back(0)
    
    # Generate tokens
    cdef GenerationResult result = generate_tokens_cpp(
        sampler, ctx, model, n_past, n_ctx, max_new_tokens, seq_ids
    )
    
    # Convert result to Python dict
    return {
        'text': result.generated_text.decode('utf-8', errors='replace'),
        'n_tokens': result.total_tokens_generated,
        'final_n_past': result.final_n_past
    }
