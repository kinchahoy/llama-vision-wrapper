"""Core llama.cpp interface with cppyy."""

import cppyy
import os
import sys

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CPP_DIR = f"{BASE_DIR}/llama.cpp"
LLAMA_CPP_LIBS_DIR = f"{LLAMA_CPP_DIR}/build/bin"
HELPER_LIB_DIR = f"{BASE_DIR}/gen-helper/build"

# Platform-specific library extension
LIB_EXT = "dylib" if sys.platform == "darwin" else "so"

# Libraries and headers
REQUIRED_LIBS = [
    f"libggml-base.{LIB_EXT}",
    f"libggml-cpu.{LIB_EXT}",
    f"libggml.{LIB_EXT}",
    f"libllama.{LIB_EXT}",
    f"libmtmd.{LIB_EXT}",
    f"libgeneration_helper.{LIB_EXT}",
]

INCLUDE_DIRS = [
    f"{LLAMA_CPP_DIR}/include",
    f"{LLAMA_CPP_DIR}/ggml/include", 
    f"{LLAMA_CPP_DIR}/common",
    f"{LLAMA_CPP_DIR}/tools/mtmd",
    f"{BASE_DIR}/gen-helper",
]

# Global state
_initialized = False
gbl = None


def initialize():
    """Initialize cppyy and load all required libraries."""
    global _initialized, gbl
    if _initialized:
        return gbl

    # Setup include paths
    for inc_path in INCLUDE_DIRS:
        if os.path.exists(inc_path):
            cppyy.add_include_path(inc_path)

    # Load libraries
    for lib_name in REQUIRED_LIBS:
        lib_dir = HELPER_LIB_DIR if "generation_helper" in lib_name else LLAMA_CPP_LIBS_DIR
        lib_path = os.path.join(lib_dir, lib_name)
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Required library not found: {lib_path}")
        cppyy.load_library(lib_path)

    # Include headers
    for header in ["vector", "string", "llama.h", "common.h", "sampling.h", "mtmd.h", "mtmd-helper.h", "generation_helper.h"]:
        cppyy.include(header)

    gbl = cppyy.gbl
    _initialized = True
    return gbl


class LlamaBackend:
    """Manages llama.cpp backend lifecycle."""
    
    def __init__(self):
        self.gbl = initialize()
        
    def __enter__(self):
        self.gbl.llama_backend_init()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gbl.llama_backend_free()


class ModelLoader:
    """Handles model and multimodal projector loading."""
    
    def __init__(self, gbl):
        self.gbl = gbl
        
    def load_model(self, model_path: str, gpu_layers: int = 0):
        """Load a GGUF model."""
        params = self.gbl.llama_model_default_params()
        params.n_gpu_layers = gpu_layers
        model = self.gbl.llama_load_model_from_file(model_path.encode("utf-8"), params)
        if not model:
            raise RuntimeError(f"Failed to load model: {model_path}")
        return model
        
    def create_context(self, model, n_ctx: int = 2048, n_batch: int = 512, n_threads: int = 8):
        """Create llama context."""
        params = self.gbl.llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_batch
        params.n_threads = params.n_threads_batch = n_threads
        params.log_level = self.gbl.GGML_LOG_LEVEL_ERROR
        
        ctx = self.gbl.llama_new_context_with_model(model, params)
        if not ctx:
            raise RuntimeError("Failed to create context")
        return ctx
        
    def load_multimodal(self, mmproj_path: str, model, use_gpu: bool = False, n_threads: int = 8):
        """Load multimodal projector."""
        params = self.gbl.mtmd_context_params_default()
        params.use_gpu = use_gpu
        params.n_threads = n_threads
        params.verbosity = self.gbl.GGML_LOG_LEVEL_ERROR
        
        ctx_mtmd = self.gbl.mtmd_init_from_file(mmproj_path.encode("utf-8"), model, params)
        if not ctx_mtmd:
            raise RuntimeError(f"Failed to load multimodal projector: {mmproj_path}")
        return ctx_mtmd


class MultimodalProcessor:
    """Handles image loading and multimodal processing."""
    
    def __init__(self, gbl):
        self.gbl = gbl
        self.loaded_embedding = None
        
    def load_image(self, ctx_mtmd, image_path: str):
        """Load an image into a bitmap."""
        bitmap = self.gbl.mtmd_helper_bitmap_init_from_file(ctx_mtmd, image_path.encode("utf-8"))
        if not bitmap:
            raise RuntimeError(f"Failed to load image: {image_path}")
        return bitmap
        
    def tokenize_prompt(self, ctx_mtmd, prompt: str, bitmaps):
        """Tokenize a prompt with embedded images."""
        prompt_cstr = prompt.encode("utf-8")
        
        input_text = self.gbl.mtmd_input_text()
        input_text.text = prompt_cstr
        input_text.add_special = True
        input_text.parse_special = True
        
        # Handle single bitmap or list
        if not isinstance(bitmaps, list):
            bitmaps = [bitmaps]
            
        bitmaps_vec = self.gbl.std.vector["const mtmd_bitmap*"]()
        for bitmap in bitmaps:
            bitmaps_vec.push_back(bitmap)
            
        chunks = self.gbl.mtmd_input_chunks_init()
        if not chunks:
            raise RuntimeError("Failed to initialize chunks")
            
        if self.gbl.mtmd_tokenize(ctx_mtmd, chunks, input_text, bitmaps_vec.data(), bitmaps_vec.size()) != 0:
            raise RuntimeError("Failed to tokenize")
            
        return chunks
        
    def process_chunks(self, ctx, ctx_mtmd, chunks, n_batch: int = 512):
        """Process all chunks and return final n_past."""
        n_past = 0
        n_chunks = self.gbl.mtmd_input_chunks_size(chunks)
        
        for i in range(n_chunks):
            chunk = self.gbl.mtmd_input_chunks_get(chunks, i)
            chunk_type = self.gbl.mtmd_input_chunk_get_type(chunk)
            is_last = i == n_chunks - 1
            
            if chunk_type == self.gbl.MTMD_INPUT_CHUNK_TYPE_TEXT:
                n_past = self._eval_text_chunk(ctx, chunk, n_past, n_batch, is_last)
            elif chunk_type in [self.gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE, self.gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO]:
                self.encode_media_chunk(ctx_mtmd, chunk)
                n_past = self._decode_media_chunk(ctx, ctx_mtmd, chunk, n_past, n_batch)
                
        return n_past
        
    def encode_media_chunk(self, ctx_mtmd, chunk):
        """Encode a media chunk."""
        if self.gbl.mtmd_encode_chunk(ctx_mtmd, chunk) != 0:
            raise RuntimeError("Failed to encode media chunk")
            
    def _decode_media_chunk(self, ctx, ctx_mtmd, chunk, n_past: int, n_batch: int):
        """Decode a media chunk."""
        embd = self.loaded_embedding.data() if self.loaded_embedding else self.gbl.mtmd_get_output_embd(ctx_mtmd)
        seq_id = self.gbl.llama_seq_id(0)
        
        new_n_past_array = self.gbl.std.array[self.gbl.llama_pos, 1]()
        new_n_past_array[0] = self.gbl.llama_pos(n_past)
        
        if self.gbl.mtmd_helper_decode_image_chunk(ctx_mtmd, ctx, chunk, embd, n_past, seq_id, n_batch, new_n_past_array.data()) != 0:
            raise RuntimeError("Failed to decode media chunk")
            
        return int(new_n_past_array[0])
        
    def _eval_text_chunk(self, ctx, chunk, n_past: int, n_batch: int, logits_for_last: bool = False):
        """Evaluate a text chunk."""
        n_tokens_out = self.gbl.std.vector["size_t"](1)
        tokens = self.gbl.mtmd_input_chunk_get_tokens_text(chunk, n_tokens_out.data())
        n_tokens = n_tokens_out[0]
        seq_id = self.gbl.llama_seq_id(0)
        
        batch = self.gbl.llama_batch_init(n_batch, 0, 1)
        
        token_idx = 0
        while token_idx < n_tokens:
            batch.n_tokens = 0
            while token_idx < n_tokens and batch.n_tokens < n_batch:
                j = batch.n_tokens
                batch.token[j] = tokens[token_idx]
                batch.pos[j] = n_past
                batch.n_seq_id[j] = 1
                batch.seq_id[j][0] = seq_id
                batch.logits[j] = False
                batch.n_tokens += 1
                n_past += 1
                token_idx += 1
                
            if logits_for_last and token_idx == n_tokens:
                batch.logits[batch.n_tokens - 1] = True
                
            if self.gbl.llama_decode(ctx, batch) != 0:
                self.gbl.llama_batch_free(batch)
                raise RuntimeError("Failed to decode text")
                
        self.gbl.llama_batch_free(batch)
        return n_past


class TextGenerator:
    """Handles text generation with sampling."""
    
    def __init__(self, gbl):
        self.gbl = gbl
        
    def create_sampler(self, model, temp: float = 0.8, top_k: int = 40, top_p: float = 0.95, repeat_penalty: float = 1.1, ctx_size: int = 2048):
        """Create a sampler with given parameters."""
        params = self.gbl.common_params_sampling()
        params.temp = temp
        params.top_k = top_k
        params.top_p = top_p
        params.penalty_repeat = repeat_penalty
        params.penalty_last_n = ctx_size
        params.grammar = ""
        
        sampler = self.gbl.common_sampler_init(model, params)
        if not sampler:
            raise RuntimeError("Failed to create sampler")
        return sampler
        
    def generate(self, sampler, ctx, model, n_past: int, n_ctx: int, max_tokens: int = 256):
        """Generate text tokens."""
        self.gbl.common_sampler_reset(sampler)
        seq_id_vec = self.gbl.std.vector[self.gbl.llama_seq_id]([self.gbl.llama_seq_id(0)])
        
        result = self.gbl.generate_tokens_cpp(sampler, ctx, model, n_past, n_ctx, max_tokens, seq_id_vec)
        return result