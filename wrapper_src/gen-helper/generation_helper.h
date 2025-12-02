#ifndef GENERATION_HELPER_H
#define GENERATION_HELPER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "llama.h"
#include "sampling.h" // For common_sampler
#include "common.h"   // For llama_token_to_piece
#include "mtmd.h"
#include "mtmd-helper.h"

// Define the callback function pointer type
// Python side will provide a function matching this signature
// Define a structure to hold the final results, including the generated text
struct GenerationResult {
    int final_n_past;
    int total_tokens_generated;
    std::string generated_text; // Add field for the full generated text
    // Add other potential outputs like timing, stop reason, etc. if desired
};

// Define a structure to hold results from loading a media embedding
struct MediaLoadResult {
    bool success;
    int nx;
    int ny;
    bool use_mrope_pos;
    uint32_t version;
    uint32_t checksum_expected;
    uint32_t checksum_computed;
    bool checksum_ok;
    bool legacy_format;
    int32_t n_tokens;
    int32_t n_embd;
    size_t n_floats;
};

struct TextPrefillResult {
    int32_t status;
    llama_pos final_n_past;
    int32_t last_logits_index;
};

// Helper functions for saving and loading media embeddings
bool save_media_embedding_c(
    const std::string& file_path,
    int nx,
    int ny,
    bool use_mrope_pos,
    int32_t n_tokens,
    int32_t n_embd,
    const float* embd_ptr
);

MediaLoadResult load_media_embedding(
    const std::string& file_path,
    std::vector<float>& embd_vec
);

// Decode a single media chunk directly from a pre-loaded embedding buffer.
// Returns 0 on success; on success, new_n_past is updated.
int32_t decode_media_chunk_from_embd(
    mtmd_context * ctx_mtmd,
    struct llama_context * ctx,
    const mtmd_input_chunk * chunk,
    const float * embd,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    llama_pos * new_n_past
);

int32_t decode_media_chunks_batch(
    mtmd_context * ctx_mtmd,
    struct llama_context * ctx,
    const mtmd_input_chunk * const * chunks,
    const llama_pos * n_past,
    const llama_seq_id * seq_ids,
    int32_t n_chunks,
    int32_t n_batch,
    llama_pos * new_n_past
);

// Prefill text tokens for a single chunk using llama_batch on the C++ side.
TextPrefillResult eval_text_chunk_cpp(
    struct llama_context * ctx,
    const mtmd_input_chunk * chunk,
    llama_pos n_past,
    int32_t n_batch,
    bool logits_for_last,
    llama_seq_id seq_id);

// Save / load a KV prefix for fast prompt warmup.
bool save_kv_prefix(
    struct llama_context * ctx,
    const std::string& file_path,
    llama_pos n_past);

bool load_kv_prefix(
    struct llama_context * ctx,
    const std::string& file_path,
    llama_pos& n_past_out);

// The core generation function, now accepting a callback
GenerationResult generate_tokens_cpp(
    common_sampler * sampler,        // Pointer to the initialized sampler
    struct llama_context * ctx,      // Pointer to the llama context
    struct llama_model * model,      // Pointer to the llama model (needed for EOS/token conversion)
    int32_t n_past_start,            // Starting position in KV cache
    int32_t n_ctx,                   // Context size
    int32_t max_new_tokens,          // Max tokens to generate
    const std::vector<llama_seq_id> seq_ids // Sequence IDs (passed by value)
);

#endif // GENERATION_HELPER_H
