#ifndef GENERATION_HELPER_H
#define GENERATION_HELPER_H

#include <vector>
#include <string> // Needed for std::string
#include "llama.h"
#include "sampling.h" // For common_sampler
#include "common.h"   // For llama_token_to_piece
#include "mtmd.h"

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
};

// Helper functions for saving and loading media embeddings
bool save_media_embedding(
    const std::string& file_path,
    int nx,
    int ny,
    bool use_mrope_pos,
    size_t n_embd,
    const float* embd_ptr
);

MediaLoadResult load_media_embedding(
    const std::string& file_path,
    std::vector<float>& embd_vec
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
