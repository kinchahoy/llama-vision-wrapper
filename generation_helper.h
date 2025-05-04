#ifndef GENERATION_HELPER_H
#define GENERATION_HELPER_H

#include <vector>
#include <string> // Needed for std::string
#include "llama.h"
#include "sampling.h" // For common_sampler
#include "common.h"   // For llama_token_to_piece

// Define the callback function pointer type
// Python side will provide a function matching this signature
// It receives the generated text chunk and the number of tokens in that chunk
typedef void (*PythonCallbackFunc)(const char* chunk, int n_tokens_in_chunk);

// Define a structure to hold the final results (excluding the full text)
struct GenerationResult {
    int final_n_past;
    int total_tokens_generated;
    // Add other potential outputs like timing, stop reason, etc. if desired
};

// The core generation function, now accepting a callback
GenerationResult generate_tokens_cpp(
    common_sampler * sampler,        // Pointer to the initialized sampler
    struct llama_context * ctx,      // Pointer to the llama context
    struct llama_model * model,      // Pointer to the llama model (needed for EOS/token conversion)
    int32_t n_past_start,            // Starting position in KV cache
    int32_t n_ctx,                   // Context size
    int32_t max_new_tokens,          // Max tokens to generate
    const std::vector<llama_seq_id>& seq_ids, // Sequence IDs (usually just {0})
    PythonCallbackFunc callback,     // Pointer to the Python callback function
    int callback_threshold           // How many tokens before calling back
);

#endif // GENERATION_HELPER_H
