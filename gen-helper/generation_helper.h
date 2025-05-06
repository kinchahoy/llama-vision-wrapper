#ifndef GENERATION_HELPER_H
#define GENERATION_HELPER_H

#include <vector>
#include <string> // Needed for std::string
#include "llama.h"
#include "sampling.h" // For common_sampler
#include "common.h"   // For llama_token_to_piece

// Define the callback function pointer type
// Python side will provide a function matching this signature
// Define a structure to hold the final results, including the generated text
struct GenerationResult {
    int final_n_past;
    int total_tokens_generated;
    std::string generated_text; // Add field for the full generated text
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
    const std::vector<llama_seq_id> seq_ids // Sequence IDs (passed by value)
);

#endif // GENERATION_HELPER_H
