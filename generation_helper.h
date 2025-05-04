#ifndef GENERATION_HELPER_H
#define GENERATION_HELPER_H

#include <vector>
#include "llama.h"
#include "sampling.h" // For common_sampler
#include "common.h"   // For common_token_to_piece etc. (if needed inside, maybe not)

// Define a structure to hold the results
struct GenerationResult {
    std::vector<llama_token> tokens;
    int final_n_past;
    // Add other potential outputs like timing, stop reason, etc. if desired
};

// The core generation function
GenerationResult generate_tokens_cpp(
    common_sampler * sampler,        // Pointer to the initialized sampler
    struct llama_context * ctx,      // Pointer to the llama context
    struct llama_model * model,      // Pointer to the llama model (needed for EOS)
    int32_t n_past_start,            // Starting position in KV cache
    int32_t n_ctx,                   // Context size
    int32_t max_new_tokens,          // Max tokens to generate
    const std::vector<llama_seq_id>& seq_ids // Sequence IDs (usually just {0})
);

#endif // GENERATION_HELPER_H