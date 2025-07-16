#include "generation_helper.h"
#include <cstdio> // For printf/fprintf
#include <vector>
#include <string> // For std::string
#include "llama.h" // Ensure llama definitions are included
#include "common.h" // For llama_token_to_piece
#include <chrono> // For timing

// Helper for timing output
static void print_timing(double total_seconds, int total_tokens) {
    if (total_tokens > 0) {
        printf("[Timing] Total: %.3f sec, Per token: %.3f ms\n",
               total_seconds, (total_seconds * 1000.0) / total_tokens);
    } else {
        printf("[Timing] Total: %.3f sec, No tokens generated.\n", total_seconds);
    }
}
GenerationResult generate_tokens_cpp(
    common_sampler * sampler,
    struct llama_context * ctx,
    struct llama_model * model,
    int32_t n_past_start,
    int32_t n_ctx,
    int32_t max_new_tokens,
    const std::vector<llama_seq_id> seq_ids)
{
    GenerationResult result;
    result.final_n_past = n_past_start;
    result.total_tokens_generated = 0;
    result.generated_text = "";

    // Check if the vector is empty
    if (seq_ids.empty()) {
        fprintf(stderr, "Error: Empty sequence ID vector provided to generate_tokens_cpp.\n");
        result.total_tokens_generated = -1;
        return result;
    }

    // Use a single batch for all token decoding - initialize once
    // For generation, a batch of size 1 is sufficient and more memory-efficient.
    struct llama_batch batch = llama_batch_init(1, 0, 1);
    if (!batch.token) {
         fprintf(stderr, "Error: Failed to initialize batch in generate_tokens_cpp.\n");
         result.total_tokens_generated = -1;
         return result;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    
    // Optimize string handling - collect tokens first, convert later
    std::vector<llama_token> generated_tokens;
    generated_tokens.reserve(max_new_tokens);

    // Start timing
    auto t_start = std::chrono::high_resolution_clock::now();

    while (result.final_n_past < n_ctx && result.total_tokens_generated < max_new_tokens) {
        // Sample next token
        llama_token token_id = common_sampler_sample(sampler, ctx, -1);
        
        // Check for end of generation first (before accepting)
        if (llama_vocab_is_eog(vocab, token_id)) {
            break;
        }
        
        // Accept the token
        common_sampler_accept(sampler, token_id, true);

        // Store token for later conversion
        generated_tokens.push_back(token_id);
        result.total_tokens_generated++;

        // Prepare batch using common functions (like CLI does)
        common_batch_clear(batch);
        common_batch_add(batch, token_id, result.final_n_past, seq_ids, true);

        // Decode the token
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "Error: llama_decode failed at position %d\n", result.final_n_past);
            break;
        }

        result.final_n_past++;
    }

    // End timing
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_seconds = std::chrono::duration<double>(t_end - t_start).count();

    // Convert all tokens to text at once (outside the hot loop)
    std::string generated_text;
    generated_text.reserve(result.total_tokens_generated * 4);
    for (llama_token token : generated_tokens) {
        generated_text += common_token_to_piece(ctx, token);
    }

    print_timing(total_seconds, result.total_tokens_generated);

    result.generated_text = std::move(generated_text);
    llama_batch_free(batch);
    return result;
}
