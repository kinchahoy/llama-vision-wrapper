#include "generation_helper.h"
#include <cstdio> // For printf/fprintf
#include <vector>
#include <string> // For std::string
#include "llama.h" // Ensure llama definitions are included
#include "common.h" // For llama_token_to_piece

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
    struct llama_batch batch = llama_batch_init(1, 0, 1);
    if (!batch.token) {
         fprintf(stderr, "Error: Failed to initialize batch in generate_tokens_cpp.\n");
         result.total_tokens_generated = -1;
         return result;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    std::string generated_text;
    generated_text.reserve(max_new_tokens * 4); // Pre-allocate for better performance

    while (result.final_n_past < n_ctx && result.total_tokens_generated < max_new_tokens) {
        // Sample next token
        llama_token token_id = common_sampler_sample(sampler, ctx, -1);
        
        // Accept the token
        common_sampler_accept(sampler, token_id, true);

        // Check for end of generation
        if (llama_vocab_is_eog(vocab, token_id)) {
            break;
        }

        // Convert token to text and append immediately
        std::string piece = common_token_to_piece(ctx, token_id);
        generated_text += piece;

        result.total_tokens_generated++;

        // Prepare batch using common functions (like CLI does)
        common_batch_clear(batch);
        common_batch_add(batch, token_id, result.final_n_past, {0}, true);

        // Decode the token
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "Error: llama_decode failed at position %d\n", result.final_n_past);
            break;
        }

        result.final_n_past++;
    }

    result.generated_text = std::move(generated_text);
    llama_batch_free(batch);
    return result;
}
