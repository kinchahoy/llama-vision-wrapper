#include "generation_helper.h"
#include <cstdio> // For printf (optional debugging)
#include <vector>
#include "llama.h" // Ensure llama definitions are included

GenerationResult generate_tokens_cpp(
    common_sampler * sampler,
    struct llama_context * ctx,
    struct llama_model * model, // Pass model directly
    int32_t n_past_start,
    int32_t n_ctx,
    int32_t max_new_tokens,
    const std::vector<llama_seq_id>& seq_ids)
{
    GenerationResult result;
    result.final_n_past = n_past_start;
    int generated_token_count = 0;

    if (seq_ids.empty()) {
        fprintf(stderr, "Error: No sequence ID provided to generate_tokens_cpp.\n");
        return result; // Return empty result
    }
    // Get a potentially non-const pointer for the batch structure,
    // but we know llama_decode won't modify the content via this.
    const llama_seq_id * p_seq_id_const = seq_ids.data();

    // Use a local batch for single-token decoding within the loop
    // Initialize batch for 1 token, 0 embedding data, 1 sequence ID list per token
    struct llama_batch batch = llama_batch_init(1, 0, 1);
    if (!batch.token) { // Check if init failed
         fprintf(stderr, "Error: Failed to initialize batch in generate_tokens_cpp.\n");
         return result;
    }

    // No longer need to get vocab pointer here

    while (result.final_n_past < n_ctx && generated_token_count < max_new_tokens) {
        // 1. Sample
        llama_token id = common_sampler_sample(sampler, ctx, -1);

        // 2. Accept
        common_sampler_accept(sampler, id, true);

        // 3. Check for End-of-Sequence
        if (llama_vocab_is_eog(llama_get_vocab(model), id)) {
            // printf("\n[EOS reached in C++]\n");
            break;
        }

        // 4. Store token
        result.tokens.push_back(id);
        generated_token_count++;

        // 5. Prepare batch for decoding this single token
        // llama_batch_clear(batch); // <-- FIX APPLIED HERE (Line removed)
        batch.n_tokens = 1; // Explicitly set number of tokens in batch
        batch.token [0] = id;
        batch.pos   [0] = result.final_n_past;
        batch.n_seq_id[0] = seq_ids.size();
        // Use const_cast                            <-- FIX APPLIED HERE
        batch.seq_id[0] = const_cast<llama_seq_id *>(p_seq_id_const);
        batch.logits[0] = true; // We need logits for sampling the *next* token

        // 6. Decode the batch
        int decode_ret = llama_decode(ctx, batch);
        if (decode_ret != 0) {
            fprintf(stderr, "\nError: llama_decode returned %d in C++ loop (pos %d)\n", decode_ret, result.final_n_past);
            llama_batch_free(batch); // Clean up local batch before returning
            return result;
        }

        // 7. Update KV cache position
        result.final_n_past++;
    }

    // Clean up the local batch
    llama_batch_free(batch);

    return result;
}
