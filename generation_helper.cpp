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
    const std::vector<llama_seq_id> seq_ids, // Changed to pass by value
    PythonCallbackFunc callback,
    int callback_threshold)
{
    GenerationResult result;
    result.final_n_past = n_past_start;
    result.total_tokens_generated = 0;

    // Check if the vector is empty (no need to check for null pointer now)
    if (seq_ids.empty()) {
        fprintf(stderr, "Error: Empty sequence ID vector provided to generate_tokens_cpp.\n");
        result.total_tokens_generated = -1; // Indicate error
        return result; // Return empty result
    }
    // Get data pointer directly from the vector value
    const llama_seq_id * p_seq_id_const = seq_ids.data();

    // Use a local batch for single-token decoding within the loop
    // Initialize batch for 1 token, 0 embedding data, 1 sequence ID list per token
    struct llama_batch batch = llama_batch_init(1, 0, 1);
    if (!batch.token) { // Check if init failed
         fprintf(stderr, "Error: Failed to initialize batch in generate_tokens_cpp.\n");
         return result;
    }

    std::string current_chunk = ""; // Accumulate text for the next callback
    int tokens_in_current_chunk = 0;
    const int buffer_size = 256; // Max size for a single token piece string
    char piece_buffer[buffer_size];

    while (result.final_n_past < n_ctx && result.total_tokens_generated < max_new_tokens) {
        // 1. Sample
        llama_token id = common_sampler_sample(sampler, ctx, -1);

        // 2. Accept
        common_sampler_accept(sampler, id, true);

        // 3. Check for End-of-Sequence
        const struct llama_vocab * vocab = llama_model_get_vocab(model);
        if (llama_vocab_is_eog(vocab, id)) {
            // printf("\n[EOS reached in C++]\n");
            // printf("\n[EOS reached in C++]\n");
            break;
        }

        // 4. Convert token to piece and accumulate
        int n_chars = llama_token_to_piece(model, id, piece_buffer, buffer_size);
        if (n_chars < 0) {
             fprintf(stderr, "\nError: llama_token_to_piece returned %d\n", n_chars);
             // Decide how to handle: continue, break, return error? Let's break.
             break;
        }
        current_chunk.append(piece_buffer, n_chars);
        tokens_in_current_chunk++;
        result.total_tokens_generated++;


        // 5. Prepare batch for decoding this single token
        batch.n_tokens = 1; // Explicitly set number of tokens in batch
        batch.token [0] = id;
        batch.pos   [0] = result.final_n_past;
        batch.n_seq_id[0] = seq_ids.size(); // Use . for value access
        // Use const_cast
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

        // 8. Check if callback threshold is met
        if (callback && tokens_in_current_chunk >= callback_threshold) {
            callback(current_chunk.c_str(), tokens_in_current_chunk);
            current_chunk.clear(); // Reset for the next chunk
            tokens_in_current_chunk = 0;
        }
    }

    // 9. Send any remaining partial chunk via callback
    if (callback && tokens_in_current_chunk > 0) {
        callback(current_chunk.c_str(), tokens_in_current_chunk);
    }


    // Clean up the local batch
    llama_batch_free(batch);

    return result;
}
