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

    // Convert all tokens to text at once (outside the hot loop)
    std::string generated_text;
    generated_text.reserve(result.total_tokens_generated * 4);
    for (llama_token token : generated_tokens) {
        generated_text += common_token_to_piece(ctx, token);
    }

    result.generated_text = std::move(generated_text);
    llama_batch_free(batch);
    return result;
}

// --- Media Embedding I/O Helpers ---

// Simple header for the embedding file
struct embedding_file_header {
    uint32_t nx;
    uint32_t ny;
    uint32_t use_mrope_pos;
};

bool save_media_embedding(
    const std::string& file_path,
    int nx,
    int ny,
    bool use_mrope_pos,
    size_t n_embd,
    const float* embd_ptr)
{
    FILE* f = fopen(file_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s for writing.\n", file_path.c_str());
        return false;
    }

    embedding_file_header header = {
        (uint32_t)nx,
        (uint32_t)ny,
        (uint32_t)use_mrope_pos
    };

    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to write header to %s.\n", file_path.c_str());
        fclose(f);
        return false;
    }

    if (fwrite(embd_ptr, sizeof(float), n_embd, f) != n_embd) {
        fprintf(stderr, "Error: Failed to write embedding data to %s.\n", file_path.c_str());
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

MediaLoadResult load_media_embedding(
    const std::string& file_path,
    std::vector<float>& embd_vec)
{
    MediaLoadResult result = {false, 0, 0, false};
    FILE* f = fopen(file_path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s for reading.\n", file_path.c_str());
        return result;
    }

    embedding_file_header header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read header from %s.\n", file_path.c_str());
        fclose(f);
        return result;
    }

    result.nx = header.nx;
    result.ny = header.ny;
    result.use_mrope_pos = (bool)header.use_mrope_pos;

    // Get file size to determine embedding size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    long data_size = file_size - sizeof(header);
    fseek(f, sizeof(header), SEEK_SET);

    if (data_size < 0 || data_size % sizeof(float) != 0) {
        fprintf(stderr, "Error: Invalid data size in %s.\n", file_path.c_str());
        fclose(f);
        return result;
    }

    size_t n_embd = data_size / sizeof(float);
    embd_vec.resize(n_embd);

    if (fread(embd_vec.data(), sizeof(float), n_embd, f) != n_embd) {
        fprintf(stderr, "Error: Failed to read embedding data from %s.\n", file_path.c_str());
        fclose(f);
        return result;
    }

    fclose(f);
    result.success = true;
    return result;
}
