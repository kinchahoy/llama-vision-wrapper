#include "generation_helper.h"
#include <cstdio> // For printf/fprintf
#include <algorithm>
#include <cstring>
#include <vector>
#include <string> // For std::string
#include "llama.h" // Ensure llama definitions are included
#include "common.h" // For llama_token_to_piece
#include "mtmd.h"
#include "mtmd-helper.h"
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

namespace {

struct media_chunk_job {
    const mtmd_input_chunk * chunk;
    mtmd_input_chunk_type chunk_type;
    llama_seq_id seq_id;
    llama_pos n_past_start;
    int32_t n_tokens;
    llama_pos n_pos_total;
    int nx = 0;
    int ny = 0;
    std::vector<float> embd;
    int32_t processed = 0;
};

} // namespace

int32_t decode_media_chunks_batch(
    mtmd_context * ctx_mtmd,
    struct llama_context * ctx,
    const mtmd_input_chunk * const * chunks,
    const llama_pos * n_past,
    const llama_seq_id * seq_ids,
    int32_t n_chunks,
    int32_t n_batch,
    llama_pos * new_n_past)
{
    if (n_chunks <= 0) {
        return 0;
    }

    if (!ctx || !ctx_mtmd || !chunks || !n_past || !seq_ids || !new_n_past) {
        return -1;
    }

    const llama_model * model = llama_get_model(ctx);
    if (!model) {
        return -1;
    }

    const int n_mmproj_embd = llama_model_n_embd_inp(model);
    const bool use_mrope = mtmd_decode_use_mrope(ctx_mtmd);
    const bool use_non_causal = mtmd_decode_use_non_causal(ctx_mtmd);
    const int n_pos_per_embd = use_mrope ? 4 : 1;

    std::vector<media_chunk_job> jobs;
    jobs.reserve(n_chunks);

    for (int32_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk * chunk = chunks[i];
        if (!chunk) {
            return -1;
        }

        auto chunk_type = mtmd_input_chunk_get_type(chunk);
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            return -1;
        }

        media_chunk_job job;
        job.chunk = chunk;
        job.chunk_type = chunk_type;
        job.seq_id = seq_ids[i];
        job.n_past_start = n_past[i];
        job.n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        job.n_pos_total = mtmd_input_chunk_get_n_pos(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            if (!image_tokens) {
                return -1;
            }
            job.nx = mtmd_image_tokens_get_nx(image_tokens);
            job.ny = mtmd_image_tokens_get_ny(image_tokens);
        }

        if (mtmd_encode_chunk(ctx_mtmd, chunk) != 0) {
            return -1;
        }

        float * embd_ptr = mtmd_get_output_embd(ctx_mtmd);
        if (!embd_ptr) {
            return -1;
        }

        job.embd.resize(static_cast<size_t>(job.n_tokens) * n_mmproj_embd);
        std::memcpy(job.embd.data(), embd_ptr, sizeof(float) * job.embd.size());

        jobs.emplace_back(std::move(job));
    }

    std::vector<float> batch_embd(static_cast<size_t>(n_batch) * n_mmproj_embd);
    std::vector<llama_pos> batch_pos(static_cast<size_t>(n_batch) * n_pos_per_embd);
    std::vector<int32_t> batch_n_seq_id(n_batch);
    std::vector<llama_seq_id> batch_seq_storage(n_batch);
    std::vector<llama_seq_id *> batch_seq_id(n_batch + 1);
    std::vector<int8_t> batch_logits(n_batch);
    batch_seq_id[n_batch] = nullptr;

    llama_batch batch = {
        /*n_tokens =*/ 0,
        /*tokens   =*/ nullptr,
        /*embd     =*/ batch_embd.data(),
        /*pos      =*/ batch_pos.data(),
        /*n_seq_id =*/ batch_n_seq_id.data(),
        /*seq_id   =*/ batch_seq_id.data(),
        /*logits   =*/ batch_logits.data(),
    };

    int32_t ret = 0;

    if (use_non_causal) {
        llama_set_causal_attn(ctx, false);
    }

    while (true) {
        int tokens_added = 0;
        for (auto & job : jobs) {
            while (job.processed < job.n_tokens && tokens_added < n_batch) {
                const size_t source_index = static_cast<size_t>(job.processed);
                float * dst = batch_embd.data() + static_cast<size_t>(tokens_added) * n_mmproj_embd;
                const float * src = job.embd.data() + source_index * n_mmproj_embd;
                std::memcpy(dst, src, sizeof(float) * n_mmproj_embd);

                if (!use_mrope) {
                    batch_pos[static_cast<size_t>(tokens_added)] = job.n_past_start + job.processed;
                } else if (job.chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    const int nx = std::max(1, job.nx);
                    const int y = job.processed / nx;
                    const int x = job.processed % nx;
                    batch_pos[static_cast<size_t>(tokens_added)] = job.n_past_start + job.processed;
                    batch_pos[static_cast<size_t>(tokens_added) + n_batch] = job.n_past_start + y;
                    batch_pos[static_cast<size_t>(tokens_added) + 2 * n_batch] = job.n_past_start + x;
                    batch_pos[static_cast<size_t>(tokens_added) + 3 * n_batch] = 0;
                } else {
                    const llama_pos base = job.n_past_start + job.processed;
                    batch_pos[static_cast<size_t>(tokens_added)] = base;
                    batch_pos[static_cast<size_t>(tokens_added) + n_batch] = base;
                    batch_pos[static_cast<size_t>(tokens_added) + 2 * n_batch] = base;
                    batch_pos[static_cast<size_t>(tokens_added) + 3 * n_batch] = 0;
                }

                batch_n_seq_id[static_cast<size_t>(tokens_added)] = 1;
                batch_seq_storage[static_cast<size_t>(tokens_added)] = job.seq_id;
                batch_seq_id[static_cast<size_t>(tokens_added)] = &batch_seq_storage[static_cast<size_t>(tokens_added)];
                batch_logits[static_cast<size_t>(tokens_added)] = false;

                ++tokens_added;
                ++job.processed;
            }
        }

        if (tokens_added == 0) {
            break;
        }

        batch.n_tokens = tokens_added;
        batch.seq_id[static_cast<size_t>(tokens_added)] = nullptr;

        ret = llama_decode(ctx, batch);
        if (ret != 0) {
            break;
        }
    }

    if (use_non_causal) {
        llama_set_causal_attn(ctx, true);
    }

    if (ret == 0) {
        for (int32_t i = 0; i < n_chunks; ++i) {
            new_n_past[i] = n_past[i] + jobs[i].n_pos_total;
        }
    }

    return ret;
}
