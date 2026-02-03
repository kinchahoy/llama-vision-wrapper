#include "generation_helper.h"

#include <algorithm>
#include <cstdio> // For printf/fprintf
#include <cstring>
#include <string> // For std::string
#include <vector>

#include "common.h" // For llama_token_to_piece
#include "llama.h"  // Ensure llama definitions are included
#include "mtmd.h"
#include "mtmd-helper.h"

namespace {

constexpr uint32_t EMB_MAGIC = 0x454D4244; // "EMBD"
constexpr uint32_t EMB_VERSION = 1;

constexpr uint32_t KV_MAGIC = 0x4B565354; // "KVST"
constexpr uint32_t KV_VERSION = 1;

struct embedding_file_header_legacy {
    uint32_t nx;
    uint32_t ny;
    uint32_t use_mrope_pos;
};

struct embedding_file_header {
    uint32_t magic;
    uint32_t version;
    uint32_t nx;
    uint32_t ny;
    uint32_t use_mrope_pos;
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t checksum;
};

struct kv_prefix_header {
    uint32_t magic;
    uint32_t version;
    uint32_t n_past;
    uint32_t n_ctx;
    uint64_t state_size;
    uint32_t checksum;
};

uint32_t fnv1a_checksum(const uint8_t * data, size_t size) {
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    return hash;
}

uint32_t fnv1a_checksum_floats(const float * data, size_t count) {
    return fnv1a_checksum(reinterpret_cast<const uint8_t *>(data), count * sizeof(float));
}

} // namespace

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

TextPrefillResult eval_text_chunk_cpp(
    struct llama_context * ctx,
    const mtmd_input_chunk * chunk,
    llama_pos n_past,
    int32_t n_batch,
    bool logits_for_last,
    llama_seq_id seq_id)
{
    TextPrefillResult result = {/*status=*/-1, /*final_n_past=*/n_past, /*last_logits_index=*/-1};
    if (!ctx || !chunk || n_batch <= 0) {
        return result;
    }

    size_t n_tokens = 0;
    const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
    if (!tokens || n_tokens == 0) {
        return result;
    }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    if (!batch.token) {
        return result;
    }

    llama_pos current_n_past = n_past;
    size_t token_idx = 0;

    while (token_idx < n_tokens) {
        batch.n_tokens = 0;

        while (token_idx < n_tokens && batch.n_tokens < static_cast<size_t>(n_batch)) {
            const size_t j = batch.n_tokens;
            batch.token[j] = tokens[token_idx];
            batch.pos[j] = current_n_past;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq_id;
            batch.logits[j] = false;

            ++batch.n_tokens;
            ++current_n_past;
            ++token_idx;
        }

        if (logits_for_last && token_idx == n_tokens && batch.n_tokens > 0) {
            batch.logits[batch.n_tokens - 1] = true;
            result.last_logits_index = static_cast<int32_t>(batch.n_tokens - 1);
        }

        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return result;
        }
    }

    llama_batch_free(batch);
    result.status = 0;
    result.final_n_past = current_n_past;
    return result;
}

// --- Media Embedding I/O Helpers ---

bool save_media_embedding_c(
    const std::string& file_path,
    int nx,
    int ny,
    bool use_mrope_pos,
    int32_t n_tokens,
    int32_t n_embd,
    const float* embd_ptr)
{
    FILE* f = fopen(file_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s for writing.\n", file_path.c_str());
        return false;
    }

    const size_t total_floats = static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd);
    const uint32_t checksum = fnv1a_checksum_floats(embd_ptr, total_floats);

    embedding_file_header header = {
        EMB_MAGIC,
        EMB_VERSION,
        (uint32_t)nx,
        (uint32_t)ny,
        (uint32_t)use_mrope_pos,
        (uint32_t)n_tokens,
        (uint32_t)n_embd,
        checksum
    };

    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to write header to %s.\n", file_path.c_str());
        fclose(f);
        return false;
    }

    if (fwrite(embd_ptr, sizeof(float), total_floats, f) != total_floats) {
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
    MediaLoadResult result = {false, 0, 0, false, 0, 0, 0, false, false, 0, 0, 0};
    FILE* f = fopen(file_path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s for reading.\n", file_path.c_str());
        return result;
    }

    embedding_file_header header = {};
    embedding_file_header_legacy header_legacy = {};
    bool legacy = false;

    if (fread(&header, sizeof(header), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read header from %s.\n", file_path.c_str());
        fclose(f);
        return result;
    }

    if (header.magic != EMB_MAGIC) {
        legacy = true;
        if (fseek(f, 0, SEEK_SET) != 0) {
            fclose(f);
            return result;
        }
        if (fread(&header_legacy, sizeof(header_legacy), 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read legacy header from %s.\n", file_path.c_str());
            fclose(f);
            return result;
        }
        result.nx = header_legacy.nx;
        result.ny = header_legacy.ny;
        result.use_mrope_pos = (bool)header_legacy.use_mrope_pos;
    } else {
        result.nx = header.nx;
        result.ny = header.ny;
        result.use_mrope_pos = (bool)header.use_mrope_pos;
        result.version = header.version;
        result.checksum_expected = header.checksum;
        result.n_tokens = header.n_tokens;
        result.n_embd = header.n_embd;
    }
    result.legacy_format = legacy;

    // Get file size to determine embedding size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    long data_size = file_size - (legacy ? (long)sizeof(header_legacy) : (long)sizeof(header));
    fseek(f, legacy ? sizeof(header_legacy) : sizeof(header), SEEK_SET);

    if (data_size < 0 || data_size % sizeof(float) != 0) {
        fprintf(stderr, "Error: Invalid data size in %s.\n", file_path.c_str());
        fclose(f);
        return result;
    }

    size_t n_floats = (size_t)data_size / sizeof(float);
    embd_vec.resize(n_floats);
    result.n_floats = n_floats;

    if (fread(embd_vec.data(), sizeof(float), n_floats, f) != n_floats) {
        fprintf(stderr, "Error: Failed to read embedding data from %s.\n", file_path.c_str());
        fclose(f);
        return result;
    }

    fclose(f);

    if (!legacy) {
        result.checksum_computed = fnv1a_checksum_floats(embd_vec.data(), n_floats);
        result.checksum_ok = (result.checksum_computed == result.checksum_expected);
        if (!result.checksum_ok) {
            fprintf(stderr, "Error: Checksum mismatch in %s (expected %u, got %u)\n",
                    file_path.c_str(), result.checksum_expected, result.checksum_computed);
            return result;
        }

        if (result.n_tokens > 0 && result.n_embd > 0) {
            size_t expected = (size_t)result.n_tokens * (size_t)result.n_embd;
            if (expected != n_floats) {
                fprintf(stderr,
                        "Error: Metadata mismatch in %s (header tokens x dim = %zu, payload floats = %zu)\n",
                        file_path.c_str(), expected, n_floats);
                return result;
            }
        }
    }

    result.success = true;
    return result;
}

int32_t decode_media_chunk_from_embd(
    mtmd_context * ctx_mtmd,
    struct llama_context * ctx,
    const mtmd_input_chunk * chunk,
    const float * embd,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    llama_pos * new_n_past)
{
    if (!ctx_mtmd || !ctx || !chunk || !embd || !new_n_past) {
        return -1;
    }

    llama_pos n_past_out = 0;
    int32_t ret = 0;
    const bool use_non_causal = mtmd_decode_use_non_causal(ctx_mtmd);
    if (use_non_causal) {
        llama_set_causal_attn(ctx, false);
    }

    // mtmd_helper_decode_image_chunk expects a mutable pointer; the buffer is not modified.
    float * embd_mutable = const_cast<float *>(embd);
    ret = mtmd_helper_decode_image_chunk(
        ctx_mtmd,
        ctx,
        chunk,
        embd_mutable,
        n_past,
        seq_id,
        n_batch,
        &n_past_out);

    if (use_non_causal) {
        llama_set_causal_attn(ctx, true);
    }

    if (ret == 0) {
        *new_n_past = n_past_out;
    }
    return ret;
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

bool save_kv_prefix(
    struct llama_context * ctx,
    const std::string& file_path,
    llama_pos n_past)
{
    if (!ctx) {
        return false;
    }

    const size_t state_size = llama_state_get_size(ctx);
    if (state_size == 0) {
        return false;
    }

    std::vector<uint8_t> buffer(state_size);
    const size_t bytes_written = llama_state_get_data(ctx, buffer.data(), buffer.size());
    if (bytes_written == 0) {
        return false;
    }

    kv_prefix_header header = {
        KV_MAGIC,
        KV_VERSION,
        (uint32_t)n_past,
        (uint32_t)llama_n_ctx(ctx),
        (uint64_t)bytes_written,
        fnv1a_checksum(buffer.data(), bytes_written),
    };

    FILE * f = fopen(file_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s for writing KV prefix.\n", file_path.c_str());
        return false;
    }

    bool ok = true;
    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        ok = false;
    } else if (fwrite(buffer.data(), 1, bytes_written, f) != bytes_written) {
        ok = false;
    }

    fclose(f);
    if (!ok) {
        fprintf(stderr, "Error: Failed to write KV prefix to %s.\n", file_path.c_str());
    }
    return ok;
}

bool load_kv_prefix(
    struct llama_context * ctx,
    const std::string& file_path,
    llama_pos& n_past_out)
{
    if (!ctx) {
        return false;
    }

    FILE * f = fopen(file_path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s for reading KV prefix.\n", file_path.c_str());
        return false;
    }

    kv_prefix_header header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        fprintf(stderr, "Error: Failed to read KV prefix header from %s.\n", file_path.c_str());
        return false;
    }

    if (header.magic != KV_MAGIC || header.version != KV_VERSION) {
        fclose(f);
        fprintf(stderr, "Error: KV prefix header mismatch in %s.\n", file_path.c_str());
        return false;
    }

    const uint32_t ctx_n_ctx = llama_n_ctx(ctx);
    if (header.n_ctx > ctx_n_ctx) {
        fprintf(stderr,
                "Warning: KV prefix context size (%u) exceeds current ctx (%u). Attempting load.\n",
                header.n_ctx, ctx_n_ctx);
    }

    std::vector<uint8_t> buffer(header.state_size);
    if (fread(buffer.data(), 1, header.state_size, f) != header.state_size) {
        fclose(f);
        fprintf(stderr, "Error: Failed to read KV prefix payload from %s.\n", file_path.c_str());
        return false;
    }
    fclose(f);

    const uint32_t computed = fnv1a_checksum(buffer.data(), buffer.size());
    if (computed != header.checksum) {
        fprintf(stderr,
                "Error: KV prefix checksum mismatch in %s (expected %u, got %u).\n",
                file_path.c_str(), header.checksum, computed);
        return false;
    }

    const size_t applied = llama_state_set_data(ctx, buffer.data(), buffer.size());
    if (applied != buffer.size()) {
        fprintf(stderr, "Error: Failed to apply KV prefix from %s (applied %zu / %zu bytes).\n",
                file_path.c_str(), applied, buffer.size());
        return false;
    }

    n_past_out = header.n_past;
    return true;
}
