// Frozen feature extraction using the installed stock Ollama vision library.
// Training learns the replacement final mm.2 projection in PyTorch. This
// helper is used at training time only; published inference uses stock Ollama.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
struct Tap {
    llama_model *model = nullptr;
    mtmd_context *vision = nullptr;
    std::vector<float> features;
    std::vector<float> projected;
    int width = 0;
    int tokens = 0;
    std::string error;
    ~Tap() {
        if (vision) mtmd_free(vision);
        if (model) llama_model_free(model);
    }
};
void log_errors(enum ggml_log_level level, const char *message, void *) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) std::cerr << message;
}
bool capture(ggml_tensor *tensor, bool ask, void *opaque) {
    if (std::strcmp(ggml_get_name(tensor), "ffn_gelu") != 0) return false;
    if (ask) return true;
    auto &tap = *static_cast<Tap *>(opaque);
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] != 4608 || !ggml_is_contiguous(tensor)) {
        tap.error = "Unexpected frozen merger feature layout";
        return true;
    }
    tap.width = tensor->ne[0];
    tap.tokens = ggml_nelements(tensor) / tap.width;
    tap.features.resize(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, tap.features.data(), 0, ggml_nbytes(tensor));
    return true;
}
}

extern "C" {
const char *kolibri_vision_last_error() { return last_error.c_str(); }
void *kolibri_vision_open(const char *text_fixture, const char *projector, bool use_gpu, int min_tokens, int max_tokens) {
    try {
        llama_log_set(log_errors, nullptr);
        mtmd_helper_log_set(log_errors, nullptr);
        ggml_backend_load_all_from_path("/usr/local/lib/ollama");
        if (use_gpu) ggml_backend_load_all_from_path("/usr/local/lib/ollama/cuda_v12");
        llama_backend_init();
        auto tap = std::make_unique<Tap>();
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        tap->model = llama_model_load_from_file(text_fixture, params);
        if (!tap->model) throw std::runtime_error("Cannot load tokenizer fixture");
        auto vision = mtmd_context_params_default();
        vision.use_gpu = use_gpu;
        vision.warmup = false;
        vision.n_threads = 8;
        if (min_tokens <= 0 || max_tokens < min_tokens) throw std::runtime_error("Invalid image token bounds");
        vision.image_min_tokens = min_tokens;
        vision.image_max_tokens = max_tokens;
        vision.cb_eval = capture;
        vision.cb_eval_user_data = tap.get();
        if (use_gpu) {
            vision.device = ggml_backend_dev_by_name("CUDA0");
            if (!vision.device) throw std::runtime_error("Reserved CUDA0 device not visible");
        }
        tap->vision = mtmd_init_from_file(projector, tap->model, vision);
        if (!tap->vision) throw std::runtime_error("Cannot load pretrained vision tower");
        return tap.release();
    } catch (const std::exception &error) {
        last_error = error.what();
        return nullptr;
    }
}
int kolibri_vision_encode(void *handle, int width, int height, const unsigned char *rgb,
                          const float **output, int *tokens, int *feature_width) {
    try {
        auto &tap = *static_cast<Tap *>(handle);
        if (width <= 0 || height <= 0 || int64_t(width) * height > 32000000)
            throw std::runtime_error("Image exceeds the bounded input size");
        tap.features.clear();
        tap.error.clear();
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(mtmd_bitmap_init(width, height, rgb), mtmd_bitmap_free);
        const mtmd_bitmap *bitmaps[] = {bitmap.get()};
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
        std::string marker(mtmd_default_marker());
        mtmd_input_text input{marker.c_str(), marker.size(), false, true};
        if (mtmd_tokenize(tap.vision, chunks.get(), &input, bitmaps, 1))
            throw std::runtime_error("Native image preprocessing failed");
        bool found = false;
        for (size_t index = 0; index < mtmd_input_chunks_size(chunks.get()); ++index) {
            const auto *chunk = mtmd_input_chunks_get(chunks.get(), index);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                if (found) throw std::runtime_error("Unexpected multi-chunk image; implement packed concatenation before use");
                if (mtmd_encode_chunk(tap.vision, chunk)) throw std::runtime_error("Native image encoding failed");
                found = true;
            }
        }
        if (!tap.error.empty()) throw std::runtime_error(tap.error);
        if (!found || tap.features.empty()) throw std::runtime_error("Frozen feature callback did not fire");
        const float *projected = mtmd_get_output_embd(tap.vision);
        const int projected_width = llama_model_n_embd_inp(tap.model);
        tap.projected.assign(projected, projected + size_t(tap.tokens) * projected_width);
        *output = tap.features.data();
        *tokens = tap.tokens;
        *feature_width = tap.width;
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return 1;
    }
}
int kolibri_vision_projected(void *handle, const float **output, int *tokens, int *width) {
    auto &tap = *static_cast<Tap *>(handle);
    if (tap.projected.empty()) { last_error = "Encode an image before reading the projection"; return 1; }
    *output = tap.projected.data(); *tokens = tap.tokens; *width = llama_model_n_embd_inp(tap.model);
    return 0;
}
void kolibri_vision_close(void *handle) { delete static_cast<Tap *>(handle); }
}
