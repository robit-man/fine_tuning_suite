#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

// CPU-only interface proof linked to the installed, official Ollama libraries.
// No generation service, CUDA allocation, or production-runtime modification.
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    ggml_backend_load_all_from_path("/usr/local/lib/ollama");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) throw std::runtime_error("text-model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = 32;
    cp.n_ubatch = 32;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    auto *lctx = llama_init_from_model(model, cp);
    if (!lctx) throw std::runtime_error("text context failed");
    auto vp = mtmd_context_params_default();
    vp.use_gpu = false;
    vp.warmup = false;
    vp.n_threads = 4;
    auto *vision = mtmd_init_from_file(argv[2], model, vp);
    if (!vision || !mtmd_support_vision(vision)) throw std::runtime_error("vision load failed");
    std::vector<std::vector<float>> outputs;
    std::vector<size_t> image_counts;
    for (int color : {0, 2, 0}) {
        llama_memory_clear(llama_get_memory(lctx), true);
        std::vector<unsigned char> pixels(28 * 28 * 3, 0);
        for (size_t i = 0; i < pixels.size(); i += 3) pixels[i + color] = 255;
        auto *bitmap = mtmd_bitmap_init(28, 28, pixels.data());
        const mtmd_bitmap *bitmaps[] = {bitmap};
        std::string prompt = std::string("Image: ") + mtmd_default_marker() + "\nDescribe the image.";
        mtmd_input_text input{prompt.c_str(), prompt.size(), false, true};
        auto *chunks = mtmd_input_chunks_init();
        if (mtmd_tokenize(vision, chunks, &input, bitmaps, 1)) throw std::runtime_error("image tokenize failed");
        std::string case_path = std::string(argv[3]) + ".case" + std::to_string(outputs.size());
        std::ofstream inputs(case_path + ".inputs.json");
        inputs << std::setprecision(9) << "{\"chunks\":[";
        size_t image_tokens = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            auto *chunk = mtmd_input_chunks_get(chunks, i);
            if (i) inputs << ',';
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                image_tokens += mtmd_input_chunk_get_n_tokens(chunk);
                if (mtmd_encode_chunk(vision, chunk)) throw std::runtime_error("vision encode failed");
                float *embeddings = mtmd_get_output_embd(vision);
                const size_t count = mtmd_input_chunk_get_n_tokens(chunk) * llama_model_n_embd_inp(model);
                inputs << "{\"type\":\"image\",\"embeddings\":[";
                for (size_t j = 0; j < count; ++j) {
                    if (j) inputs << ',';
                    inputs << embeddings[j];
                }
                inputs << "]}";
            } else {
                size_t count = 0;
                auto *tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
                inputs << "{\"type\":\"text\",\"tokens\":[";
                for (size_t j = 0; j < count; ++j) {
                    if (j) inputs << ',';
                    inputs << tokens[j];
                }
                inputs << "]}";
            }
        }
        inputs << "]}\n";
        inputs.close();
        llama_pos new_past = 0;
        if (mtmd_helper_eval_chunks(vision, lctx, chunks, 0, 0, 32, true, &new_past))
            throw std::runtime_error("image embedding decode failed");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float *logits = llama_get_logits(lctx);
        outputs.emplace_back(logits, logits + vocab);
        std::ofstream raw(case_path + ".logits.bin", std::ios::binary);
        raw.write(reinterpret_cast<const char *>(logits), vocab * sizeof(float));
        raw.close();
        image_counts.push_back(image_tokens);
        for (float value : outputs.back())
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite logits");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
    }
    double change = 0, repeat = 0;
    for (size_t i = 0; i < outputs[0].size(); ++i) {
        change += std::pow(double(outputs[0][i]) - outputs[1][i], 2);
        repeat += std::pow(double(outputs[0][i]) - outputs[2][i], 2);
    }
    change = std::sqrt(change / outputs[0].size());
    repeat = std::sqrt(repeat / outputs[0].size());
    bool passed = image_counts[0] > 0 && image_counts[0] == image_counts[1] && image_counts[1] == image_counts[2] &&
                  change > 1e-5 && repeat < 1e-5;
    std::ofstream report(argv[3]);
    report << "{\"passed\":" << (passed ? "true" : "false")
           << ",\"scope\":\"synthetic CPU interface proof; not learned image understanding\""
           << ",\"runtime\":\"installed official Ollama libllama/libmtmd\""
           << ",\"text_architecture\":\"afmoe\""
           << ",\"image_tokens\":[" << image_counts[0] << ',' << image_counts[1] << ',' << image_counts[2]
           << "],\"red_blue_logit_rmse\":" << change
           << ",\"red_red_logit_rmse\":" << repeat << "}\n";
    llama_free(lctx);
    mtmd_free(vision);
    llama_model_free(model);
    llama_backend_free();
    return passed ? 0 : 1;
}
