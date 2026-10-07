// CPU diagnostics using the installed official Ollama libraries only.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
struct Capture { std::string root; };
static bool capture(ggml_tensor *tensor, bool ask, void *opaque) {
    std::string name = ggml_get_name(tensor);
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) return false;
    bool first = name.size() > 2 && name.substr(name.size()-2) == "-0";
    bool output = name.rfind("l_out-", 0) == 0;
    if (!first && !output) return false;
    if (ask) return true;
    auto &state = *static_cast<Capture *>(opaque);
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    std::ofstream data(state.root + "/" + name + ".f32", std::ios::binary);
    data.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
    std::ofstream shape(state.root + "/" + name + ".shape");
    for (int i=0;i<4;++i) shape << tensor->ne[i] << " ";
    return true;
}
int main(int argc, char **argv) {
    if (argc!=4 && argc!=6) return 2;
    try {
        ggml_backend_load_all_from_path("/usr/local/lib/ollama");
        if (argc==6) ggml_backend_load_all_from_path("/usr/local/lib/ollama/cuda_v12");
        llama_backend_init();
        auto mp=llama_model_default_params(); mp.n_gpu_layers=0;
        if (argc==6) mp.n_gpu_layers=std::stoi(argv[4]);
        auto *model=llama_model_load_from_file(argv[1],mp);
        if (!model) throw std::runtime_error("Model load failed");
        Capture state{argv[3]};std::filesystem::create_directories(state.root);
        auto cp=llama_context_default_params();cp.n_ctx=64;cp.n_batch=64;cp.n_ubatch=64;
        cp.n_threads=cp.n_threads_batch=16;cp.type_k=cp.type_v=GGML_TYPE_F32;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.cb_eval=capture;cp.cb_eval_user_data=&state;
        auto *ctx=llama_init_from_model(model,cp);
        if (!ctx) throw std::runtime_error("Context load failed");
        if (argc==6) {
            // Scoped supervisor may acknowledge readiness only after model and
            // context allocations are resident. Leave enough time to observe it.
            std::ofstream ready(argv[5]);ready << "native model/context resident\n";ready.close();
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
        std::ifstream file(argv[2]);std::vector<llama_token> tokens;int token;
        while (file >> token) tokens.push_back(token);
        if (tokens.empty() || tokens.size()>64) throw std::runtime_error("Invalid bounded token input");
        auto batch=llama_batch_get_one(tokens.data(),tokens.size());
        std::vector<int8_t> flags(tokens.size(),1);batch.logits=flags.data();
        if (llama_decode(ctx,batch)) throw std::runtime_error("Decode failed");
        std::ofstream logits(state.root+"/logits.f32",std::ios::binary);
        logits.write(reinterpret_cast<const char *>(llama_get_logits_ith(ctx,-1)),llama_vocab_n_tokens(llama_model_get_vocab(model))*sizeof(float));
        llama_free(ctx);llama_model_free(model);llama_backend_free();
        std::cout << "Captured official stock CPU layer outputs\n";
    } catch(const std::exception &e) {std::cerr << e.what() << "\n";return 1;}
}
