#include "bert.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace fs = std::filesystem;
constexpr int dim = 32, vocabulary_size = 12, position_rows = 18;
using Model = std::unique_ptr<bert_ctx, decltype(&bert_free)>;

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static float word_value(int token, int i) {
    return 0.1f * (token + 1) * (i % 5 - 2) + std::sin(float((token + 2) * (i + 1)) * 0.31f);
}
static float position_value(int position, int i) {
    return 0.4f * std::cos(float((position + 1) * (i + 2)) * 0.17f);
}
struct Options {
    bool roberta = true;
    std::string arch = "xlm-roberta";
    std::optional<int> pooling;
    std::optional<bool> lowercase;
    bool remap_ids = false;
    bool unsigned_ids = false;
    bool write_epsilon = true;
    bool token_metadata = true;
    std::string model_name;
};

static void write_fixture(const fs::path& file, const Options& options) {
    ggml_context* weights = ggml_init({1 << 20, nullptr, false});
    gguf_context* meta = gguf_init_empty();
    require(weights && meta, "allocate tiny exported model");
    const std::string prefix = options.arch + ".";
    gguf_set_val_str(meta, "general.architecture", options.arch.c_str());
    gguf_set_val_str(meta, "general.name", options.model_name.empty() ?
                        (options.roberta ? "BGE-M3 fixture" : "MiniLM fixture") : options.model_name.c_str());
    for (const auto& pair : std::vector<std::pair<std::string, uint32_t>>{
         {"context_length", 18}, {"embedding_length", dim}, {"feed_forward_length", dim},
         {"attention.head_count", 1}, {"block_count", 1}})
        gguf_set_val_u32(meta, (prefix + pair.first).c_str(), pair.second);
    if (options.write_epsilon) gguf_set_val_f32(meta, (prefix + "attention.layer_norm_epsilon").c_str(), 1e-5f);
    if (options.pooling) gguf_set_val_u32(meta, (prefix + "pooling_type").c_str(), *options.pooling);
    if (options.lowercase) gguf_set_val_bool(meta, "tokenizer.ggml.do_lower_case", *options.lowercase);
    std::vector<const char*> tokens = options.roberta ? std::vector<const char*>{
        "<s>", "<pad>", "</s>", "<unk>", "▁Searle", "▁Derrida", "▁searle", "▁think", "▁John", "▁john", "▁Derr", "ida"} :
        std::vector<const char*>{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "searle", "derrida", "think", "john", "##s", "Searle", "Derrida", "▁"};
    const int cls = options.roberta ? 0 : 2, sep = options.roberta ? 2 : 3;
    const int unk = options.roberta ? 3 : 1, pad = options.roberta ? 1 : 0;
    if (options.token_metadata) {
        gguf_set_val_u32(meta, "tokenizer.ggml.bos_token_id", cls);
        gguf_set_val_u32(meta, "tokenizer.ggml.eos_token_id", sep);
        gguf_set_val_u32(meta, "tokenizer.ggml.unknown_token_id", unk);
        gguf_set_val_u32(meta, "tokenizer.ggml.padding_token_id", pad);
    }
    if (options.remap_ids) {
        std::vector<int32_t> ids(vocabulary_size);
        std::iota(ids.begin(), ids.end(), 0);
        std::swap(ids[0], ids[4]); std::swap(tokens[0], tokens[4]);
        std::swap(ids[2], ids[6]); std::swap(tokens[2], tokens[6]);
        gguf_set_arr_data(meta, "tokenizer.ggml.token_ids", options.unsigned_ids ? GGUF_TYPE_UINT32 : GGUF_TYPE_INT32,
                          ids.data(), ids.size());
    }
    gguf_set_arr_str(meta, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    auto tensor = [&](const std::string& name, int rows, int columns, float value = 0) {
        ggml_tensor* t = columns == 1 ? ggml_new_tensor_1d(weights, GGML_TYPE_F32, rows) :
                                       ggml_new_tensor_2d(weights, GGML_TYPE_F32, rows, columns);
        ggml_set_name(t, name.c_str());
        std::fill_n(static_cast<float*>(t->data), rows * columns, value);
        gguf_add_tensor(meta, t);
        return static_cast<float*>(t->data);
    };
    float* word = tensor("token_embd.weight", dim, vocabulary_size);
    for (int token = 0; token < vocabulary_size; ++token)
        for (int i = 0; i < dim; ++i) word[token * dim + i] = word_value(token, i);
    tensor("token_types.weight", dim, 1);
    float* positions = tensor("position_embd.weight", dim, position_rows);
    for (int position = 0; position < position_rows; ++position)
        for (int i = 0; i < dim; ++i) positions[position * dim + i] = position_value(position, i);
    tensor("token_embd_norm.weight", dim, 1, 1); tensor("token_embd_norm.bias", dim, 1);
    for (const std::string name : {"attn_output_norm", "layer_output_norm"}) {
        tensor("blk.0." + name + ".weight", dim, 1, 1);
        tensor("blk.0." + name + ".bias", dim, 1);
    }
    for (const std::string name : {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_up", "ffn_down"}) {
        tensor("blk.0." + name + ".weight", dim, dim);
        tensor("blk.0." + name + ".bias", dim, 1);
    }
    require(gguf_write_to_file(meta, file.c_str(), false), "write tiny GGUF model");
    gguf_free(meta); ggml_free(weights);
}

static std::vector<float> reference(const bert_tokens& tokens, bool roberta, int pooling) {
    std::vector<float> result(dim, 0);
    for (size_t position = 0; position < tokens.size(); ++position) {
        std::vector<float> value(dim);
        for (int i = 0; i < dim; ++i)
            value[i] = word_value(tokens[position], i) + position_value(int(position) + (roberta ? 2 : 0), i);
        // Embedding LN followed by the two residual LNs in our zero-projection layer.
        for (int stage = 0; stage < 3; ++stage) {
            const double mean = std::accumulate(value.begin(), value.end(), 0.0) / dim;
            double variance = 0;
            for (float v : value) variance += (v - mean) * (v - mean) / dim;
            for (auto& v : value) v = float((v - mean) / std::sqrt(variance + 1e-5));
        }
        const float weight = pooling == 2 ? float(position == 0) :
                             pooling == 3 ? float(position + 1 == tokens.size()) : 1.0f / tokens.size();
        for (int i = 0; i < dim; ++i) result[i] += value[i] * weight;
    }
    double norm = 0;
    for (float v : result) norm += double(v) * v;
    if (norm > 0) for (auto& v : result) v /= float(std::sqrt(norm));
    return result;
}

static void run_case(const fs::path& directory, int number, const Options& options) {
    const fs::path path = directory / (std::to_string(number) + ".gguf");
    write_fixture(path, options);
    Model model(bert_load_from_file(path.c_str(), true), bert_free);
    require(bool(model), "load fixture through production GGUF loader");
    const bool lowercase = options.lowercase.value_or(!options.roberta);
    const bool bge = options.model_name.empty() ? options.roberta : options.model_name.find("bge") != std::string::npos;
    const int pooling = options.pooling.value_or(bge ? 2 : 1);
    require(model->embedding.lowercase == lowercase && int(model->embedding.pooling) == pooling,
            "metadata and family defaults select casing and pooling");
    require(std::abs(model->model.hparams.layer_norm_eps - 1e-5f) < 1e-10f, "model-specific layer-norm epsilon");
    const int cls = options.roberta ? 0 : 2, sep = options.roberta ? 2 : 3;
    const int unk = options.roberta ? 3 : 1;
    require(model->embedding.cls_token == cls && model->embedding.sep_token == sep && model->embedding.unk_token == unk,
            "special tokens come from model metadata and vocabulary");
    const bert_tokens expected = options.roberta ? bert_tokens{0, 4, 5, 2} :
                                  lowercase ? bert_tokens{2, 4, 5, 3} : bert_tokens{2, 9, 10, 3};
    const auto tokens = bert_tokenize(model.get(), "Searle Derrida", 16);
    require(tokens == expected, "casing and token IDs match the exported vocabulary");
    require(bert_tokenize(model.get(), "Absent", 16) == bert_tokens({cls, unk, sep}), "model-specific unknown token");
    require(bert_tokenize(model.get(), "Searle Derrida", 0).empty(), "zero token budget");
    require(bert_tokenize(model.get(), "Searle Derrida", 1).empty(), "insufficient special-token budget");
    require(bert_tokenize(model.get(), "Searle Derrida", 2) == bert_tokens({cls, sep}), "special-token-only budget");
    require(bert_tokenize(model.get(), "Searle Derrida", 3).size() == 3, "truncation reserves the separator");
    if (options.roberta) require(bert_n_max_tokens(model.get()) == position_rows - 2, "RoBERTa position-table capacity");
    bert_tokens shorter{cls, tokens[1], sep};
    for (const bert_batch& batch : {bert_batch{tokens}, bert_batch{shorter, tokens}, bert_batch{tokens, shorter}}) {
        std::vector<float> actual(batch.size() * dim);
        bert_forward_batch(model.get(), batch, actual.data(), 1);
        for (size_t row = 0; row < batch.size(); ++row) {
            const auto wanted = reference(batch[row], options.roberta, pooling);
            for (int i = 0; i < dim; ++i)
                require(std::isfinite(actual[row * dim + i]) && std::abs(actual[row * dim + i] - wanted[i]) < 0.0001f,
                        "production pooling/position graph matches the independent float reference");
        }
    }
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--export-fixture") {
        write_fixture(argv[2], Options{});
        return 0;
    }
    const auto directory = fs::current_path() / ("bert-profile-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    bool created = false;
    try {
        created = fs::create_directory(directory);
        require(created, "create fixture directory");
        Options m3; m3.write_epsilon = false;
        run_case(directory, 0, m3);
        m3.arch = "bert"; m3.pooling = 2; m3.remap_ids = true; m3.unsigned_ids = true;
        run_case(directory, 1, m3);
        m3.pooling = 1; run_case(directory, 2, m3);
        m3.pooling = 3; run_case(directory, 3, m3);
        Options bert; bert.roberta = false; bert.arch = "bert";
        run_case(directory, 4, bert);
        bert.lowercase = false; bert.remap_ids = true;
        run_case(directory, 5, bert);
        Options e5; e5.model_name = "multilingual-e5-fixture";
        run_case(directory, 6, e5);
        Options vocabulary_only; vocabulary_only.token_metadata = false;
        run_case(directory, 7, vocabulary_only);
        fs::remove_all(directory);
        std::cout << "Embedding metadata, tokens, pooling, positions and batch cache: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        if (created) fs::remove_all(directory);
        std::cerr << "Embedding profile regression: " << error.what() << '\n';
        return 1;
    }
}
