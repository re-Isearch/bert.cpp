// Inspect raw model embeddings before Schmate packing, HNSW or calibration.
#include "bert.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>

static double squared_norm(const std::vector<float>& vector) {
    double sum = 0;
    for (float value : vector) {
        if (!std::isfinite(value)) throw std::runtime_error("model produced a non-finite embedding");
        sum += double(value) * value;
    }
    return sum;
}
int main(int argc, char** argv) try {
    if (argc < 3) {
        std::cerr << "Usage: embedding_probe MODEL.gguf QUERY [passages.txt] [--cpu] [--threads N] [--pooling cls|mean|last]\n";
        return 1;
    }
    bool cpu = false;
    int threads = 4;
    std::optional<bert_pooling_type> override_pool;
    std::string passages;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cpu") cpu = true;
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (arg == "--pooling" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "cls") override_pool = bert_pooling_type::CLS;
            else if (value == "mean") override_pool = bert_pooling_type::MEAN;
            else if (value == "last") override_pool = bert_pooling_type::LAST;
            else throw std::runtime_error("pooling must be cls, mean or last");
        } else if (!arg.starts_with("--") && passages.empty()) passages = arg;
        else throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    if (threads < 1) throw std::runtime_error("threads must be positive");
    std::unique_ptr<bert_ctx, decltype(&bert_free)> model(bert_load_from_file(argv[1], cpu), bert_free);
    if (!model) throw std::runtime_error("cannot load model");
    if (override_pool) model->embedding.pooling = *override_pool;
    const auto& config = model->embedding;
    std::cout << "Raw unprefixed model embeddings; no index packing, HNSW or score calibration\n"
              << "model=" << bert_get_model_name(model.get()) << " arch=" << bert_get_architecture(model.get())
              << " dim=" << bert_n_embd(model.get()) << " pooling=" << int(config.pooling)
              << " lowercase=" << config.lowercase << " position_offset="
              << (config.roberta_positions ? config.pad_token + 1 : 0) << '\n';
    const auto tokens = bert_tokenize(model.get(), argv[2], bert_n_max_tokens(model.get()));
    std::cout << "query tokens:";
    for (int token : tokens) std::cout << ' ' << token << ':' << bert_vocab_id_to_token(model.get(), token);
    std::cout << '\n';
    std::vector<float> query(bert_n_embd(model.get()));
    bert_forward_batch(model.get(), {tokens}, query.data(), threads);
    const double query_norm = squared_norm(query);
    if (query_norm <= 0) throw std::runtime_error("query embedding is zero");
    std::vector<std::string> texts;
    if (passages.empty()) {
        texts = {
            "John Searle extended speech act theory by categorizing illocutionary acts into assertives, directives, commissives, expressives and declarations.",
            "John Searle's Chinese Room argument distinguishes symbol manipulation from genuine understanding.",
            "Great minds think alike in many cases.",
            "Donald Trump won the 2016 presidential election against Hillary Clinton.",
            "Japan's military government promoted ultranationalist ideology and prepared for total war.",
            "Keep your friends close and enemies closer."};
    } else {
        std::ifstream input(passages);
        if (!input) throw std::runtime_error("cannot open passages file");
        for (std::string line; std::getline(input, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) texts.push_back(std::move(line));
        }
    }
    if (texts.empty()) throw std::runtime_error("no passages to compare");
    struct Hit { double cosine, dot, norm; size_t row, unknown; };
    std::vector<Hit> results;
    for (size_t row = 0; row < texts.size(); ++row) {
        const auto encoded = bert_tokenize(model.get(), texts[row], bert_n_max_tokens(model.get()));
        std::vector<float> vector(query.size());
        bert_forward_batch(model.get(), {encoded}, vector.data(), threads);
        const double norm = squared_norm(vector);
        if (norm <= 0) throw std::runtime_error("passage embedding is zero");
        double dot = 0;
        for (size_t i = 0; i < query.size(); ++i) dot += double(query[i]) * vector[i];
        results.push_back({dot / std::sqrt(query_norm * norm), dot, std::sqrt(norm), row,
                          size_t(std::count(encoded.begin(), encoded.end(), config.unk_token))});
    }
    std::stable_sort(results.begin(), results.end(), [](const Hit& a, const Hit& b) { return a.cosine > b.cosine; });
    std::cout << std::setprecision(9) << "query_norm=" << std::sqrt(query_norm) << '\n';
    for (const auto& hit : results)
        std::cout << "cosine=" << hit.cosine << " dot=" << hit.dot << " norm=" << hit.norm
                  << " unknown=" << hit.unknown << " row=" << hit.row + 1 << " " << texts[hit.row] << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "embedding_probe: " << error.what() << '\n';
    return 1;
}
