#include "bert.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct bench_params {
    const char * model     = nullptr;
    int          n_threads = 6;
    int          warmup    = 5;
    int          iters     = 50;
    bool         use_cpu   = false;
};

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m MODEL [-c] [-t N] [-w N] [-n N]\n"
        "  -m  model path (gguf)\n"
        "  -c  CPU backend (default: accelerated)\n"
        "  -t  threads (default 6)\n"
        "  -w  warmup iterations (default 5)\n"
        "  -n  measured iterations (default 50)\n", argv0);
}

static bool parse(int argc, char ** argv, bench_params & p) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "-m") p.model     = argv[++i];
        else if (a == "-c") p.use_cpu   = true;
        else if (a == "-t") p.n_threads = std::stoi(argv[++i]);
        else if (a == "-w") p.warmup    = std::stoi(argv[++i]);
        else if (a == "-n") p.iters     = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") { usage(argv[0]); return false; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return false; }
    }
    return p.model != nullptr;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n/2] : 0.5*(v[n/2 - 1] + v[n/2]);
}

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)((v.size() - 1) * p);
    return v[idx];
}

static double mean(const std::vector<double> & v) {
    double s = 0; for (double x : v) s += x; return s / v.size();
}

// build a prompt of roughly `target_tokens` tokens (each filler word becomes ~1 token)
static std::string make_prompt(int target_tokens) {
    static const char * filler = "the quick brown fox jumps over the lazy dog ";
    std::string s;
    // -2 for [CLS]/[SEP]
    int words = std::max(1, target_tokens - 2);
    // each filler word is ~1 token; pad with single tokens
    s.reserve(words * 6);
    while ((int)s.size()/5 < words) s += filler;
    return s;
}

static void run_scenario(bert_ctx * ctx, const bench_params & p,
                         const char * label, int target_tokens, int batch_size) {
    std::string prompt = make_prompt(target_tokens);
    bert_tokens toks = bert_tokenize(ctx, prompt, bert_n_max_tokens(ctx));
    int actual_tokens = (int)toks.size();

    bert_batch batch(batch_size, toks);
    const int n_embd = bert_n_embd(ctx);
    std::vector<float> embed((size_t)batch_size * n_embd);

    // warmup
    for (int i = 0; i < p.warmup; i++) {
        bert_forward_batch(ctx, batch, embed.data(), p.n_threads);
    }

    // measure
    std::vector<double> ms;
    ms.reserve(p.iters);
    for (int i = 0; i < p.iters; i++) {
        int64_t t0 = ggml_time_us();
        bert_forward_batch(ctx, batch, embed.data(), p.n_threads);
        int64_t t1 = ggml_time_us();
        ms.push_back((t1 - t0) / 1000.0);
    }

    double m  = mean(ms);
    double md = median(ms);
    double p95= percentile(ms, 0.95);
    double mn = *std::min_element(ms.begin(), ms.end());
    double mx = *std::max_element(ms.begin(), ms.end());
    double tokens_per_call = (double)actual_tokens * batch_size;
    double tok_per_sec     = tokens_per_call / (m / 1000.0);
    double seq_per_sec     = (double)batch_size / (m / 1000.0);

    printf("  %-14s tok=%3d  bs=%3d  | mean=%7.2f ms  med=%7.2f  p95=%7.2f  min=%6.2f  max=%6.2f  | %8.0f tok/s  %7.1f seq/s\n",
           label, actual_tokens, batch_size, m, md, p95, mn, mx, tok_per_sec, seq_per_sec);
}

int main(int argc, char ** argv) {
    ggml_time_init();
    bench_params p;
    if (!parse(argc, argv, p)) return 1;

    printf("Loading %s on %s backend (threads=%d)...\n",
           p.model, p.use_cpu ? "CPU" : "accelerated", p.n_threads);
    int64_t tl0 = ggml_time_us();
    bert_ctx * ctx = bert_load_from_file(p.model, p.use_cpu);
    int64_t tl1 = ggml_time_us();
    if (!ctx) { fprintf(stderr, "failed to load model\n"); return 1; }
    printf("Model: %s  arch=%s  n_embd=%d  load=%.0f ms\n",
           std::string(bert_get_model_name(ctx)).c_str(),
           std::string(bert_get_architecture(ctx)).c_str(),
           bert_n_embd(ctx),
           (tl1 - tl0) / 1000.0);
    printf("Warmup=%d iters, measure=%d iters\n\n", p.warmup, p.iters);

    printf("== latency by input length (batch=1) ==\n");
    run_scenario(ctx, p, "very-short", 8,   1);
    run_scenario(ctx, p, "short",      32,  1);
    run_scenario(ctx, p, "medium",     128, 1);
    run_scenario(ctx, p, "long",       256, 1);
    run_scenario(ctx, p, "max",        512, 1);

    printf("\n== throughput by batch size (tok=128) ==\n");
    run_scenario(ctx, p, "bs=1",   128, 1);
    run_scenario(ctx, p, "bs=4",   128, 4);
    run_scenario(ctx, p, "bs=8",   128, 8);
    run_scenario(ctx, p, "bs=16",  128, 16);
    run_scenario(ctx, p, "bs=32",  128, 32);

    bert_free(ctx);
    return 0;
}
