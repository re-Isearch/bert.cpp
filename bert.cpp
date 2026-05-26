/*
BERT inference in GGML

Forked from:
https://github.com/ggerganov/bert.cpp

which itself was forked from:
https://github.com/skeskinen/bert.cpp
https://github.com/xyzhang626/embeddings.cpp

This has been heavily modified for the re-Isearch/Schmate project
to support the latest ggml library (0.12.0) of May 2026

Modernised for the current ggml-org/ggml API:
  - ggml_allocr_*              →  ggml_gallocr_*
  - GGML_USE_CUBLAS            →  GGML_USE_CUDA  (renamed upstream)
  - ggml_backend_metal_set_n_cb→  ggml_backend_metal_set_n_threads
  - gf->nodes[gf->n_nodes-1]  →  ggml_graph_node(gf, -1)
  - manual weight alloc loop   →  ggml_backend_alloc_ctx_tensors

At this time support for CUDA, Metal, Vulkan and CPU: 95% of use cases.

  - CUDA: Secures absolute peak performance on NVIDIA enterprise and consumer GPUs.
  - Metal: Absolute peak performance on Apple Silicon.
  - Vulkan: Covers AMD GPUs, Intel Arc GPUs, older NVIDIA cards, Android devices,
    and Raspberry Pi clusters. Performance is closing the gap with native backends,
    reducing the need for vendor-specific frameworks.
    See: https://www.youtube.com/watch?v=xaQkt3iWsTQ&t=1783s
    (Vulkanised 2026: Vulkan Machine Learning in ggml/llama.cpp)

What this interface does NOT support are some of the newer models such as
jina-embeddings-v5-omni-nano-text-matching which uses the EuroBERT design.

EuroBERT utilizes Rotary Position Embeddings (RoPE), which marks a significant
architectural departure from standard BERT.

EuroBERT replaces standard BERT's Multi-Head Attention (MHA) with Grouped-Query
Attention (GQA) to save memory and boost speed.  It swaps out original LayerNorm
for Root Mean Square Layer Normalization (RMSNorm), implements SwiGLU (Swish Gated
Linear Units) instead of BERT's traditional GELU activation and following modern
optimizations, all bias terms are entirely removed from the dense transformer layers.

While the original BERT uses absolute positional embeddings, EuroBERT incorporates
modern architectural choices closely mirroring Llama architectures. This structural
choice specifically impacts position handling, token length, and attention mechanisms.

For these use the llama.cpp interface.
*/

#include "bert.h"

#ifdef GGML_USE_CUDA
# include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
# include "ggml-metal.h"
#endif

#ifdef GGML_USE_VULKAN
# include "ggml-vulkan.h"
#endif

#include <cmath>
#include <fstream>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

#define BERT_MAX_NODES 4096
#define KEY_TOKEN_LIST "tokenizer.ggml.tokens"

static const int verbosity = 0;

extern "C" {
    void ggml_log_internal(enum ggml_log_level level, const char * format, ...);
};

// ---------------------------------------------------------------------------
// GGUF metadata helpers
// ---------------------------------------------------------------------------

static void print_all_metadata_keys(const struct gguf_context * ctx) {
    fprintf(stderr, "\n--- GGUF METADATA DISCOVERY ---\n");
    const int n = gguf_get_n_kv(ctx);
    for (int i = 0; i < n; ++i)
        fprintf(stderr, "  [%d] %s\n", i, gguf_get_key(ctx, i));
    fprintf(stderr, "-------------------------------\n");
}

static const char * get_string_c(const struct gguf_context * ctx, const std::string & key) {
    int key_id = gguf_find_key(ctx, key.c_str());
    if (key_id < 0 || gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_STRING)
        return nullptr;
    return gguf_get_val_str(ctx, key_id);
}

const std::string_view bert_get_model_name(const bert_ctx * ctx) {
    if (ctx) return ctx->model_name;
    return "";
}

const std::string_view bert_get_architecture(const bert_ctx * ctx) {
    if (ctx) return ctx->model_arch;
    return "";
}

static uint32_t get_u32_robust(const struct gguf_context * ctx,
                                std::vector<std::string> keys) {
    for (const auto & key : keys) {
        const int id = gguf_find_key(ctx, key.c_str());
        if (id == -1) continue;
        switch (gguf_get_kv_type(ctx, id)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
            case GGUF_TYPE_INT32:  return static_cast<uint32_t>(gguf_get_val_i32(ctx, id));
            case GGUF_TYPE_UINT64: return static_cast<uint32_t>(gguf_get_val_u64(ctx, id));
            case GGUF_TYPE_INT64:  return static_cast<uint32_t>(gguf_get_val_i64(ctx, id));
            default: break;
        }
    }
    fprintf(stderr, "Fatal: could not find any of the requested GGUF keys.\n");
    print_all_metadata_keys(ctx);
    fprintf(stderr, "NOTE: Model may work via the llama.cpp interface.\n");
    exit(1);
}

static float get_f32(const struct gguf_context * ctx, const char * key) {
    const int i = gguf_find_key(ctx, key);
    return (i == -1) ? 1e-12f : gguf_get_val_f32(ctx, i);
}

// ---------------------------------------------------------------------------
// Tensor lookup helpers
// ---------------------------------------------------------------------------

static struct ggml_tensor * get_tensor_exhaustive(struct ggml_context * ctx,
                                                   struct gguf_context * ctx_gguf,
                                                   std::vector<std::string> names) {
    for (const auto & name : names) {
        struct ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        if (t) return t;
    }
    fprintf(stderr, "\n--- TENSOR LOADING ERROR ---\n");
    fprintf(stderr, "Cannot find any of the requested variants. First 50 tensors in file:\n");
    const int n = gguf_get_n_tensors(ctx_gguf);
    for (int i = 0; i < std::min(n, 50); ++i)
        fprintf(stderr, "  [%d] %s\n", i, gguf_get_tensor_name(ctx_gguf, i));
    fprintf(stderr, "---------------------------\n");
    exit(1);
}

// ---------------------------------------------------------------------------
// Tokeniser
// ---------------------------------------------------------------------------

static size_t utf8_len(char src) {
    const size_t lookup[] = {1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,4};
    return lookup[static_cast<uint8_t>(src) >> 4];
}

static bool is_chinese_char(const std::string & s) {
    if (s.empty()) return false;
    unsigned int cp = 0; int nb = 0;
    const unsigned char c = static_cast<unsigned char>(s[0]);
    if      (c <= 0x7f)          { cp = c;        nb = 1; }
    else if ((c >> 5) == 0x06)   { cp = c & 0x1f; nb = 2; }
    else if ((c >> 4) == 0x0e)   { cp = c & 0x0f; nb = 3; }
    else if ((c >> 3) == 0x1e)   { cp = c & 0x07; nb = 4; }
    for (int j = 1; j < nb; ++j) {
        if (j >= (int)s.size()) return false;
        const unsigned char nc = static_cast<unsigned char>(s[j]);
        if ((nc >> 6) != 0x02) return false;
        cp = (cp << 6) | (nc & 0x3f);
    }
    return (cp>=0x4E00&&cp<=0x9FFF)||(cp>=0x3400&&cp<=0x4DBF)||
           (cp>=0x20000&&cp<=0x2A6DF)||(cp>=0x2A700&&cp<=0x2B73F)||
           (cp>=0x2B740&&cp<=0x2B81F)||(cp>=0x2B920&&cp<=0x2CEAF)||
           (cp>=0xF900&&cp<=0xFAFF)||(cp>=0x2F800&&cp<=0x2FA1F)||
           (cp>=0x3000&&cp<=0x303F)||(cp>=0xFF00&&cp<=0xFFEF);
}

static std::string strip_accents(const std::string & in) {
    static const std::map<std::string,char> am = {
        {"À",'A'},{"Á",'A'},{"Â",'A'},{"Ã",'A'},{"Ä",'A'},{"Å",'A'},
        {"à",'a'},{"á",'a'},{"â",'a'},{"ã",'a'},{"ä",'a'},{"å",'a'},
        {"È",'E'},{"É",'E'},{"Ê",'E'},{"Ë",'E'},
        {"è",'e'},{"é",'e'},{"ê",'e'},{"ë",'e'},
        {"Ì",'I'},{"Í",'I'},{"Î",'I'},{"Ï",'I'},
        {"ì",'i'},{"í",'i'},{"î",'i'},{"ï",'i'},
        {"Ò",'O'},{"Ó",'O'},{"Ô",'O'},{"Õ",'O'},{"Ö",'O'},
        {"ò",'o'},{"ó",'o'},{"ô",'o'},{"õ",'o'},{"ö",'o'},
        {"Ù",'U'},{"Ú",'U'},{"Û",'U'},{"Ü",'U'},
        {"ù",'u'},{"ú",'u'},{"û",'u'},{"ü",'u'},
        {"Ý",'Y'},{"ý",'y'},{"Ç",'C'},{"ç",'c'},{"Ñ",'N'},{"ñ",'n'},
    };
    std::string out;
    for (size_t i = 0; i < in.size(); ) {
        const int len = utf8_len(in[i]);
        const std::string ch = in.substr(i, len);
        auto it = am.find(ch);
        out += (it != am.end()) ? std::string(1, it->second) : ch;
        i += len;
    }
    return out;
}

static std::string bert_normalize_prompt(const std::string & text) {
    std::string out = strip_accents(text);
    for (size_t i = 0; i < out.size(); i += utf8_len(out[i])) {
        char c = out[i];
        if (c >= 'A' && c <= 'Z') out[i] = c - 'A' + 'a';
    }
    return out;
}


// WordPiece 
bert_tokens bert_tokenize_wordpiece(bert_ctx * ctx, bert_string text, uint64_t n_max_tokens) {
    constexpr int CLS = 101, SEP = 102, UNK = 100;
    const bert_vocab & vocab = ctx->vocab;

    //fprintf(stderr, "[BERT_TOKENIZE] ctx=%p vocab.token_to_id.size=%zu\n", (void*)ctx, ctx->vocab.token_to_id.size());

    // Normalize: lowercase + strip accents
    std::string ori = bert_normalize_prompt(text);

    // Insert spaces around punctuation and Chinese characters
    std::string spaced;
    for (size_t i = 0; i < ori.size(); ) {
        const int cl = utf8_len(ori[i]);
        if (cl == 1 && ispunct(ori[i])) {
            spaced += ' '; spaced += ori[i]; spaced += ' '; i++;
        } else if (cl == 3 && is_chinese_char(ori.substr(i, 3))) {
            spaced += ' '; spaced += ori.substr(i, 3); spaced += ' '; i += 3;
        } else {
            spaced += ori[i++];
        }
    }

    // Split on whitespace into words
    std::vector<std::string> words;
    for (size_t l = 0, r = 0; r <= spaced.size(); ++r) {
        if (r == spaced.size() || isspace(spaced[r])) {
            if (r > l) words.push_back(spaced.substr(l, r - l));
            l = r + 1;
        }
    }

    bert_tokens tokens;
    tokens.push_back(CLS);

    // ---------------------------------------------------------------------------
    // WordPiece tokenizer
    //
    // Vocabulary layout (set by bert_load_from_file):
    //   vocab.token_to_id         — root tokens, keys WITHOUT "##"  e.g. "who", "mat"
    //   vocab.subword_token_to_id — subword tokens, keys WITH "##"  e.g. "##ho", "##tion"
    //
    // Algorithm: for each word, greedily match the longest prefix from the
    // root map (i==0) or the subword map with "##"+fragment (i>0).
    // If any fragment cannot be matched, replace the whole word with [UNK].
    // ---------------------------------------------------------------------------
    for (const auto & word : words) {
        const int n = (int)word.size();
        if (!n) continue;

        int  i               = 0;
        bool word_ok         = true;
        const size_t start_tokens_idx = tokens.size();

        while (i < n) {
            if ((int)tokens.size() >= (int)n_max_tokens - 1) break;

            int matched_len = 0;

            for (int j = n; j > i; --j) {
                const std::string substr = word.substr(i, j - i);

                if (i == 0) {
                    // Word-initial fragment: look up in root map as-is
                    auto it = vocab.token_to_id.find(substr);
                    if (it != vocab.token_to_id.end()) {
                        tokens.push_back(it->second);
                        matched_len = j - i;
                        break;
                    }
                } else {
                    // Mid-word fragment: prepend "##" and look up in subword map
                    auto it = vocab.subword_token_to_id.find("##" + substr);
                    if (it != vocab.subword_token_to_id.end()) {
                        tokens.push_back(it->second);
                        matched_len = j - i;
                        break;
                    }
                }
            }

            if (matched_len > 0) {
                i += matched_len;
            } else {
                word_ok = false;
                break;
            }
        }

        if (!word_ok) {
            // Roll back any partial tokens for this word and emit [UNK]
            tokens.resize(start_tokens_idx);
            tokens.push_back(UNK);
        }
    }

    tokens.push_back(SEP);

    if (verbosity >= 1) {
        fprintf(stderr, "[TOKENIZER OUTPUT] ");
        for (int id : tokens) {
            auto it = vocab._id_to_token.find(id);
            if (it != vocab._id_to_token.end())
                fprintf(stderr, "'%s'(%d) ", it->second.c_str(), id);
            else {
                auto it2 = vocab._id_to_subword_token.find(id);
                if (it2 != vocab._id_to_subword_token.end())
                    fprintf(stderr, "'%s'(%d) ", it2->second.c_str(), id);
                else
                    fprintf(stderr, "[UNK](%d) ", id);
            }
        }
        fprintf(stderr, "\n");
    }

    return tokens;
}

uint64_t bert_tokenize_c(bert_ctx * ctx, const char * text,
                          int32_t * output, uint64_t n_max_tokens) {
    bert_tokens toks = bert_tokenize(ctx, text, n_max_tokens);
    const uint64_t n = std::min((uint64_t)toks.size(), n_max_tokens);
    for (uint64_t i = 0; i < n; ++i) output[i] = toks[i];
    return n;
}

// ---------------------------------------------------------------------------
// GPU selection
// ---------------------------------------------------------------------------

static int get_best_gpu_index(const std::string & backend_type) {
    int   best_idx = 0;
    size_t max_vram = 0;

#ifdef GGML_USE_CUDA
    if (backend_type == "CUDA") {
        const int count = ggml_backend_cuda_get_device_count();
        for (int i = 0; i < count; ++i) {
            size_t free, total;
            char desc[256];
            ggml_backend_cuda_get_device_description(i, desc, sizeof(desc));
            ggml_backend_cuda_get_device_memory(i, &free, &total);
            ggml_log_internal(GGML_LOG_LEVEL_INFO,
                "CUDA Device %d: %s (%.2f GB VRAM)\n", i, desc, total / 1e9);
            if (total > max_vram) { max_vram = total; best_idx = i; }
        }
    }
#endif

#ifdef GGML_USE_VULKAN
    if (backend_type == "VULKAN") {
        const int count = ggml_backend_vk_get_device_count();
        for (int i = 0; i < count; ++i) {
            size_t free, total;
            char desc[256];
            ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
            ggml_backend_vk_get_device_memory(i, &free, &total);
            ggml_log_internal(GGML_LOG_LEVEL_INFO,
                "Vulkan Device %d: %s | Total VRAM: %.2f GB\n",
                i, desc, total / 1024.0 / 1024.0 / 1024.0);
            if (total > max_vram) { max_vram = total; best_idx = i; }
        }
    }
#endif

    (void)backend_type; // suppress warning when no GPU backends compiled
    return best_idx;
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------



// =============================================================================
//
// The all-MiniLM-L6-v2-Q4_K_M.gguf uses a SentencePiece-style vocabulary:
//   - Word-initial tokens are stored with a ▁ prefix (U+2581, \xe2\x96\x81)
//     e.g. "who" is stored as "▁who" at ID 2040
//   - Continuation fragments are stored bare, no ## prefix
//     e.g. "ho" at ID 6806
//   - Special tokens [PAD],[UNK],[CLS],[SEP],[MASK] are stored as-is
//
// The tokenizer therefore:
//   1. Lowercases input
//   2. Splits on whitespace
//   3. For each word, greedily matches longest prefix from token_to_id
//      using "▁" + fragment for the first piece, bare fragment for the rest
// =============================================================================

// ---------------------------------------------------------------------------
// bert_tokenize  (replaces the WordPiece version)
// ---------------------------------------------------------------------------
bert_tokens bert_tokenize(bert_ctx * ctx, bert_string text, uint64_t n_max_tokens) {
    constexpr int CLS = 101, SEP = 102, UNK = 100;
    const bert_vocab & vocab = ctx->vocab;
    const std::string SP = "\xe2\x96\x81"; // UTF-8 encoding of ▁ (U+2581)

    if (!vocab.is_sentencepiece) {
      // Use the old wordpiece tokenizer
      return bert_tokenize_wordpiece(ctx, text, n_max_tokens);
    } 

    // Lowercase only — SentencePiece does not strip accents
    std::string normalized;
    for (size_t i = 0; i < text.size(); ) {
        const int cl = utf8_len(text[i]);
        if (cl == 1) {
            normalized += (char)tolower((unsigned char)text[i]);
            i++;
        } else {
            normalized += text.substr(i, cl);
            i += cl;
        }
    }

    // Split on whitespace
    std::vector<std::string> words;
    for (size_t l = 0, r = 0; r <= normalized.size(); ++r) {
        if (r == normalized.size() || isspace((unsigned char)normalized[r])) {
            if (r > l) words.push_back(normalized.substr(l, r - l));
            l = r + 1;
        }
    }

    bert_tokens tokens;
    tokens.push_back(CLS);

    // -------------------------------------------------------------------------
    // SentencePiece greedy tokenizer
    //
    // For each word:
    //   i==0 : look up  "▁" + substr  in token_to_id
    //   i >0 : look up  substr        in token_to_id  (bare, no prefix)
    // If any fragment has no match, emit [UNK] for the whole word.
    // -------------------------------------------------------------------------
    for (const auto & word : words) {
        const int n = (int)word.size();
        if (!n) continue;

        int    i         = 0;
        bool   word_ok   = true;
        const size_t start_idx = tokens.size();

        while (i < n) {
            if ((int)tokens.size() >= (int)n_max_tokens - 1) break;

            int matched_len = 0;

            for (int j = n; j > i; --j) {
                const std::string candidate = (i == 0)
                    ? SP + word.substr(0, j)        // word-initial: prepend ▁
                    : word.substr(i, j - i);        // continuation: bare

                auto it = vocab.token_to_id.find(candidate);
                if (it != vocab.token_to_id.end()) {
                    tokens.push_back(it->second);
                    matched_len = j - i;
                    break;
                }
            }

            if (matched_len > 0) {
                i += matched_len;
            } else {
                word_ok = false;
                break;
            }
        }

        if (!word_ok) {
            tokens.resize(start_idx);
            tokens.push_back(UNK);
        }
    }

    tokens.push_back(SEP);

    // Debug output — set verbosity >= 1 to enable, or always print during testing
    if ( verbosity > 1) {
      fprintf(stderr, "[TOKENIZER OUTPUT] ");
      for (int id : tokens) {
        auto it = vocab._id_to_token.find(id);
        fprintf(stderr, "'%s'(%d) ",
            it != vocab._id_to_token.end() ? it->second.c_str() : "[?]", id);
      }
      fprintf(stderr, "\n");
    }

    return tokens;
}

// ---------------------------------------------------------------------------
// bert_load_from_file  (replaces the old version with ## WordPiece logic)
// ---------------------------------------------------------------------------
BERT_API struct bert_ctx * bert_load_from_file(const char * fname, bool use_cpu) {
    // fprintf(stderr, "[BERT_LOAD] bert_load_from_file called: %s\n", fname);

    struct ggml_context  * ctx_ggml = nullptr;
    struct gguf_init_params gp = { /*.no_alloc=*/true, /*.ctx=*/&ctx_ggml };
    struct gguf_context  * ctx_gguf = gguf_init_from_file(fname, gp);
    if (!ctx_gguf) {
        ggml_log_internal(GGML_LOG_LEVEL_ERROR,
            "%s: failed to open '%s'\n", __func__, fname);
        return nullptr;
    }

    bert_ctx   * bctx  = new bert_ctx{};
    bert_hparams & hp   = bctx->model.hparams;
    bert_vocab   & vocab = bctx->vocab;

    // Model identity
    {
        const char * model = get_string_c(ctx_gguf, "general.name");
        if (model) bctx->model_name = model;
        const char * arch  = get_string_c(ctx_gguf, "general.architecture");
        if (arch)  bctx->model_arch = arch;
    }
    bctx->ctx_meta = ctx_gguf;

    // Hyperparameters
    {
        const int v_id = gguf_find_key(ctx_gguf, KEY_TOKEN_LIST);
        hp.n_vocab        = (v_id != -1) ? (uint32_t)gguf_get_arr_n(ctx_gguf, v_id) : 0;
        hp.n_max_tokens   = get_u32_robust(ctx_gguf, {"bert.context_length",
                                "qwen2.context_length", "llama.context_length",
                                "general.context_length"});
        hp.n_embd         = get_u32_robust(ctx_gguf, {"bert.embedding_length",
                                "qwen2.embedding_length", "general.embedding_length"});
        hp.n_intermediate = get_u32_robust(ctx_gguf, {"bert.feed_forward_length",
                                "qwen2.feed_forward_length", "general.feed_forward_length"});
        hp.n_head         = get_u32_robust(ctx_gguf, {"bert.attention.head_count",
                                "qwen2.attention.head_count", "general.num_attention_heads"});
        hp.n_layer        = get_u32_robust(ctx_gguf, {"bert.block_count",
                                "qwen2.block_count", "general.block_count"});
        hp.layer_norm_eps = get_f32(ctx_gguf, "bert.attention.layer_norm_epsilon");
    }

    // -------------------------------------------------------------------------
    // Vocabulary — SentencePiece style
    //
    // ALL tokens go into token_to_id / _id_to_token keyed by their exact
    // stored string (which includes the ▁ prefix where present).
    // subword_token_to_id / _id_to_subword_token are left empty.
    // -------------------------------------------------------------------------
    {
        const int v_id = gguf_find_key(ctx_gguf, KEY_TOKEN_LIST);
        if (v_id != -1) {
            const int n = gguf_get_arr_n(ctx_gguf, v_id);
            vocab.tokens.resize(n);

	    // Detect vocab type by inspecting a mid-range token
	    // While the standard Ur-BERT was WordPiece most modern are using SentencePiece
	    const std::string probe = gguf_get_arr_str(ctx_gguf, v_id, 999);
	    bctx->vocab.is_sentencepiece = (probe.size() >= 3 && (unsigned char)probe[0] == 0xe2 &&
                (unsigned char)probe[1] == 0x96 && (unsigned char)probe[2] == 0x81);
	    ggml_log_internal(GGML_LOG_LEVEL_INFO, "%s: vocab type: %s (probe token 999='%s')\n",
                __func__, bctx->vocab.is_sentencepiece ? "SentencePiece (▁)" : "WordPiece (##)", probe.c_str());

	    // Load the vocabulary
            for (int i = 0; i < n; ++i) {
                const std::string tok = gguf_get_arr_str(ctx_gguf, v_id, i);
		if (bctx->vocab.is_sentencepiece) {
                  vocab.tokens[i]        = tok;
                  vocab.token_to_id[tok] = i;
                  vocab._id_to_token[i]  = tok;
		} else {
		  // WordPiece split
		  vocab.tokens[i]        = tok;
		  if (tok.size() > 2 && tok.substr(0, 2) == "##") {
		    vocab.subword_token_to_id[tok]      = i;
		    vocab._id_to_subword_token[i]       = tok;
		  } else {
		    vocab.token_to_id[tok] = i;
		    vocab._id_to_token[i]  = tok;
		  }
		} // end wordpiece
            }
        }
    }

    // Backend selection
#ifdef GGML_USE_CUDA
    if (!use_cpu) {
        int best = get_best_gpu_index("CUDA");
        bctx->backend = ggml_backend_cuda_init(best);
        if (bctx->backend)
            ggml_log_internal(GGML_LOG_LEVEL_INFO,
                "%s: using CUDA device %d\n", __func__, best);
    }
#endif
#ifdef GGML_USE_METAL
    if (!use_cpu && !bctx->backend) {
        bctx->backend = ggml_backend_metal_init();
        if (bctx->backend)
            ggml_log_internal(GGML_LOG_LEVEL_INFO,
                "%s: using Metal backend\n", __func__);
    }
#endif
#ifdef GGML_USE_VULKAN
    if (!use_cpu && !bctx->backend) {
        int best = get_best_gpu_index("VULKAN");
        bctx->backend = ggml_backend_vk_init(best);
        if (bctx->backend)
            ggml_log_internal(GGML_LOG_LEVEL_INFO,
                "%s: using Vulkan backend, device %d\n", __func__, best);
    }
#endif
    if (!bctx->backend) {
        bctx->backend = ggml_backend_cpu_init();
        ggml_log_internal(GGML_LOG_LEVEL_INFO,
            "%s: using CPU backend\n", __func__);
    }

    // ggml context for tensor metadata (shapes/names only, no data)
    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
    {
        struct ggml_init_params p = {
            static_cast<size_t>(n_tensors + 1) * ggml_tensor_overhead(),
            nullptr, true
        };
        bctx->ctx_data = ggml_init(p);
    }

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * src = ggml_get_tensor(ctx_ggml, name);
        struct ggml_tensor * dst = ggml_dup_tensor(bctx->ctx_data, src);
        ggml_set_name(dst, name);
    }

    // Allocate one contiguous backend buffer for all weights
    bctx->weights_buffer = ggml_backend_alloc_ctx_tensors(bctx->ctx_data, bctx->backend);
    if (!bctx->weights_buffer) {
        ggml_log_internal(GGML_LOG_LEVEL_ERROR,
            "%s: failed to allocate weights buffer\n", __func__);
        bert_free(bctx);
        return nullptr;
    }

    // Read weight data from file
    {
        std::ifstream fin(fname, std::ios::binary);
        if (!fin) {
            ggml_log_internal(GGML_LOG_LEVEL_ERROR,
                "%s: cannot open '%s'\n", __func__, fname);
            bert_free(bctx);
            return nullptr;
        }
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            struct ggml_tensor * cur = ggml_get_tensor(bctx->ctx_data, name);
            fin.seekg(static_cast<std::streamoff>(
                gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i)),
                std::ios::beg);
            if (ggml_backend_buffer_is_host(bctx->weights_buffer)) {
                fin.read(reinterpret_cast<char *>(cur->data), ggml_nbytes(cur));
            } else {
                std::vector<uint8_t> buf(ggml_nbytes(cur));
                fin.read(reinterpret_cast<char *>(buf.data()), ggml_nbytes(cur));
                ggml_backend_tensor_set(cur, buf.data(), 0, ggml_nbytes(cur));
            }
        }
    }

    // Map model weight pointers
    bert_model & model = bctx->model;
    {
        model.word_embeddings = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
            {"token_embd.weight",
             "bert.embeddings.word_embeddings.weight",
             "embeddings.word_embeddings.weight"});
        model.token_type_embeddings = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
            {"token_types.weight",
             "bert.embeddings.token_type_embeddings.weight",
             "embeddings.token_type_embeddings.weight"});
        model.position_embeddings = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
            {"position_embd.weight",
             "bert.embeddings.position_embeddings.weight",
             "embeddings.position_embeddings.weight"});
        model.ln_e_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
            {"token_embd_norm.weight",
             "bert.embeddings.LayerNorm.weight",
             "embeddings.LayerNorm.weight"});
        model.ln_e_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
            {"token_embd_norm.bias",
             "bert.embeddings.LayerNorm.bias",
             "embeddings.LayerNorm.bias"});

        model.layers.resize(hp.n_layer);
        for (int i = 0; i < (int)hp.n_layer; ++i) {
            bert_layer & layer = model.layers[i];
            const std::string p = "blk." + std::to_string(i) + ".";
            const std::string s = "encoder.layer." + std::to_string(i) + ".";

            layer.ln_att_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_output_norm.weight", p+"attn_out_norm.weight",
                 p+"attn_norm.weight",        p+"ln_1.weight",
                 s+"attention.output.LayerNorm.weight"});
            layer.ln_att_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_output_norm.bias",   p+"attn_out_norm.bias",
                 p+"attn_norm.bias",          p+"ln_1.bias",
                 s+"attention.output.LayerNorm.bias"});
            layer.ln_out_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"layer_output_norm.weight", p+"ffn_norm.weight",
                 p+"ln_2.weight",              s+"output.LayerNorm.weight"});
            layer.ln_out_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"layer_output_norm.bias",   p+"ffn_norm.bias",
                 p+"ln_2.bias",                s+"output.LayerNorm.bias"});

            layer.q_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_q.weight", s+"attention.self.query.weight"});
            layer.q_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_q.bias",   s+"attention.self.query.bias"});
            layer.k_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_k.weight", s+"attention.self.key.weight"});
            layer.k_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_k.bias",   s+"attention.self.key.bias"});
            layer.v_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_v.weight", s+"attention.self.value.weight"});
            layer.v_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_v.bias",   s+"attention.self.value.bias"});
            layer.o_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_output.weight", p+"attn_out.weight",
                 s+"attention.output.dense.weight"});
            layer.o_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_output.bias",   p+"attn_out.bias",
                 s+"attention.output.dense.bias"});
            layer.ff_i_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"ffn_up.weight",   s+"intermediate.dense.weight"});
            layer.ff_i_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"ffn_up.bias",     s+"intermediate.dense.bias"});
            layer.ff_o_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"ffn_down.weight", s+"output.dense.weight"});
            layer.ff_o_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"ffn_down.bias",   s+"output.dense.bias"});
        }
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_ggml);
    return bctx;
}




BERT_API struct bert_ctx * bert_load_from_file_wordpiece(const char * fname, bool use_cpu) {
    struct ggml_context  * ctx_ggml = nullptr;
    struct gguf_init_params gp = { /*.no_alloc=*/true, /*.ctx=*/&ctx_ggml };
    struct gguf_context  * ctx_gguf = gguf_init_from_file(fname, gp);
    if (!ctx_gguf) {
        ggml_log_internal(GGML_LOG_LEVEL_ERROR,
            "%s: failed to open '%s'\n", __func__, fname);
        return nullptr;
    }

    // fprintf(stderr, "[BERT_LOAD] bert_load_from_file_wordpiece called: %s\n", fname);

    bert_ctx   * bctx  = new bert_ctx{};
    bert_hparams & hp    = bctx->model.hparams;
    bert_vocab   & vocab = bctx->vocab;

    // Model identity
    {
        const char * model = get_string_c(ctx_gguf, "general.name");
        if (model) bctx->model_name = model;
        const char * arch  = get_string_c(ctx_gguf, "general.architecture");
        if (arch)  bctx->model_arch = arch;
    }
    bctx->ctx_meta = ctx_gguf;

    // Hyperparameters
    {
        const int v_id = gguf_find_key(ctx_gguf, KEY_TOKEN_LIST);
        hp.n_vocab        = (v_id != -1) ? (uint32_t)gguf_get_arr_n(ctx_gguf, v_id) : 0;
        hp.n_max_tokens   = get_u32_robust(ctx_gguf, {"bert.context_length",
                                "qwen2.context_length", "llama.context_length",
                                "general.context_length"});
        hp.n_embd         = get_u32_robust(ctx_gguf, {"bert.embedding_length",
                                "qwen2.embedding_length", "general.embedding_length"});
        hp.n_intermediate = get_u32_robust(ctx_gguf, {"bert.feed_forward_length",
                                "qwen2.feed_forward_length", "general.feed_forward_length"});
        hp.n_head         = get_u32_robust(ctx_gguf, {"bert.attention.head_count",
                                "qwen2.attention.head_count", "general.num_attention_heads"});
        hp.n_layer        = get_u32_robust(ctx_gguf, {"bert.block_count",
                                "qwen2.block_count", "general.block_count"});
        hp.layer_norm_eps = get_f32(ctx_gguf, "bert.attention.layer_norm_epsilon");

#if 1
	// Detect vocab type by inspecting a mid-range token
	const std::string probe = gguf_get_arr_str(ctx_gguf, v_id, 999);
	bctx->vocab.is_sentencepiece = (probe.size() >= 3 && (unsigned char)probe[0] == 0xe2 &&
		(unsigned char)probe[1] == 0x96 && (unsigned char)probe[2] == 0x81);
	ggml_log_internal(GGML_LOG_LEVEL_INFO, "%s: vocab type: %s (probe token 999='%s')\n",
		__func__, bctx->vocab.is_sentencepiece ? "SentencePiece (▁)" : "WordPiece (##)", probe.c_str());
#endif
    }



    // ---------------------------------------------------------------------------
    // Vocabulary
    //
    // Some GGUF files store tokens in arbitrary order with a separate
    // "tokenizer.ggml.token_ids" array mapping array-position → true vocab ID.
    // We must use true_id (not array index i) as the map key, otherwise common
    // words like "who" end up at wrong IDs and the tokenizer produces all [UNK].
    // ---------------------------------------------------------------------------
    {
        const int v_id = gguf_find_key(ctx_gguf, KEY_TOKEN_LIST);
        if (v_id != -1) {
            const int n = gguf_get_arr_n(ctx_gguf, v_id);
            vocab.tokens.resize(n);

            const int ids_key = gguf_find_key(ctx_gguf, "tokenizer.ggml.token_ids");
#if 0
            fprintf(stderr, "%s: tokenizer.ggml.token_ids: %s\n",
		__func__, ids_key != -1 ? "PRESENT (using true IDs)" : "absent (using array index as ID)");

            for (int i = 0; i < 5 && i < n; ++i)
                fprintf(stderr, "%s: array[%d] = '%s'\n",
                    __func__, i, gguf_get_arr_str(ctx_gguf, v_id, i));
#endif

            for (int i = 0; i < n; ++i) {
                const std::string tok = gguf_get_arr_str(ctx_gguf, v_id, i);

                // Use the true vocab ID from the remapping array if present,
                // otherwise the array index IS the vocab ID (standard BERT GGUF).
		const int true_id = (ids_key != -1)
                    ? ((const int32_t *)gguf_get_arr_data(ctx_gguf, ids_key))[i]
                    : i;

                if (true_id < n) vocab.tokens[true_id] = tok;

                if (tok.size() > 2 && tok.substr(0, 2) == "##") {
                    vocab.subword_token_to_id[tok]       = true_id;
                    vocab._id_to_subword_token[true_id]  = tok;
                } else {
                    vocab.token_to_id[tok]      = true_id;
                    vocab._id_to_token[true_id] = tok;
                }
            }

            fprintf(stderr, "%s: vocab loaded: %d tokens (%d root, %d subword)\n",
                __func__, n,
                (int)vocab.token_to_id.size(),
                (int)vocab.subword_token_to_id.size());

#if 0
            // Sanity check — these must exist in any BERT WordPiece vocab
            for (const char * s : {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "who", "##ho"}) {
                auto it1 = vocab.token_to_id.find(s);
                auto it2 = vocab.subword_token_to_id.find(s);
                fprintf(stderr, "%s: sanity '%s' -> %s\n", __func__, s,
                    it1 != vocab.token_to_id.end()         ? std::to_string(it1->second).c_str() :
                    it2 != vocab.subword_token_to_id.end() ? std::to_string(it2->second).c_str() :
                    "MISSING");
            }
#endif
        }
    }

    // Backend selection
#ifdef GGML_USE_CUDA
    if (!use_cpu) {
        int best = get_best_gpu_index("CUDA");
        bctx->backend = ggml_backend_cuda_init(best);
        if (bctx->backend)
            ggml_log_internal(GGML_LOG_LEVEL_INFO,
                "%s: using CUDA device %d\n", __func__, best);
    }
#endif
#ifdef GGML_USE_METAL
    if (!use_cpu && !bctx->backend) {
        bctx->backend = ggml_backend_metal_init();
        if (bctx->backend)
            ggml_log_internal(GGML_LOG_LEVEL_INFO,
                "%s: using Metal backend\n", __func__);
    }
#endif
#ifdef GGML_USE_VULKAN
    if (!use_cpu && !bctx->backend) {
        int best = get_best_gpu_index("VULKAN");
        bctx->backend = ggml_backend_vk_init(best);
        if (bctx->backend)
            ggml_log_internal(GGML_LOG_LEVEL_INFO,
                "%s: using Vulkan backend, device %d\n", __func__, best);
    }
#endif
    if (!bctx->backend) {
        bctx->backend = ggml_backend_cpu_init();
        ggml_log_internal(GGML_LOG_LEVEL_INFO,
            "%s: using CPU backend\n", __func__);
    }

    // ggml context for tensor metadata (shapes/names only, no data)
    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
    {
        struct ggml_init_params p = {
            static_cast<size_t>(n_tensors + 1) * ggml_tensor_overhead(),
            nullptr, true
        };
        bctx->ctx_data = ggml_init(p);
    }

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * src = ggml_get_tensor(ctx_ggml, name);
        struct ggml_tensor * dst = ggml_dup_tensor(bctx->ctx_data, src);
        ggml_set_name(dst, name);
    }

    // Allocate one contiguous backend buffer for all weights
    bctx->weights_buffer = ggml_backend_alloc_ctx_tensors(bctx->ctx_data, bctx->backend);
    if (!bctx->weights_buffer) {
        ggml_log_internal(GGML_LOG_LEVEL_ERROR,
            "%s: failed to allocate weights buffer\n", __func__);
        bert_free(bctx);
        return nullptr;
    }

    // Read weight data from file
    {
        std::ifstream fin(fname, std::ios::binary);
        if (!fin) {
            ggml_log_internal(GGML_LOG_LEVEL_ERROR,
                "%s: cannot open '%s'\n", __func__, fname);
            bert_free(bctx);
            return nullptr;
        }
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            struct ggml_tensor * cur = ggml_get_tensor(bctx->ctx_data, name);
            fin.seekg(static_cast<std::streamoff>(
                gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i)),
                std::ios::beg);
            if (ggml_backend_buffer_is_host(bctx->weights_buffer)) {
                fin.read(reinterpret_cast<char *>(cur->data), ggml_nbytes(cur));
            } else {
                std::vector<uint8_t> buf(ggml_nbytes(cur));
                fin.read(reinterpret_cast<char *>(buf.data()), ggml_nbytes(cur));
                ggml_backend_tensor_set(cur, buf.data(), 0, ggml_nbytes(cur));
            }
        }
    }

    // Map model weight pointers
    bert_model & model = bctx->model;
    {
        model.word_embeddings = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
            {"token_embd.weight",
             "bert.embeddings.word_embeddings.weight",
             "embeddings.word_embeddings.weight"});
        model.token_type_embeddings = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
            {"token_types.weight",
             "bert.embeddings.token_type_embeddings.weight",
             "embeddings.token_type_embeddings.weight"});
        model.position_embeddings = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
            {"position_embd.weight",
             "bert.embeddings.position_embeddings.weight",
             "embeddings.position_embeddings.weight"});
        model.ln_e_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
            {"token_embd_norm.weight",
             "bert.embeddings.LayerNorm.weight",
             "embeddings.LayerNorm.weight"});
        model.ln_e_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
            {"token_embd_norm.bias",
             "bert.embeddings.LayerNorm.bias",
             "embeddings.LayerNorm.bias"});

        model.layers.resize(hp.n_layer);
        for (int i = 0; i < (int)hp.n_layer; ++i) {
            bert_layer & layer = model.layers[i];
            const std::string p = "blk." + std::to_string(i) + ".";
            const std::string s = "encoder.layer." + std::to_string(i) + ".";

            layer.ln_att_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_output_norm.weight", p+"attn_out_norm.weight",
                 p+"attn_norm.weight",        p+"ln_1.weight",
                 s+"attention.output.LayerNorm.weight"});
            layer.ln_att_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_output_norm.bias",   p+"attn_out_norm.bias",
                 p+"attn_norm.bias",          p+"ln_1.bias",
                 s+"attention.output.LayerNorm.bias"});
            layer.ln_out_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"layer_output_norm.weight", p+"ffn_norm.weight",
                 p+"ln_2.weight",              s+"output.LayerNorm.weight"});
            layer.ln_out_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"layer_output_norm.bias",   p+"ffn_norm.bias",
                 p+"ln_2.bias",                s+"output.LayerNorm.bias"});

            layer.q_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_q.weight", s+"attention.self.query.weight"});
            layer.q_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_q.bias",   s+"attention.self.query.bias"});
            layer.k_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_k.weight", s+"attention.self.key.weight"});
            layer.k_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_k.bias",   s+"attention.self.key.bias"});
            layer.v_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_v.weight", s+"attention.self.value.weight"});
            layer.v_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_v.bias",   s+"attention.self.value.bias"});
            layer.o_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_output.weight", p+"attn_out.weight",
                 s+"attention.output.dense.weight"});
            layer.o_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"attn_output.bias",   p+"attn_out.bias",
                 s+"attention.output.dense.bias"});
            layer.ff_i_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"ffn_up.weight",   s+"intermediate.dense.weight"});
            layer.ff_i_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"ffn_up.bias",     s+"intermediate.dense.bias"});
            layer.ff_o_w = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"ffn_down.weight", s+"output.dense.weight"});
            layer.ff_o_b = get_tensor_exhaustive(bctx->ctx_data, ctx_gguf,
                {p+"ffn_down.bias",   s+"output.dense.bias"});
        }
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_ggml);
    return bctx;
}

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

// Forward declarations for graph caching — full definitions live with the
// graph-construction code further below. bert_free needs the complete type
// to call ~bert_graph_cache_map, so we provide a small helper there too.
struct bert_cached_graph;
struct bert_graph_cache_map;
static void bert_destroy_graph_cache(void * p);

int32_t bert_n_embd      (bert_ctx * ctx) { return ctx->model.hparams.n_embd; }
int32_t bert_n_max_tokens(bert_ctx * ctx) { return ctx->model.hparams.n_max_tokens; }

void bert_allocate_buffers(bert_ctx * ctx, int32_t n_max_tokens, int32_t batch_size) {
    bert_deallocate_buffers(ctx);

    ctx->buf_compute_meta.resize(
        GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());

    ctx->compute_alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(ctx->backend));

    // Dummy max-size batch for graph measurement
    bert_batch batch(batch_size, bert_tokens(n_max_tokens));
    struct ggml_cgraph * gf = bert_build_graph(ctx, batch);
    if (!gf) return;

    if (!ggml_gallocr_reserve(ctx->compute_alloc, gf)) {
        ggml_log_internal(GGML_LOG_LEVEL_ERROR,
            "%s: ggml_gallocr_reserve failed\n", __func__);
        return;
    }

    if (verbosity >= 1) {
        ggml_log_internal(GGML_LOG_LEVEL_INFO,
            "%s: compute buffer: %.2f MB\n", __func__,
            ggml_gallocr_get_buffer_size(ctx->compute_alloc, 0) / (1024.0*1024.0));
    }
}

void bert_deallocate_buffers(bert_ctx * ctx) {
    if (ctx->compute_alloc) {
        ggml_gallocr_free(ctx->compute_alloc);
        ctx->compute_alloc = nullptr;
    }
    if (ctx->compute_buffer) {
        ggml_backend_buffer_free(ctx->compute_buffer);
        ctx->compute_buffer = nullptr;
    }
}

void bert_free(bert_ctx * ctx) {
    if (!ctx) return;
    if (ctx->graph_cache) {
        bert_destroy_graph_cache(ctx->graph_cache);
        ctx->graph_cache = nullptr;
    }
    if (ctx->compute_alloc) {
        ggml_gallocr_free(ctx->compute_alloc);
        ctx->compute_alloc = nullptr;
    }
    if (ctx->compute_buffer) {
        ggml_backend_buffer_free(ctx->compute_buffer);
        ctx->compute_buffer = nullptr;
    }
    if (ctx->weights_buffer) {
        ggml_backend_buffer_free(ctx->weights_buffer);
        ctx->weights_buffer = nullptr;
    }
    if (ctx->ctx_data) {
        ggml_free(ctx->ctx_data);
        ctx->ctx_data = nullptr;
    }
    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
        ctx->backend = nullptr;
    }
    delete ctx;
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------
//
// The graph is parameterised by (max_len, n_batch) only — input contents are
// filled at forward time. We cache one graph per distinct (max_len, n_batch)
// in ctx->graph_cache so repeated forwards at the same shape skip both the
// graph build and the gallocr planning step. This is the dominant cost for
// short-input latency.

struct bert_cached_graph {
    int                  max_len    = 0;
    int                  n_batch    = 0;

    std::vector<uint8_t> buf_meta;             // backing for ctx_compute
    ggml_context       * ctx_compute = nullptr;
    ggml_cgraph        * gf          = nullptr;
    ggml_gallocr_t       alloc       = nullptr;

    // Cached input/output tensor pointers — set during build, used by every
    // subsequent forward to avoid name lookups.
    ggml_tensor * token_layer = nullptr;
    ggml_tensor * token_types = nullptr;  // pre-filled with zeros at build time
    ggml_tensor * pad_mask    = nullptr;
    ggml_tensor * positions   = nullptr;  // pre-filled with [0..max_len-1] at build time
    ggml_tensor * sum         = nullptr;
    ggml_tensor * minus_one   = nullptr;
    ggml_tensor * output      = nullptr;

    // Per-call scratch buffers — sized once per (max_len, n_batch) to avoid
    // heap churn on every forward. Only the contents are rewritten per call.
    std::vector<int32_t> scratch_tl;   // token ids
    std::vector<float>   scratch_pm;   // pad mask (1=real, 0=padding)
    std::vector<float>   scratch_sv;   // mean-pool weights (1/seq_len on real, 0 on pad)

    // Constant buffers — filled once at build, re-uploaded each call on CPU
    // (gallocr quirk: input-flagged tensors don't stay stable between computes).
    std::vector<int32_t> scratch_pos;
    std::vector<int32_t> scratch_types;

    ~bert_cached_graph() {
        if (alloc)       ggml_gallocr_free(alloc);
        if (ctx_compute) ggml_free(ctx_compute);
    }
};

using bert_graph_key = std::pair<int,int>;   // (max_len, n_batch)
struct bert_graph_cache_map {
    std::map<bert_graph_key, std::unique_ptr<bert_cached_graph>> entries;
};

static bert_graph_cache_map & bert_get_cache(bert_ctx * ctx) {
    if (!ctx->graph_cache) ctx->graph_cache = new bert_graph_cache_map();
    return *static_cast<bert_graph_cache_map *>(ctx->graph_cache);
}

static void bert_destroy_graph_cache(void * p) {
    delete static_cast<bert_graph_cache_map *>(p);
}

// Build the compute graph for fixed (max_len, n_batch). The returned struct
// owns its ggml_context / cgraph / allocator and is cached in bert_ctx.
//
// Notes:
//   - Self-attention uses ggml_flash_attn_ext (fused scaled-dot-product
//     attention with mask) instead of an unrolled mul_mat / softmax / mul_mat
//     so we get the Metal fused kernel and skip 3 cont+permute copies.
//   - Mean-pooling uses a 3D mul_mat that preserves the batch dim, fixing the
//     bs>1 reshape assertion in the previous implementation.
//   - L2 normalisation is still done on the host after backend_tensor_get.
static std::unique_ptr<bert_cached_graph>
build_cached_graph(bert_ctx * ctx, int max_len, int n_batch) {
    const bert_hparams & hp    = ctx->model.hparams;
    const bert_model   & model = ctx->model;

    const int   n_embd         = hp.n_embd;
    const int   n_layer        = hp.n_layer;
    const int   n_head         = hp.n_head;
    const float layer_norm_eps = hp.layer_norm_eps;
    const int   d_head         = n_embd / n_head;

    auto cg = std::make_unique<bert_cached_graph>();
    cg->max_len = max_len;
    cg->n_batch = n_batch;

    cg->buf_meta.resize(BERT_MAX_NODES * ggml_tensor_overhead() + ggml_graph_overhead());
    ggml_init_params p = { cg->buf_meta.size(), cg->buf_meta.data(), /*no_alloc=*/true };
    cg->ctx_compute = ggml_init(p);
    cg->gf = ggml_new_graph_custom(cg->ctx_compute, BERT_MAX_NODES, false);
    ggml_context * ctx0 = cg->ctx_compute;

    // Input tensors
    cg->token_layer = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, max_len * n_batch);
    ggml_set_name(cg->token_layer, "token_layer");
    cg->token_types = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, max_len * n_batch);
    ggml_set_name(cg->token_types, "token_types");
    cg->pad_mask    = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, max_len, 1, n_batch);
    ggml_set_name(cg->pad_mask, "pad_mask");
    cg->positions   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, max_len * n_batch);
    ggml_set_name(cg->positions, "positions");
    cg->sum         = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, max_len, 1, n_batch);
    ggml_set_name(cg->sum, "sum");
    cg->minus_one   = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(cg->minus_one, "minus_one");

    ggml_set_input(cg->token_layer);
    ggml_set_input(cg->token_types);
    ggml_set_input(cg->pad_mask);
    ggml_set_input(cg->positions);
    ggml_set_input(cg->sum);
    ggml_set_input(cg->minus_one);

    // Attention mask: 0 for unmasked pairs, -large for masked.
    // pad_mask [1, max_len, 1, n_batch] outer product -> [max_len, max_len, 1, n_batch]
    ggml_tensor * attn_mask_f32 = ggml_mul_mat(ctx0, cg->pad_mask, cg->pad_mask);
    attn_mask_f32 = ggml_scale_inplace(ctx0, attn_mask_f32, 100000.0f);
    ggml_tensor * large_offset = ggml_scale(ctx0, cg->minus_one, 100000.0f);
    attn_mask_f32 = ggml_add(ctx0, attn_mask_f32, large_offset);
    // flash_attn_ext requires f16 contiguous mask
    ggml_tensor * attn_mask = ggml_cast(ctx0, attn_mask_f32, GGML_TYPE_F16);

    // Embeddings — all shaped [n_embd, max_len, n_batch]
    ggml_tensor * word_embd = ggml_get_rows(ctx0, model.word_embeddings, cg->token_layer);
    word_embd = ggml_reshape_3d(ctx0, word_embd, n_embd, max_len, n_batch);
    ggml_tensor * type_embd = ggml_get_rows(ctx0, model.token_type_embeddings, cg->token_types);
    type_embd = ggml_reshape_3d(ctx0, type_embd, n_embd, max_len, n_batch);
    ggml_tensor * pos_embd  = ggml_get_rows(ctx0, model.position_embeddings, cg->positions);
    pos_embd  = ggml_reshape_3d(ctx0, pos_embd, n_embd, max_len, n_batch);

    ggml_tensor * inpL = ggml_add(ctx0, word_embd, type_embd);
    inpL = ggml_add(ctx0, inpL, pos_embd);
    inpL = ggml_norm_inplace(ctx0, inpL, layer_norm_eps);
    inpL = ggml_add(ctx0, ggml_mul(ctx0, inpL, model.ln_e_w), model.ln_e_b);

    const float attn_scale = 1.0f / sqrtf((float)d_head);

    for (int il = 0; il < n_layer; ++il) {
        const bert_layer & L = model.layers[il];
        ggml_tensor * cur = inpL;

        // Self-attention: Q, K, V projections then flash_attn_ext
        // Projections produce [n_embd, max_len, n_batch]. Reshape+permute to
        // [d_head, max_len, n_head, n_batch] — the layout flash_attn_ext wants.
        // No ggml_cont needed; flash_attn handles non-contiguous q/k/v.
        auto proj = [&](ggml_tensor * w, ggml_tensor * b) {
            ggml_tensor * x = ggml_add(ctx0, ggml_mul_mat(ctx0, w, cur), b);
            x = ggml_reshape_4d(ctx0, x, d_head, n_head, max_len, n_batch);
            return ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
        };
        ggml_tensor * Q = proj(L.q_w, L.q_b);
        ggml_tensor * K = proj(L.k_w, L.k_b);
        ggml_tensor * V = proj(L.v_w, L.v_b);

        // flash_attn_ext result: [d_head, n_head, max_len, n_batch] contiguous.
        // Reshape directly to [n_embd, max_len, n_batch] — d_head and n_head
        // are contiguous in memory so they merge cleanly.
        // Precision: F16 accumulate on accelerated backends is ~1.3-1.5x faster
        // (native fp16 hw) with negligible numerical drift for sentence embeddings.
        // CPU has no native fp16 path, so F16 there would just upconvert and pay
        // extra work — force F32 in that case.
        ggml_tensor * sdpa = ggml_flash_attn_ext(ctx0, Q, K, V, attn_mask,
                                                 attn_scale, 0.0f, 0.0f);
        if (ggml_backend_is_cpu(ctx->backend))
            ggml_flash_attn_ext_set_prec(sdpa, GGML_PREC_F32);
        cur = ggml_reshape_3d(ctx0, sdpa, n_embd, max_len, n_batch);

        // Output projection + residual + post-attention LayerNorm
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, L.o_w, cur), L.o_b);
        cur = ggml_add(ctx0, cur, inpL);
        cur = ggml_norm_inplace(ctx0, cur, layer_norm_eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L.ln_att_w), L.ln_att_b);

        ggml_tensor * att_out = cur;

        // FFN + residual + output LayerNorm
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, L.ff_i_w, cur), L.ff_i_b);
        cur = ggml_gelu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, L.ff_o_w, cur), L.ff_o_b);
        cur = ggml_add(ctx0, att_out, cur);
        cur = ggml_norm_inplace(ctx0, cur, layer_norm_eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L.ln_out_w), L.ln_out_b);

        inpL = cur;
    }

    // Mean pooling via 3D mul_mat that preserves the batch dim.
    //   inpL: [n_embd, max_len, n_batch]
    //   sum:  [max_len, 1, n_batch]    (1/seq_len on active positions, 0 on padding)
    //
    // For broadcasted batched mul_mat(A, B):
    //   A: [k, n, batch], B: [k, m, batch]  -> [n, m, batch]
    // We want [n_embd, 1, n_batch]. So set A = inpL_T with shape [max_len, n_embd, n_batch]
    // (transpose dims 0/1 only — batch dim preserved) and B = sum.
    ggml_tensor * inpL_T = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));   // [max_len, n_embd, n_batch]
    ggml_tensor * pooled = ggml_mul_mat(ctx0, inpL_T, cg->sum);           // [n_embd, 1, n_batch]
    pooled = ggml_reshape_2d(ctx0, pooled, n_embd, n_batch);

    cg->output = pooled;
    ggml_set_output(cg->output);
    ggml_build_forward_expand(cg->gf, cg->output);

    // Allocate compute buffers for this fixed graph shape.
    ggml_backend_dev_t          dev  = ggml_backend_get_device(ctx->backend);
    ggml_backend_buffer_type_t  buft = ggml_backend_dev_buffer_type(dev);
    cg->alloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(cg->alloc, cg->gf)) {
        fprintf(stderr, "%s: gallocr_alloc_graph failed (max_len=%d, n_batch=%d)\n",
                __func__, max_len, n_batch);
        return nullptr;
    }

    // ----------------------------------------------------------------------
    // One-time fill for tensors that never vary across forwards at this shape:
    //   - positions: always [0, 1, ..., max_len-1] for every batch element
    //   - token_types: always 0 for sentence-embedding BERT
    //   - minus_one: literal -1.0
    // After this point the backend buffers hold the correct values forever,
    // so bert_forward_batch can skip writing them on every call.
    // ----------------------------------------------------------------------
    {
        cg->scratch_pos.assign((size_t)max_len * n_batch, 0);
        for (int ba = 0; ba < n_batch; ++ba)
            for (int i = 0; i < max_len; ++i)
                cg->scratch_pos[(size_t)ba * max_len + i] = i;
        cg->scratch_types.assign((size_t)max_len * n_batch, 0);
        const float m1 = -1.0f;

        ggml_backend_tensor_set(cg->positions,   cg->scratch_pos.data(),   0, ggml_nbytes(cg->positions));
        ggml_backend_tensor_set(cg->token_types, cg->scratch_types.data(), 0, ggml_nbytes(cg->token_types));
        ggml_backend_tensor_set(cg->minus_one,   &m1,                      0, sizeof(m1));
    }

    // Size per-call scratch once — every forward at this shape rewrites contents
    // in place, never reallocs.
    cg->scratch_tl.resize((size_t)max_len * n_batch);
    cg->scratch_pm.resize((size_t)max_len * n_batch);
    cg->scratch_sv.resize((size_t)max_len * n_batch);

    return cg;
}

// Kept for ABI compatibility with bert.h: builds (and caches) a graph sized to
// the given batch. The returned cgraph is owned by ctx — callers MUST NOT free it.
struct ggml_cgraph * bert_build_graph(bert_ctx * ctx, bert_batch batch) {
    int max_len = 0;
    for (const auto & seq : batch) max_len = std::max(max_len, (int)seq.size());
    if (max_len > ctx->model.hparams.n_max_tokens) {
        ggml_log_internal(GGML_LOG_LEVEL_ERROR,
            "Too many tokens: max %d, got %d\n", ctx->model.hparams.n_max_tokens, max_len);
        return nullptr;
    }
    const int n_batch = (int)batch.size();
    auto & cache = bert_get_cache(ctx);
    auto key = std::make_pair(max_len, n_batch);
    auto it = cache.entries.find(key);
    if (it == cache.entries.end()) {
        auto cg = build_cached_graph(ctx, max_len, n_batch);
        if (!cg) return nullptr;
        it = cache.entries.emplace(key, std::move(cg)).first;
    }
    return it->second->gf;
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

void bert_forward_batch(bert_ctx * ctx, bert_batch batch,
                        float * embeddings, int32_t n_threads) {
    const int n_batch = (int)batch.size();
    if (n_batch == 0) return;

    int max_len = 0;
    for (const auto & s : batch) max_len = std::max(max_len, (int)s.size());
    if (max_len == 0) return;
    if (max_len > ctx->model.hparams.n_max_tokens) {
        ggml_log_internal(GGML_LOG_LEVEL_ERROR,
            "Too many tokens: max %d, got %d\n", ctx->model.hparams.n_max_tokens, max_len);
        return;
    }

    auto & cache = bert_get_cache(ctx);
    auto key = std::make_pair(max_len, n_batch);
    auto it = cache.entries.find(key);
    if (it == cache.entries.end()) {
        auto cg = build_cached_graph(ctx, max_len, n_batch);
        if (!cg) { fprintf(stderr, "%s: build graph failed\n", __func__); return; }
        it = cache.entries.emplace(key, std::move(cg)).first;
    }
    bert_cached_graph & cg = *it->second;

    // Fill per-call input buffers. positions / token_types / minus_one were
    // pre-filled at build time and never change. Scratch vectors are sized
    // once in build_cached_graph; we only rewrite their contents here.
    {
        int32_t * tl = cg.scratch_tl.data();
        float   * pm = cg.scratch_pm.data();
        float   * sv = cg.scratch_sv.data();

        for (int ba = 0; ba < n_batch; ++ba) {
            const int cl = (int)batch[ba].size();
            const float w = (cl > 0) ? 1.0f / (float)cl : 0.0f;
            const int row = ba * max_len;
            for (int i = 0; i < cl; ++i) {
                tl[row + i] = batch[ba][i];
                pm[row + i] = 1.0f;
                sv[row + i] = w;
            }
            for (int i = cl; i < max_len; ++i) {
                tl[row + i] = 0;
                pm[row + i] = 0.0f;
                sv[row + i] = 0.0f;
            }
        }
        ggml_backend_tensor_set(cg.token_layer, tl, 0, ggml_nbytes(cg.token_layer));
        ggml_backend_tensor_set(cg.pad_mask,    pm, 0, ggml_nbytes(cg.pad_mask));
        ggml_backend_tensor_set(cg.sum,         sv, 0, ggml_nbytes(cg.sum));
        // Workaround: CPU gallocr appears to reuse these slots between
        // compute calls even though they're flagged ggml_set_input. Metal
        // keeps them stable. Cheap to rewrite each call.
        if (ggml_backend_is_cpu(ctx->backend)) {
            ggml_backend_tensor_set(cg.positions,   cg.scratch_pos.data(),   0, ggml_nbytes(cg.positions));
            ggml_backend_tensor_set(cg.token_types, cg.scratch_types.data(), 0, ggml_nbytes(cg.token_types));
        }
    }

    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);

    ggml_backend_graph_compute(ctx->backend, cg.gf);

    ggml_backend_tensor_get(cg.output, embeddings, 0, ggml_nbytes(cg.output));

    // L2 normalisation (in-place) — sentence-transformer convention.
    const int n_embd = ctx->model.hparams.n_embd;
    for (int b = 0; b < n_batch; ++b) {
        float * vec = embeddings + b * n_embd;
        float  sum_sq = 0.0f;
        for (int i = 0; i < n_embd; ++i) sum_sq += vec[i] * vec[i];
        if (sum_sq > 1e-12f) {
            const float norm = std::sqrt(sum_sq);
            for (int i = 0; i < n_embd; ++i) vec[i] /= norm;
        }
    }
}

void bert_encode_batch(bert_ctx * ctx, bert_strings texts,
                       float * embeddings, int32_t n_threads) {
    const int32_t N = bert_n_max_tokens(ctx);
    bert_batch batch;
    batch.reserve(texts.size());
    for (const auto & t : texts)
        batch.push_back(bert_tokenize(ctx, t, N));
    bert_forward_batch(ctx, batch, embeddings, n_threads);
}

void bert_encode_batch_c(bert_ctx * ctx, const char ** texts,
                         float * embeddings, int32_t n_input, int32_t n_threads) {
    bert_encode_batch(ctx, bert_strings(texts, texts + n_input), embeddings, n_threads);
}

void bert_forward(bert_ctx * ctx, bert_tokens tokens,
                  float * embeddings, int32_t n_threads) {
    bert_forward_batch(ctx, {tokens}, embeddings, n_threads);
}

void bert_encode(bert_ctx * ctx, bert_string text,
                 float * embeddings, int32_t n_threads) {
    bert_encode_batch(ctx, {text}, embeddings, n_threads);
}

const char * bert_vocab_id_to_token(bert_ctx * ctx, int32_t id) {
    const bert_vocab & v = ctx->vocab;
    auto it = v._id_to_token.find(id);
    if (it != v._id_to_token.end()) return it->second.c_str();
    auto it2 = v._id_to_subword_token.find(id);
    if (it2 != v._id_to_subword_token.end()) return it2->second.c_str();
    return "[UNK]";
}


/*

// In bert_tokenize:
if (vocab.is_sentencepiece) {
    // ▁-prefix greedy tokenizer (current working code)
} else {
    // ## WordPiece greedy tokenizer (old code)
}

*/
