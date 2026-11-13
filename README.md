# bert.cpp

This is a *heavily refactored* version of [bert.cpp](https://github.com/ggerganov/bert.cpp) to support the newest ggml tensor library (0.12 May 2026). It has been designed for re-Isearch's Schmate project [schmate](https://nlnet.nl/project/Re-Isearch-Vector/).  It may be a drop-in replacement for the last bert.cpp.

Since many newer BERTish GGUF models use Sentencepiece rather than Wordpiece tokenization we try to detect things and choose the right tokenizer.

Within this bert.cpp we support CUDA (NVIDIA), Metal (Apple hardware), Vulkan and fallback CPU. Support for other accelerators (as supported by the ggml tensor library) can be added as needed.

Because of the shift we no longer support GGML model files but just the newer GGUF format which replaced it. We have tried to support the various metadata names in GGUF files for BERT models but the implmentation does NOT support all GGUF, in particular EuroBERT, models. On failure we dump a copy of the metadata so one can decide to either Update the internal list of names or use llama.cpp instead. 

A simple test is:
   if (key == "general.architecture")
     architecture = value;
   ...
   if (key == "bert.pooling_type" || key == "pooling_type")
     pooling_type = value;
   ...
   bert_cpp_compatible = (architecture == "bert" || architecture == "nomic-bert");
   if (is_bert_cpp_compatible && pooling_type == -1) {
      // Might NOT be OK.
    }

Why bert.cpp instead of llama.cpp? Size and performance.

## Model inputs and pooling

Sentence embedding conventions come from the model, independently of GGUF
weight precision or the downstream vector index. The loader reads special-token
IDs and casing from GGUF metadata, with vocabulary-name fallbacks. It honors
`<architecture>.pooling_type`, `bert.pooling_type`, or `pooling_type`: 1 means
mean, 2 CLS, and 3 last-token pooling. Without that metadata, known BGE encoders
default to CLS; other existing models retain mean pooling, including multilingual
E5. RoBERTa architecture alone does not determine pooling. Unsupported sentence
pooling values are rejected, and non-CLS BGE metadata is reported so a mismatched
export is visible.

XLM-R/M3 uses `<s>`/`</s>` and case-sensitive text, rather than fixed BERT
101/102 IDs and unconditional lowercasing. Its non-padding position IDs begin
at `padding_idx + 1`; padding positions retain `padding_idx`. Position capacity
is bounded by the actual loaded table. The selected token IDs, casing,
position offset, and pooling type are logged when a model loads.

Changing these conventions changes the embeddings themselves. Re-embed and
rebuild indexes created with a different input/pooling profile.

### Inspect embeddings before indexing

`embedding_probe` computes exact cosine/dot products on raw float embeddings,
without Schmate packing, HNSW, or query-energy score calibration. It prints the
query token IDs/pieces, model profile, vector norms and unknown-token counts.
With no passage file it uses a small Searle/election/proverb contrast set:

```sh
./build/bin/embedding_probe ../models/bge-m3-f16.gguf \
  "what did John Searle think about Derrida" --cpu
```

An optional `passages.txt` supplies one passage per line. `--pooling cls|mean|last`
allows a controlled pooling comparison; `--threads N` sets CPU threads.
Inputs are unprefixed, matching M3's symmetric interface. Prefix-trained models
need their intended instruction text supplied explicitly.

### Regression tests

```sh
cmake -S . -B build -DBERT_BUILD_TESTS=ON -DGGML_METAL=OFF \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BUILD_TESTS=OFF
cmake --build build --target embedding_profile_test embedding_probe -j2
ctest --test-dir build --output-on-failure
```

The test exports small one-layer GGUF fixtures and loads them through the real
backend. Independent float calculations verify CLS/mean/last pooling, model
position offsets, unequal-length batches and graph-cache reuse. It also covers
special IDs, case sensitivity, remapped vocabularies, a WordPiece vocabulary
containing a literal `▁`, short vocabularies, token budgets and metadata defaults.
Large model weights are not required. CPU fixtures do not establish full M3
retrieval quality or Metal/CUDA parity.

The existing greedy SentencePiece segmenter remains an approximation: these
changes do not implement SentencePiece's full normalization/unigram pipeline.
Compare probe token IDs with the source tokenizer when validating a GGUF export.

Reference conventions: [M3 model configuration](https://huggingface.co/BAAI/bge-m3/blob/main/config.json),
[casing configuration](https://huggingface.co/BAAI/bge-m3/blob/main/sentence_bert_config.json),
[CLS dense embedding guidance](https://huggingface.co/BAAI/bge-m3/discussions/80), and
[XLM-R position construction](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py).

On an Apple M1:
model:   sentence-transformers_all-MiniLM-L12-v2
main:    token time =     0.03 ms / 0.01 ms per token
main:     eval time =    15.07 ms / 3.77 ms per token

model:   bge-large-en-v1.5-q4_k_m.gguf
main:    token time =     0.03 ms / 0.01 ms per token
main:     eval time =    30.91 ms / 7.73 ms per token


model:   sentence-transformers_all-MiniLM-L12-v2-f32.gguf
main:    token time =     0.02 ms / 0.00 ms per token
main:     eval time =    21.52 ms / 5.38 ms per token

Above were all single shot.

Turning now to a 335-million parameter model and more exhaustive production-like benchmarks:
<PRE>
Model: bge-large-en-v1.5  arch=bert  n_embd=1024  load=150 ms
Warmup=5 iters, measure=50 iters

== latency by input length (batch=1) ==
  very-short     tok= 11  bs=  1  | mean=  17.47 ms  med=  17.47  p95=  18.01  min= 16.93  max= 18.38  |      630 tok/s     57.3 seq/s
  short          tok= 38  bs=  1  | mean=  23.97 ms  med=  23.83  p95=  24.92  min= 22.97  max= 25.14  |     1586 tok/s     41.7 seq/s
  medium         tok=137  bs=  1  | mean=  44.67 ms  med=  44.41  p95=  46.24  min= 43.39  max= 46.92  |     3067 tok/s     22.4 seq/s
  long           tok=263  bs=  1  | mean=  78.83 ms  med=  78.07  p95=  81.44  min= 76.53  max= 82.25  |     3336 tok/s     12.7 seq/s
  max            tok=512  bs=  1  | mean= 132.63 ms  med= 132.67  p95= 133.81  min=130.45  max=134.25  |     3860 tok/s      7.5 seq/s

== throughput by batch size (tok=128) ==
  bs=1           tok=137  bs=  1  | mean=  44.20 ms  med=  43.95  p95=  45.58  min= 43.41  max= 46.02  |     3100 tok/s     22.6 seq/s
  bs=4           tok=137  bs=  4  | mean= 161.03 ms  med= 160.01  p95= 166.50  min=156.61  max=167.15  |     3403 tok/s     24.8 seq/s
  bs=8           tok=137  bs=  8  | mean= 314.57 ms  med= 313.94  p95= 324.07  min=291.38  max=328.69  |     3484 tok/s     25.4 seq/s
</PRE>


Because throughput peaks at around 3,484 tokens/sec (0.29 ms per token) we use these metrics to calculate some real-world document ingestion limits.  If you block your raw text files into standard overlapping paragraph windows of roughly 256 tokens each: A single passage will compute in ~78 ms.

Our test M1-Pro chews through roughly 45 passages per second per instance (3484 tok/s÷78 ms). That means one can expect to process 2,700 fully dense semantic records per minute on a baseline Apple Silicon chip.  Our tests on M3Pro and M4Pro showed even significantly higher throughputs (80k tokens/s or as much as 20x).

We are developing and benchmarking using the M1 as a baseline upon which to base preformance expectations. Instead of demanding the latest and greatest (or an array of H200s) our design goals and constaints are to provide useful performance on literaly "*the smallest machine possible*"-- and not to demand some hyperscalar cluster.

- M1 Pro (ARMv8): Uses an older iteration of Apple's proprietary AMX (Apple Matrix Coprocessor). It is highly optimized for accelerating single-sequence matrix-vector math (bs=1).
- M4 Family (ARMv9 + SME): The M4 introduced SME (Scalable Matrix Extension). SME is explicitly designed to handle outer-product matrix-matrix tiles natively in hardware. When you increase the batch size, the mathematical operation changes from a series of matrix-vector multiplies into a massive matrix-matrix multiplication (GEMM). The M4's SME hardware can swallow those batched matrix tiles concurrently, executing them multiple times faster per clock cycle than the M1 Pro's AMX.
- An M4 Pro has up to 20 next-generation GPU cores, and an M4 Max has up to 40. Combined with a memory bandwidth of up to 546 GB/s, the M4 Max has a massive parallel execution pool.

With a base M4 Mac mini, we can expect benchmarking metrics something like this:
- Single Query Latency (bs=1, tok=38): Will drop from your current ~24 ms down to under 8 ms, making live UI search-as-you-type operations completely instantaneous.
- Peak Batch Throughput: Will scale up from your current 3,484 tokens/second to a sustained bracket of 12,000 to 15,000 tokens/second.

The Projected M5 Performance Matrix:
- Running bge-large-en-v1.5 on a cheap M5 Mac mini will yield metrics that look more like a dedicated server data center than a compact desktop:
- Interactive Query Latency (bs=1, short query): ~3 to 5 ms (making search-as-you-type feel physically instantaneous).
- Peak Batch Throughput: Easily hitting 45,000 to 60,000+ tokens/second.

At 60,000 tokens per second, a single entry-level M5 mini will let you index up to 14,000 full paragraphs every single minute.

### Install

Fetch this respository then download submodules and install packages with
```sh
git submodule update --init --recursive
```

GGUF models can be fetched from `huggingface` or convert other formats to `gguf`.
