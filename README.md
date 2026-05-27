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

Turning now to a 335-million parameter model and more exhaustive produuction-like benchmarks:
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

Our test M1-Pro chews through roughly 45 passages per second per instance (3484 tok/s÷78 ms). That means one can expect to process 2,700 fully dense semantic records per minute on a baseline Apple Silicon chip.  Our tests on M3Pro and M4Pro should even significntly high throughputs (80k tokens/s or as much as 20x).

We are developing and benchmarking using the M1 as a baseline upon which to base preformance expectations. Instead of demanding the latest and greatest (or an array of H200s) we want to provide usefull performance on litteraly "*the smallest machine possible*".

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
