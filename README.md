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


### Install

Fetch this respository then download submodules and install packages with
```sh
git submodule update --init --recursive
```

GGUF models can be fetched from `huggingface` or convert other formats to `gguf`.
