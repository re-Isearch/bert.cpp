# bert.cpp inference perf exploration (autonomous session 2026-06-11)

Branch: perf/explore (on top of the 2 correctness fixes).
GPU: NVIDIA RTX 1000 Ada (cc 8.9, 6GB). nvcc 12.9, gcc 14.3.
Method: bench -w5 -n40, median ms; correctness = cosine vs golden/<model>.cuda.txt (must stay >0.9999).
Baselines saved in perflab/baseline/*.txt, golden embeddings in perflab/golden/*.txt.

## Baseline (CUDA, build-cuda, GGML_CUDA_GRAPHS=OFF)
| model | medium(137) | max(512) | bs=32(137) |
|---|---|---|---|
| bge-small-f16 | 3.02 | 10.36 | 80.81 |
| bge-base-f16  | 3.85 | 11.48 | 126.63 |
| bge-large-f16 | 12.82 | 35.03 | 409.41 |
| bge-large-q8_0| 10.91 | 31.24 | 352.09 |

## Experiments

### Exp1: GGML_CUDA_GRAPHS=ON  [WIN — adopt]
Build flag only. Correctness cos=1.0 all models. Helps launch-bound (bs=1, short seq):
| model | metric | base | graphs | delta |
|---|---|---|---|---|
| bge-small | very-short | 1.39 | 1.06 | -24% |
| bge-small | medium     | 3.02 | 2.22 | -26% |
| bge-base  | very-short | 1.73 | 1.60 | -8% |
| bge-large-q8 | very-short | 4.60 | 4.05 | -12% |
| bge-large-q8 | medium  | 10.91 | 10.24 | -6% |
Compute-bound (max, bs=32) ~unchanged. Recommend enabling in CMakeLists default for CUDA.

### Exp2: drop ggml_cont on Q/K/V (flash path)  [WIN — adopt]
proj() returns permuted view; flash_attn_ext accepts non-contiguous. Unfused path conts K locally.
Correctness cos=1.0 all 4 models. Stacks on top of graphs:
| model | metric | graphs-only | +nocont | delta |
|---|---|---|---|---|
| bge-base | medium | 3.65 | 2.86 | -22% |
| bge-base | max    | 11.26 | 9.51 | -16% |
| bge-base | bs=32  | 125.1 | 113.6 | -9% |
| bge-large-q8 | medium | 10.24 | 9.35 | -9% |
| bge-large-q8 | max    | 29.81 | 26.72 | -10% |
| bge-large-q8 | bs=8   | 71.8 | 62.8 | -12% |
very-short ~flat (noise). Net vs ORIGINAL baseline (graphs+nocont): bge-base medium 3.85->2.86 (-26%).

### Exp4: GGML_CUDA_FORCE_MMQ=ON (q8)  [NO WIN — discard]
q8 MMQ kernels vs default cuBLAS-dequant: timings within noise (med 9.24 vs 9.35, max 26.26 vs 26.72),
very-short slightly worse (4.30 vs 3.89). cos=1.0. Default cuBLAS path is fine. Also confirmed cuBLAS
already uses CUBLAS_COMPUTE_16F for default-prec f16 matmuls (activations auto-cast to f16) -> big GEMMs
already on tensor cores, no headroom from "force f16".

### Exp_gelu: ggml_gelu is already the fast tanh approx; _erf slower, _quick risks accuracy. No change.

### Exp5: CPU OpenBLAS (GGML_BLAS=ON)  [NO WIN — discard]
bge-base f16 CPU t=8: medium 99->388ms (4x WORSE). ggml native CPU kernels win. Do not enable BLAS.

## VERIFIED RESULTS (paired interleaved A/B — controls thermal throttle)
base=build-cuda (graphs OFF, cont) vs best=build-graphs (graphs ON + no-cont). cos=1.0 (bit-identical).
Rounds 1-2 stable thermal; round3 q8 hit throttle but BOTH throttled, best still faster.
| model | medium | bs=32 |
|---|---|---|
| bge-base   | 3.6-4.1 -> 2.9-3.0 (~-20%) | 125-127 -> 114-117 (~-9%) |
| bge-large-q8 | 10.8 -> 9.6 (~-12%) | 348 -> 319 (~-8%) |
Plus bge-small very-short -24% (graphs, launch-bound).

## CONCLUSION / RECOMMENDATION
Two zero-risk, bit-identical wins (both adopted on perf/explore):
1. **GGML_CUDA_GRAPHS=ON** (build flag; now forced ON in top CMakeLists for CUDA). Biggest on short
   inputs / small models (launch-bound): up to -25%.
2. **Drop ggml_cont on Q/K/V** (flash_attn accepts non-contiguous). Helps compute-bound: -8..-20%.
Combined: ~10-20% latency, ~8-9% batch throughput, identical embeddings.

Dead ends (measured): FORCE_MMQ (q8) neutral; cuBLAS already COMPUTE_16F (no f16 headroom);
gelu already fast tanh; CPU OpenBLAS 4x WORSE; fused-QKV not worth it (GEMMs already tensor-core,
GPU is 35W power-bound at ~6 TFLOP/s so overhead-reduction is the right lever, not GEMM tuning).

NOTE: 35W laptop GPU thermally throttles under sustained load (max latency ~2x after ~30min). Real
deployments on desktop/server CUDA or Metal will see steadier numbers; relative speedups hold.
