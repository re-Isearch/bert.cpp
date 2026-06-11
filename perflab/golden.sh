#!/bin/bash
# capture embedding for a fixed prompt -> $3
source /home/yzimmermann/miniforge3/etc/profile.d/conda.sh; conda activate cuda
BIN="$1"; MODEL="$2"; OUT="$3"; EXTRA="$4"
P="Transformer models compute contextual embeddings for semantic search and retrieval over documents."
$BIN -m "$MODEL" -p "$P" $EXTRA > "$OUT" 2>/dev/null
