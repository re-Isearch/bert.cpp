#!/bin/bash
source /home/yzimmermann/miniforge3/etc/profile.d/conda.sh; conda activate cuda
BIN="$1"; MODEL="$2"; EXTRA="$3"
$BIN -m "$MODEL" -w 5 -n 40 $EXTRA 2>/dev/null | awk '
/very-short|short |medium|long|max |bs=1 |bs=8 |bs=32/ {
  printf "  %-11s %s\n", $1, $0
}' | sed 's/|.*med=/ med=/; s/p95.*tok\/s/ /; s/  */ /g'
