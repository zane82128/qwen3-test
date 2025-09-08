#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
PY=python
SCRIPT=debate_rounds_new_pipe.py
# OUTDIR=runs

declare -a PROMPTS=(
  "Fauvism, Miyazaki Hayao, a girl and a dragon in the cave"
  "Impressionism, Ukiyo-e, a fox and a paper lantern on a misty bridge"
  "Van Gogh, Studio Ghibli, a windmill and a bicycle under swirling stars"
)
declare -a ROUNDS=(1 3 3)
declare -a OUTDIRS=(runs/1 runs/2 runs/3)

for i in "${!PROMPTS[@]}"; do
  "$PY" "$SCRIPT" --prompt "${PROMPTS[$i]}" --rounds "${ROUNDS[$i]}" --outdir "${OUTDIRS[$i]}"
done
