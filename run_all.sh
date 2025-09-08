#!/usr/bin/env bash
set -u

PY=python
SCRIPT=debate_rounds_new_pipe.py

PROMPTS=(
  "Fauvism, Miyazaki Hayao, a girl and a dragon in the cave"
  "Impressionism, Ukiyo-e, a fox and a paper lantern on a misty bridge"
  "Van Gogh, Studio Ghibli, a windmill and a bicycle under swirling stars"
)
ROUNDS=(3 3 3)
OUTDIRS=(runs/1 runs/2 runs/3)
GPUS=(0 2 4)   # 只有一張卡就全改成同一個值（注意顯存）

mkdir -p runs/logs

for i in "${!PROMPTS[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES="${GPUS[$i]:-0}"
    "$PY" "$SCRIPT" \
      --prompt "${PROMPTS[$i]}" \
      --rounds "${ROUNDS[$i]}" \
      --outdir "${OUTDIRS[$i]}"
  ) >"runs/logs/job_$i.log" 2>&1 &
done

wait
echo "All jobs done."
