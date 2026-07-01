#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

dst_root=${1:-/mnt/hdd/rintarootsubo/icl/data/share/predictions/temporal}

models=(
  gemini_2_5_flash
  gemini_2_5_pro
  gemini_3_flash_preview
  gemini_3_1_pro_preview
  gpt-4o
  gpt5_1
  LLaVA-ST
  qwen3_vl_4B_instruct
  qwen3_vl_8B_instruct
  qwen3_5_4B
  qwen3_5_9B
  internvl3_8B
  internvl3_5_8B
  internvl3_14B
)

mkdir -p "${dst_root}"

for model in "${models[@]}"; do
  mkdir -p "${dst_root}/${model}_0shot"
  cp data/eval/"${model}"/outputs/* "${dst_root}/${model}_0shot/"

  mkdir -p "${dst_root}/${model}_2shot"
  cp data/eval/"${model}"_2shot/outputs/* "${dst_root}/${model}_2shot/"
done

mkdir -p "${dst_root}/eagle_2shot"
cp data/eval/eagle_2shot/outputs/* "${dst_root}/eagle_2shot/"
