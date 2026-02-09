#!/usr/bin/env bash
set -euo pipefail

python -m sfm.infer_t2v \
  --ckpt_dir "$1" \
  --prompt "$2" \
  --camera_traj "$3" \
  --lora_ckpt "$4" \
  --camera_ckpt "$5" \
  --output "$6" \
  --size "${7:-832*480}" \
  --sample_steps "${8:-30}"
