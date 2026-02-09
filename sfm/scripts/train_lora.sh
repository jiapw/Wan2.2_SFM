#!/usr/bin/env bash
set -euo pipefail

python -m sfm.train_lora \
  --ckpt_dir "$1" \
  --scene_image_dir "$2" \
  --colmap_cameras_txt "$3" \
  --colmap_images_txt "$4" \
  --prompt "$5" \
  --out_dir "$6" \
  --size "${7:-832*480}" \
  --epochs "${8:-10}" \
  --batch_size 1 \
  --cam_mode both
