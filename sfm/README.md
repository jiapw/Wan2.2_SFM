# SFM Camera-Controlled LoRA for Wan2.2

本目录提供一个**不改动 Wan 原始代码**的完整工程：

1. 用 COLMAP 风格多视角照片训练 LoRA。
2. 保持 Wan 原架构不变，通过外部 LoRA + Camera Encoder 注入条件。
3. 推理时输入 prompt + 外部相机轨迹（每帧参数），生成可控游览视频。

> 所有新增文件都在 `sfm/` 下，与 Wan 主工程解耦。

---

## 1) 独立安装（与 Wan 主环境分开）

```bash
cd /workspace/Wan2.2_SFM
python -m venv .venv-sfm
source .venv-sfm/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -r sfm/requirements.txt
```

如果你已经有 Wan 运行环境，也建议用独立 venv，避免依赖冲突。

---

## 2) 训练数据格式（COLMAP）

训练输入使用 COLMAP 文本导出：

- `images/`：真实照片
- `sparse_txt/cameras.txt`
- `sparse_txt/images.txt`

示例结构：

```text
scene/
  images/
    0001.png
    0002.png
    ...
  sparse_txt/
    cameras.txt
    images.txt
```

### 预处理约定

- 自动做**反畸变**（如果相机模型带畸变参数）。
- 然后做**中心裁剪 + resize**到训练分辨率。

---

## 3) LoRA 训练

### 方式 A：直接命令行

```bash
python -m sfm.train_lora \
  --ckpt_dir /path/to/Wan2.2-T2V-A14B \
  --scene_image_dir /data/scene/images \
  --colmap_cameras_txt /data/scene/sparse_txt/cameras.txt \
  --colmap_images_txt /data/scene/sparse_txt/images.txt \
  --prompt "A realistic fly-through of this room" \
  --out_dir ./sfm_outputs/scene_room \
  --size 832*480 \
  --epochs 10 \
  --batch_size 1 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --cam_mode both
```

### 方式 B：shell 脚本

```bash
bash sfm/scripts/train_lora.sh \
  /path/to/Wan2.2-T2V-A14B \
  /data/scene/images \
  /data/scene/sparse_txt/cameras.txt \
  /data/scene/sparse_txt/images.txt \
  "A realistic fly-through of this room" \
  ./sfm_outputs/scene_room
```

### 关键参数

- `--lora_targets`：稳妥版本默认注入 Self/Cross Attention 和 FFN 线性层。
- `--cam_mode`：
  - `latent_bias`
  - `context_token`
  - `both`（默认，按你的要求 C）
- `--lr_lora`, `--lr_camera`：LoRA 与 camera encoder 分离学习率。

### 训练输出

输出目录会包含：

- `lora_epoch_XXX.pt`, `lora_final.pt`
- `camera_encoder_epoch_XXX.pt`, `camera_encoder_final.pt`
- `train_args.json`

---

## 4) T2V 推理（外部轨迹控制）

相机轨迹文件使用 JSON：`list[list[13]]`，每一行对应 1 帧：

`[q0,q1,q2,q3, tx,ty,tz, fx,fy,cx,cy, width,height]`

可参考：`sfm/examples/camera_traj_example.json`。

### 命令行方式

```bash
python -m sfm.infer_t2v \
  --ckpt_dir /path/to/Wan2.2-T2V-A14B \
  --prompt "A cinematic walkthrough of the same room" \
  --camera_traj sfm/examples/camera_traj_example.json \
  --lora_ckpt ./sfm_outputs/scene_room/lora_final.pt \
  --camera_ckpt ./sfm_outputs/scene_room/camera_encoder_final.pt \
  --output ./sfm_outputs/scene_room/demo.mp4 \
  --size 832*480 \
  --sample_steps 30
```

### shell 脚本方式

```bash
bash sfm/scripts/infer_t2v.sh \
  /path/to/Wan2.2-T2V-A14B \
  "A cinematic walkthrough of the same room" \
  sfm/examples/camera_traj_example.json \
  ./sfm_outputs/scene_room/lora_final.pt \
  ./sfm_outputs/scene_room/camera_encoder_final.pt \
  ./sfm_outputs/scene_room/demo.mp4
```

---

## 5) 实现说明（和 proposal 对齐）

- Wan 主模型参数默认冻结。
- 仅训练：
  - LoRA 参数（注入到 DiT 线性层）
  - Camera Encoder + 投影头
- 相机条件注入采用可配置策略：
  - latent bias
  - context token
  - both
- 推理使用外部轨迹逐帧控制，满足 frame-level camera control。

---

## 6) 注意事项

- 该版本先实现你确认的**第一阶段：图像重建导向训练**。
- 在 4090 24G 上建议 `batch_size=1`，并先用较短轨迹做验证。
- 若 COLMAP 使用二进制文件，请先导出成 txt（本实现读取 txt）。
