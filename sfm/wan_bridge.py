from __future__ import annotations

import random
from types import SimpleNamespace
from typing import List, Tuple

import torch

from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.text2video import WanT2V


def parse_size(size: str) -> Tuple[int, int]:
    if size in SIZE_CONFIGS:
        return SIZE_CONFIGS[size]
    w, h = size.split("*")
    return (int(w), int(h))


def build_wan_t2v(
    ckpt_dir: str,
    task: str = "t2v-A14B",
    device_id: int = 0,
    t5_cpu: bool = False,
    offload: bool = False,
):
    cfg = WAN_CONFIGS[task]
    return WanT2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=device_id,
        rank=0,
        t5_cpu=t5_cpu,
        init_on_cpu=offload,
    )


def encode_prompt(model: WanT2V, prompt: str, n_prompt: str = "") -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # Follow Wan generate() flow.
    context = model.text_encoder([prompt], model.device)
    context_null = model.text_encoder([n_prompt], model.device)
    return context, context_null


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
