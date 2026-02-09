from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear that keeps the base weight frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze base weight/bias; only LoRA params are trainable.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out


@dataclass
class LoRAInjectResult:
    replaced_modules: List[str]


def _get_parent_module(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora_modules(
    model: nn.Module,
    rank: int,
    alpha: float,
    target_keywords: Iterable[str],
    dropout: float = 0.0,
) -> LoRAInjectResult:
    """Replace target nn.Linear modules with LoRALinear wrappers by name keyword match."""
    targets = list(target_keywords)
    replaced: List[str] = []

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(k in module_name for k in targets):
            continue
        parent, leaf = _get_parent_module(model, module_name)
        setattr(parent, leaf, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(module_name)

    return LoRAInjectResult(replaced_modules=replaced)


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if ("lora_A" in k or "lora_B" in k)]
    if bad_missing:
        raise RuntimeError(f"Missing LoRA keys during load: {bad_missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys during LoRA load: {unexpected[:10]}")
