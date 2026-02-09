from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from sfm.camera import CameraConditionConfig, CameraConditioner
from sfm.lora import inject_lora_modules, load_lora_state_dict
from sfm.wan_bridge import build_wan_t2v
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import save_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Camera trajectory controlled T2V inference")
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--camera_traj", type=str, required=True, help="JSON file: list[list[13]]")
    p.add_argument("--lora_ckpt", type=str, required=True)
    p.add_argument("--camera_ckpt", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--size", type=str, default="832*480")
    p.add_argument("--sample_steps", type=int, default=30)
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--sample_shift", type=float, default=5.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--cam_mode", type=str, default="both", choices=["latent_bias", "context_token", "both"])
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument(
        "--lora_targets",
        type=str,
        default="self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,ffn.0,ffn.2",
    )
    return p.parse_args()


def parse_size(size_str: str) -> tuple[int, int]:
    w, h = size_str.split("*")
    return int(w), int(h)


def _set_scheduler(args, device):
    if args.sample_solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(args.sample_steps, device=device, shift=args.sample_shift)
        return scheduler, scheduler.timesteps
    scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sigmas = get_sampling_sigmas(args.sample_steps, args.sample_shift)
    timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sigmas)
    return scheduler, timesteps


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        args.seed = random.randint(0, sys.maxsize)

    wan = build_wan_t2v(ckpt_dir=args.ckpt_dir, task="t2v-A14B", device_id=0)

    # Inject LoRA into both low/high noise models for consistency with Wan sampling.
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    for model in [wan.low_noise_model, wan.high_noise_model]:
        inject_lora_modules(model, rank=args.lora_rank, alpha=args.lora_alpha, target_keywords=targets)

    lora_obj = torch.load(args.lora_ckpt, map_location="cpu")
    lora_sd = lora_obj["lora"] if isinstance(lora_obj, dict) and "lora" in lora_obj else lora_obj
    load_lora_state_dict(wan.low_noise_model, lora_sd)
    load_lora_state_dict(wan.high_noise_model, lora_sd)

    cam = CameraConditioner(
        CameraConditionConfig(
            cam_dim=13,
            embed_dim=512,
            latent_channels=wan.low_noise_model.in_dim,
            context_dim=wan.low_noise_model.text_dim,
            mode=args.cam_mode,
        )
    ).to(wan.device)
    cam.load_state_dict(torch.load(args.camera_ckpt, map_location=wan.device))
    cam.eval()

    camera_traj = torch.tensor(json.loads(Path(args.camera_traj).read_text(encoding="utf-8")), dtype=torch.float32, device=wan.device)
    width, height = parse_size(args.size)
    videos = []

    context = wan.text_encoder([args.prompt], wan.device)[0]
    context_null = wan.text_encoder([wan.sample_neg_prompt], wan.device)[0]

    generator = torch.Generator(device=wan.device)
    generator.manual_seed(args.seed)

    for cam_vec in tqdm(camera_traj, desc="render frames"):
        cam_embed = cam(cam_vec.unsqueeze(0)).squeeze(0)
        cond_context = cam.apply_to_context(context, cam_embed)
        uncond_context = cam.apply_to_context(context_null, cam_embed)

        F = 1
        target_shape = (
            wan.vae.model.z_dim,
            (F - 1) // wan.vae_stride[0] + 1,
            height // wan.vae_stride[1],
            width // wan.vae_stride[2],
        )
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (wan.patch_size[1] * wan.patch_size[2]) * target_shape[1])

        latent = torch.randn(*target_shape, device=wan.device, generator=generator)
        latent = cam.apply_to_latent(latent, cam_embed)

        scheduler, timesteps = _set_scheduler(args, wan.device)
        boundary = wan.boundary * wan.num_train_timesteps
        for t in timesteps:
            tt = torch.tensor([t], device=wan.device)
            model = wan.high_noise_model if t.item() >= boundary else wan.low_noise_model
            noise_cond = model([latent], t=tt, context=[cond_context], seq_len=seq_len)[0]
            noise_uncond = model([latent], t=tt, context=[uncond_context], seq_len=seq_len)[0]
            noise = noise_uncond + args.guide_scale * (noise_cond - noise_uncond)
            latent = scheduler.step(noise.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=generator)[0].squeeze(0)

        frame = wan.vae.decode([latent])[0]  # [3,1,H,W]
        videos.append(frame)

    video = torch.cat(videos, dim=1).unsqueeze(0)  # [1,3,T,H,W]
    save_video(video, save_file=args.output, fps=args.fps)
    print(f"Saved controlled video to {args.output}")


if __name__ == "__main__":
    main()
