from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sfm.camera import CameraConditionConfig, CameraConditioner
from sfm.colmap_dataset import ColmapSceneDataset
from sfm.lora import inject_lora_modules, lora_state_dict
from sfm.wan_bridge import build_wan_t2v, encode_prompt, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Wan2.2 LoRA with COLMAP cameras")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--scene_image_dir", type=str, required=True)
    parser.add_argument("--colmap_cameras_txt", type=str, required=True)
    parser.add_argument("--colmap_images_txt", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--size", type=str, default="832*480", help="W*H")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_lora", type=float, default=1e-4)
    parser.add_argument("--lr_camera", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,ffn.0,ffn.2",
    )

    parser.add_argument("--cam_embed_dim", type=int, default=512)
    parser.add_argument("--cam_mode", type=str, default="both", choices=["latent_bias", "context_token", "both"])
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def parse_wh(size_str: str) -> tuple[int, int]:
    w, h = size_str.split("*")
    return int(h), int(w)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    dataset = ColmapSceneDataset(
        image_dir=args.scene_image_dir,
        cameras_txt=args.colmap_cameras_txt,
        images_txt=args.colmap_images_txt,
        output_hw=parse_wh(args.size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    wan = build_wan_t2v(ckpt_dir=args.ckpt_dir, task="t2v-A14B", device_id=0)
    model = wan.low_noise_model
    model.train()

    # Freeze all Wan params first; only LoRA + camera conditioner will train.
    for p in model.parameters():
        p.requires_grad = False

    lora_targets = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
    inject = inject_lora_modules(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_keywords=lora_targets,
        dropout=args.lora_dropout,
    )
    if not inject.replaced_modules:
        raise RuntimeError("No LoRA modules were injected. Check --lora_targets.")

    cam = CameraConditioner(
        CameraConditionConfig(
            cam_dim=13,
            embed_dim=args.cam_embed_dim,
            latent_channels=model.in_dim,
            context_dim=model.text_dim,
            mode=args.cam_mode,
        )
    ).to(wan.device)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
    if not lora_params:
        raise RuntimeError("LoRA parameters are empty after injection.")

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr_lora},
            {"params": cam.parameters(), "lr": args.lr_camera},
        ],
        weight_decay=args.weight_decay,
    )

    context_cond, _ = encode_prompt(wan, args.prompt, n_prompt="")
    base_context = context_cond[0].detach()

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            image = batch["image"].to(wan.device)  # [B,3,H,W]
            camera = batch["camera"].to(wan.device)  # [B,13]

            # Encode images to VAE latent. Wan expects [C,T,H,W], so T=1 here.
            videos = [img.unsqueeze(1) for img in image]
            with torch.no_grad():
                latents = wan.vae.encode(videos)

            losses = []
            for latent, cam_param in zip(latents, camera):
                cam_embed = cam(cam_param.unsqueeze(0)).squeeze(0)
                latent_cond = cam.apply_to_latent(latent, cam_embed)

                # Build context with optional camera token (mode C includes this branch).
                context = cam.apply_to_context(base_context, cam_embed)

                t = torch.randint(0, wan.num_train_timesteps, (1,), device=wan.device)
                noise = torch.randn_like(latent_cond)
                sigma = t.float().view(1, 1, 1, 1) / float(wan.num_train_timesteps)
                x_t = (1.0 - sigma) * latent_cond + sigma * noise

                pred = model([x_t], t=t, context=[context], seq_len=(x_t.shape[1] * x_t.shape[2] * x_t.shape[3]))[0]
                loss = F.mse_loss(pred, noise)
                losses.append(loss)

            total_loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(total_loss.detach().cpu()))

        torch.save({"lora": lora_state_dict(model), "epoch": epoch + 1}, out_dir / f"lora_epoch_{epoch+1:03d}.pt")
        torch.save(cam.state_dict(), out_dir / f"camera_encoder_epoch_{epoch+1:03d}.pt")

    torch.save({"lora": lora_state_dict(model), "epoch": args.epochs}, out_dir / "lora_final.pt")
    torch.save(cam.state_dict(), out_dir / "camera_encoder_final.pt")
    print(f"Done. Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
