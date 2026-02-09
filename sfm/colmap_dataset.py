from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass
class ColmapImage:
    image_id: int
    qvec: List[float]
    tvec: List[float]
    camera_id: int
    name: str


def _read_colmap_cameras_txt(path: Path) -> Dict[int, ColmapCamera]:
    cameras: Dict[int, ColmapCamera] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cid = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(x) for x in parts[4:]]
        cameras[cid] = ColmapCamera(cid, model, width, height, params)
    return cameras


def _read_colmap_images_txt(path: Path) -> List[ColmapImage]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows: List[ColmapImage] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("#"):
            i += 1
            continue
        parts = lines[i].split()
        # image record line
        image_id = int(parts[0])
        qvec = [float(v) for v in parts[1:5]]
        tvec = [float(v) for v in parts[5:8]]
        camera_id = int(parts[8])
        name = parts[9]
        rows.append(ColmapImage(image_id, qvec, tvec, camera_id, name))
        i += 2  # skip points2d line
    return rows


def _camera_to_k_dist(camera: ColmapCamera) -> Tuple[np.ndarray, np.ndarray]:
    m = camera.model.upper()
    p = camera.params
    if m == "SIMPLE_PINHOLE":
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        dist = np.zeros(5, dtype=np.float32)
    elif m == "PINHOLE":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        dist = np.zeros(5, dtype=np.float32)
    elif m == "SIMPLE_RADIAL":
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        dist = np.array([p[3], 0, 0, 0, 0], dtype=np.float32)
    elif m == "RADIAL":
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        dist = np.array([p[3], p[4], 0, 0, 0], dtype=np.float32)
    elif m == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        dist = np.array([k1, k2, p1, p2, 0], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {camera.model}")
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return k, dist


def _center_crop_and_resize(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    h, w = img.shape[:2]
    target_ratio = out_w / out_h
    src_ratio = w / h
    if src_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        crop = img[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        crop = img[y0:y0 + new_h, :]
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


class ColmapSceneDataset(Dataset):
    """Load real scene photos + camera poses from COLMAP text export."""

    def __init__(
        self,
        image_dir: str,
        cameras_txt: str,
        images_txt: str,
        output_hw: Tuple[int, int],
    ):
        self.image_dir = Path(image_dir)
        self.cameras = _read_colmap_cameras_txt(Path(cameras_txt))
        self.images = _read_colmap_images_txt(Path(images_txt))
        self.output_hw = output_hw

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.images[idx]
        cam = self.cameras[item.camera_id]
        img = np.array(Image.open(self.image_dir / item.name).convert("RGB"))

        # Undistort first, then center crop/resize.
        k, dist = _camera_to_k_dist(cam)
        if np.any(np.abs(dist) > 1e-8):
            img = cv2.undistort(img, k, dist)
        img = _center_crop_and_resize(img, self.output_hw)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        # Build camera vector: qvec(4), tvec(3), intrinsics fx fy cx cy, image size.
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        cam_vec = torch.tensor(
            item.qvec + item.tvec + [fx, fy, cx, cy, float(cam.width), float(cam.height)],
            dtype=torch.float32,
        )

        return {
            "image": img_t,
            "camera": cam_vec,
            "name": item.name,
        }
