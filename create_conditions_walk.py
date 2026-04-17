#!/usr/bin/env python3
"""
Genera 3 set di condizioni da UN'unica stima MoGe:
  1. forward     — cammina avanti lungo la Via Appia
  2. look_right  — volge lo sguardo a destra (tomba/statua)
  3. look_left   — volge lo sguardo a sinistra (sweep destra→sinistra)

Le traiettorie sono CONTINUE: la fase 2 parte esattamente dove
finisce la 1, la fase 3 parte dove finisce la 2.

Uso:
    IMAGE=appia_strada.png WORK_DIR=workspace_run python3 create_conditions_walk.py
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_engine.create_input import (
    depth_to_world_coords_points,
    render_from_cameras_videos,
    create_video_input,
)


def build_intrinsics(num_frames, Width, Height, fx, fy):
    cx, cy = Width // 2, Height // 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return np.stack([K] * num_frames)


def build_extrinsics(camera_centers, target_points):
    """
    Costruisce matrici w2c da posizioni camera e punti target nel mondo.
    Stessa convenzione di camera_list() in create_input.py:
      +X = destra, +Y = su (invertito in coord camera), +Z = avanti
    """
    extrinsics = []
    for t, tgt in zip(camera_centers, target_points):
        z = (tgt - t).astype(float)
        z = z / np.linalg.norm(z)
        x = np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-6:
            # Degenerate case: looking straight up/down — use fallback
            x = np.array([0.0, 0.0, 1.0])
            y = np.cross(z, x)
            y = y / np.linalg.norm(y)
        else:
            y = y / norm_y
        x = np.cross(y, z)
        R = np.stack([x, y, z], axis=0)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ t
        extrinsics.append(w2c)
    return np.stack(extrinsics)


def make_trajectory(phase, n=49):
    """
    Restituisce (camera_centers, target_points) per ogni fase.

    Unità relative (non metri). Scala: la camera avanza di 1 unità mentre
    il target è a 100 unità → angolo di campo realistico.

    Fase 1 — avanza dritto:
      cam  [0,0,0]   → [0,0,1]
      tgt  [0,0,100] → [0,0,101]

    Fase 2 — gira la testa a destra (lieve avanzamento):
      cam  [0,0,1]   → [0,0,1.3]
      tgt  [0,0,101] → [100,0,101.3]   (≈45° a destra)

    Fase 3 — spazzata destra→sinistra (lieve avanzamento):
      cam  [0,0,1.3]     → [0,0,1.6]
      tgt  [100,0,101.3] → [-100,0,101.6]   (90° sweep)
    """
    if phase == 1:
        cams = np.linspace([0,   0, 0],   [0,   0, 1],     n)
        tgts = np.linspace([0,   0, 100], [0,   0, 101],   n)
    elif phase == 2:
        cams = np.linspace([0,   0, 1],   [0,   0, 1.3],   n)
        tgts = np.linspace([0,   0, 101], [100, 0, 101.3], n)
    elif phase == 3:
        cams = np.linspace([0,   0, 1.3],   [0,   0,   1.6],   n)
        tgts = np.linspace([100, 0, 101.3], [-100, 0, 101.6],  n)
    else:
        raise ValueError(f"Fase sconosciuta: {phase}")
    return cams.astype(float), tgts.astype(float)


def main():
    image_path = os.environ.get("IMAGE", "appia_strada.png")
    work_dir   = os.environ.get("WORK_DIR", "workspace_run")
    n_frames   = 49
    W, H       = 1280, 720   # full res per MoGe
    rW, rH     = W // 2, H // 2   # res rendering condizioni

    print(f"==> Apertura immagine: {image_path}")
    image = np.array(Image.open(image_path).convert("RGB").resize((W, H)))
    image_tensor = (
        torch.tensor(image / 255, dtype=torch.float32, device="cuda:0")
        .permute(2, 0, 1)
    )

    print("==> Caricamento MoGe...")
    from moge.model.v1 import MoGeModel
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to("cuda:0")

    print("==> Stima profondità...")
    output = moge_model.infer(image_tensor)
    depth = np.array(output["depth"].detach().cpu())
    depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4
    print("    Profondità stimata.")

    # --- Backproiezione point cloud dalla prima posizione (origine) ---
    init_intr = build_intrinsics(1, W, H, 256, 256)
    init_cams, init_tgts = make_trajectory(1, n=1)
    init_extr = build_extrinsics(init_cams, init_tgts)   # shape (1,4,4)

    print("==> Backproiezione point cloud...")
    point_map = depth_to_world_coords_points(depth, init_extr[0], init_intr[0])
    points = point_map.reshape(-1, 3)
    colors = image.reshape(-1, 3)
    print(f"    {points.shape[0]:,} punti nel world space.")

    # --- Rendering 3 fasi ---
    phases = [
        (1, "condition_forward"),
        (2, "condition_right"),
        (3, "condition_left"),
    ]

    for phase, name in phases:
        out_dir = os.path.join(work_dir, name)
        vid_dir = os.path.join(out_dir, "video_input")
        if os.path.isdir(vid_dir):
            print(f"==> Phase {phase} ({name}): già presente, skip.")
            continue

        print(f"==> Phase {phase} ({name}): rendering {n_frames} frames...")
        cams, tgts = make_trajectory(phase, n=n_frames)
        intrinsics = build_intrinsics(n_frames, rW, rH, 128, 128)
        extrinsics = build_extrinsics(cams, tgts)

        render_list, mask_list, depth_list = render_from_cameras_videos(
            points, colors, extrinsics, intrinsics,
            height=rH, width=rW,
        )
        create_video_input(
            render_list, mask_list, depth_list,
            out_dir,
            separate=True,
            ref_image=image,
            ref_depth=depth,
            Width=W,
            Height=H,
        )
        print(f"    Salvato in: {out_dir}")

    print("==> Tutte le condizioni pronte.")


if __name__ == "__main__":
    main()
