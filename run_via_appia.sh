#!/bin/bash
# ============================================================
# Esegui sul pod GPU (A100 SXM 80GB) con Network Volume su /workspace.
# Carica prima l'immagine via_appia.jpg tramite Jupyter nel repo.
# ============================================================
set -e

WORKSPACE=/workspace
REPO_DIR=$WORKSPACE/HunyuanWorld-Voyager
WORK_DIR=$REPO_DIR/workspace_run

# Variabili sovrascrivibili
IMAGE="${IMAGE:-$REPO_DIR/appia_strada.png}"
DIRECTION="${DIRECTION:-forward}"
SEED="${SEED:-42}"
INFER_STEPS="${INFER_STEPS:-50}"

source "$WORKSPACE/.venv/bin/activate"
cd "$REPO_DIR"
git pull

CONDITION_DIR=$WORK_DIR/condition
OUTPUT_DIR=$WORK_DIR/output
mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

# ---- STEP 1: MoGe depth estimation + rendering condizione ----
echo "==> [1/2] Creazione condizione da $IMAGE ..."

IMAGE="$IMAGE" DIRECTION="$DIRECTION" WORK_DIR="$WORK_DIR" python3 -c "
import os, sys, numpy as np
from PIL import Image
import torch
sys.path.insert(0, '.')
from moge.model.v1 import MoGeModel
from data_engine.create_input import (
    camera_list, depth_to_world_coords_points,
    render_from_cameras_videos, create_video_input
)

image_path = os.environ['IMAGE']
direction  = os.environ['DIRECTION']
save_path  = os.environ['WORK_DIR']

image = np.array(Image.open(image_path).convert('RGB').resize((1280, 720)))
image_tensor = torch.tensor(image / 255, dtype=torch.float32, device='cuda:0').permute(2, 0, 1)

print('  Caricamento MoGe...')
moge_model = MoGeModel.from_pretrained('Ruicheng/moge-vitl').to('cuda:0')

print('  Stima profondita...')
output = moge_model.infer(image_tensor)
depth = np.array(output['depth'].detach().cpu())
depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4

Height, Width = image.shape[:2]

intrinsics, extrinsics = camera_list(num_frames=1, type=direction, Width=Width, Height=Height, fx=256, fy=256)
point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
points = point_map.reshape(-1, 3)
colors = image.reshape(-1, 3)

intrinsics, extrinsics = camera_list(num_frames=49, type=direction, Width=Width//2, Height=Height//2, fx=128, fy=128)
render_list, mask_list, depth_list = render_from_cameras_videos(
    points, colors, extrinsics, intrinsics, height=Height//2, width=Width//2
)
create_video_input(
    render_list, mask_list, depth_list,
    os.path.join(save_path, 'condition'),
    separate=True, ref_image=image, ref_depth=depth, Width=Width, Height=Height
)
print('  Condizione pronta.')
"

# ---- STEP 2: Generazione video ----
echo "==> [2/2] Generazione video (A100 80GB, bf16, no cpu offload)..."

PROMPT="A highly realistic cinematic video of ancient Rome, showing a slow forward walking movement along the Via Appia Antica during the Roman Imperial period. The road is paved with large irregular basalt stones (basolato), slightly worn and uneven. On both sides of the road there are monumental Roman tombs, mausoleums, and funerary architectures of different shapes: cylindrical tombs, temple-like structures with columns, pyramidal roofs, statues, and relief decorations. The environment is bright daylight with warm natural sunlight, soft shadows, and a slightly dusty atmosphere. The camera simulates a human walking at eye level, moving slowly forward along the road. The movement is smooth and stable, with slight natural head motion. While moving forward, the camera gently looks to the right and left, alternating focus between the architectural details of the tombs, statues, and decorations. Occasionally, the camera lingers briefly on details such as carved reliefs, columns, or sculptures before returning to the forward path. A few distant Roman figures in tunics walk along the road, adding scale but not distracting from the environment. Vegetation is sparse: some grass, shrubs, and Roman umbrella pine trees (Pinus pinea) in the background. Ultra-realistic textures, physically accurate lighting, cinematic depth of field, historical accuracy, immersive atmosphere. first-person perspective, photorealistic, cinematic, ancient Roman architecture, no modern elements"

python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "$CONDITION_DIR" \
    --prompt "$PROMPT" \
    --i2v-stability \
    --infer-steps "$INFER_STEPS" \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed "$SEED" \
    --cfg-scale 6.0 \
    --embedded-cfg-scale 6.0 \
    --video-size 720 1280 \
    --video-length 49 \
    --save-path "$OUTPUT_DIR" \
    --precision bf16 \
    --vae-precision fp16 \
    --text-encoder-precision fp16 \
    --vae-tiling

echo ""
echo "==> Fatto! Output in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
