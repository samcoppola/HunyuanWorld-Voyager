#!/bin/bash
# ============================================================
# Genera un video Via Appia in 3 fasi continue:
#   1. Cammina avanti
#   2. Guarda a destra
#   3. Spazzata destra→sinistra
#
# Richiede 2× A100 80GB (NUM_GPUS=2 ULYSSES=2).
# Carica appia_strada.png via JupyterLab prima di lanciare.
#
# Uso:
#   NUM_GPUS=2 ULYSSES=2 bash run_via_appia_walk.sh
# ============================================================
set -e

WORKSPACE=/workspace
REPO_DIR=$WORKSPACE/HunyuanWorld-Voyager
WORK_DIR=$REPO_DIR/workspace_walk

IMAGE="${IMAGE:-$REPO_DIR/appia_strada.png}"
SEED="${SEED:-42}"
INFER_STEPS="${INFER_STEPS:-50}"
NUM_GPUS="${NUM_GPUS:-2}"
ULYSSES="${ULYSSES:-2}"

source "$WORKSPACE/.venv/bin/activate"
cd "$REPO_DIR"
git pull

export MODEL_BASE=$REPO_DIR/ckpts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$WORK_DIR"

# ---- STEP 1: MoGe depth + rendering 3 condizioni ----
echo "==> [1/5] Creazione condizioni da $IMAGE ..."
IMAGE="$IMAGE" WORK_DIR="$WORK_DIR" python3 "$REPO_DIR/create_conditions_walk.py"

# ---- Prompt per ogni fase ----
PROMPT_FORWARD="A highly realistic cinematic video of ancient Rome, showing a slow forward walking movement along the Via Appia Antica during the Roman Imperial period. The road is paved with large irregular basalt stones (basolato), slightly worn and uneven. On both sides of the road there are monumental Roman tombs, mausoleums, and funerary architectures: cylindrical tombs, temple-like structures with columns, pyramidal roofs, statues, and relief decorations. The environment is bright daylight with warm natural sunlight, soft shadows, and a slightly dusty atmosphere. The camera simulates a human walking at eye level, moving slowly and steadily forward along the road. Movement is smooth and stable with slight natural head motion. Sparse vegetation: grass, shrubs, and Roman umbrella pine trees (Pinus pinea) in the background. Ultra-realistic textures, physically accurate lighting, cinematic depth of field, historical accuracy, immersive atmosphere. first-person perspective, photorealistic, cinematic, ancient Roman architecture, no modern elements"

PROMPT_RIGHT="A highly realistic cinematic video of ancient Rome along the Via Appia Antica. The camera is at eye level and slowly turns its gaze to the RIGHT, revealing a monumental Roman tomb or mausoleum on the right side of the road. The funerary architecture features cylindrical forms, columns, carved relief decorations, and marble statues. Bright daylight with warm natural sunlight, soft directional shadows, slightly dusty atmosphere. The movement is a slow, natural head turn to the right, as if a person walking is drawn to look at an impressive monument. Ultra-realistic textures, physically accurate lighting, cinematic depth of field, historical accuracy. first-person perspective, photorealistic, cinematic, ancient Roman architecture, no modern elements"

PROMPT_LEFT="A highly realistic cinematic video of ancient Rome along the Via Appia Antica. The camera is at eye level and slowly sweeps its gaze from the RIGHT side back to the LEFT side, passing through the forward direction, revealing funerary monuments and mausoleums on the left side of the road: cylindrical tombs, temple-like structures with columns, pyramidal roofs, carved reliefs, and statues. Bright daylight with warm natural sunlight, slightly dusty atmosphere. The movement is a slow, natural head sweep as if a person walking scans the monuments on both sides. Ultra-realistic textures, physically accurate lighting, cinematic depth of field, historical accuracy. first-person perspective, photorealistic, cinematic, ancient Roman architecture, no modern elements"

# ---- STEP 2/3/4: Generazione video per ogni fase ----
PHASE_NUM=1
for PHASE in forward right left; do
    PHASE_NUM=$((PHASE_NUM + 1))
    COND_DIR=$WORK_DIR/condition_$PHASE
    OUT_DIR=$WORK_DIR/output_$PHASE
    mkdir -p "$OUT_DIR"

    # Skip se già generato
    EXISTING=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | head -1)
    if [ -n "$EXISTING" ]; then
        echo "==> [$PHASE_NUM/5] $PHASE già generato: $EXISTING — skip."
        continue
    fi

    case $PHASE in
        forward) PROMPT="$PROMPT_FORWARD" ;;
        right)   PROMPT="$PROMPT_RIGHT" ;;
        left)    PROMPT="$PROMPT_LEFT" ;;
    esac

    echo "==> [$PHASE_NUM/5] Generazione video: $PHASE (${NUM_GPUS}× GPU, bf16)..."
    torchrun --nproc_per_node="$NUM_GPUS" sample_image2video.py \
        --model HYVideo-T/2 \
        --model-base "$REPO_DIR/ckpts" \
        --input-path "$COND_DIR" \
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
        --save-path "$OUT_DIR" \
        --precision bf16 \
        --vae-precision fp16 \
        --text-encoder-precision fp16 \
        --vae-tiling \
        --ulysses-degree "$ULYSSES" \
        --ring-degree 1
done

# ---- STEP 5: Crop RGB (metà superiore) + concatenazione con crossfade ----
echo "==> [5/5] Crop RGB e concatenazione finale..."
FINAL_DIR=$WORK_DIR/final
mkdir -p "$FINAL_DIR"

for PHASE in forward right left; do
    RAW_VIDEO=$(ls "$WORK_DIR/output_$PHASE"/*.mp4 | sort | tail -1)
    RGB_VIDEO="$FINAL_DIR/${PHASE}_rgb.mp4"

    if [ -f "$RGB_VIDEO" ]; then
        echo "    $PHASE crop già presente, skip."
    else
        echo "    Crop RGB: $PHASE..."
        ffmpeg -y -i "$RAW_VIDEO" \
            -vf "crop=iw:ih/2:0:0" \
            -c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast \
            "$RGB_VIDEO"
    fi
done

FINAL_VIDEO="$FINAL_DIR/walk_appia_final.mp4"
if [ -f "$FINAL_VIDEO" ]; then
    echo "    Video finale già presente: $FINAL_VIDEO"
else
    echo "    Concatenazione con crossfade..."

    # Durata di ogni clip (in secondi, formato decimale)
    DUR=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$FINAL_DIR/forward_rgb.mp4")

    # Crossfade 0.3 s tra i clip
    XFADE_DUR=0.3

    # Offset per il primo xfade: fine del primo clip
    OFFSET1=$(echo "scale=3; $DUR - $XFADE_DUR" | bc)
    # Offset per il secondo xfade: fine del secondo clip nel flusso già fuso
    OFFSET2=$(echo "scale=3; $DUR * 2 - $XFADE_DUR * 2" | bc)

    ffmpeg -y \
        -i "$FINAL_DIR/forward_rgb.mp4" \
        -i "$FINAL_DIR/right_rgb.mp4" \
        -i "$FINAL_DIR/left_rgb.mp4" \
        -filter_complex "
            [0:v][1:v]xfade=transition=fade:duration=${XFADE_DUR}:offset=${OFFSET1}[v01];
            [v01][2:v]xfade=transition=fade:duration=${XFADE_DUR}:offset=${OFFSET2}[vout]" \
        -map "[vout]" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast \
        "$FINAL_VIDEO"
fi

echo ""
echo "==> Completato!"
echo "    Video finale: $FINAL_VIDEO"
echo "    Clip individuali RGB:"
ls -lh "$FINAL_DIR/"
