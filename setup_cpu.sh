#!/bin/bash
# ============================================================
# Esegui su un pod CPU di RunPod (Network Volume montato su /workspace).
# Crea il venv e scarica tutti i modelli — una volta sola.
# ============================================================
set -e

WORKSPACE=/workspace
REPO_DIR=$WORKSPACE/HunyuanWorld-Voyager

# Clona il repo se non c'è ancora
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/samcoppola/HunyuanWorld-Voyager "$REPO_DIR"
else
    cd "$REPO_DIR" && git pull
fi
cd "$REPO_DIR"

# Crea il venv sul volume (persiste tra i pod)
if [ ! -f "$WORKSPACE/.venv/setup_complete" ]; then
    echo "==> Creazione venv in $WORKSPACE/.venv ..."
    python3.11 -m venv "$WORKSPACE/.venv"
    source "$WORKSPACE/.venv/bin/activate"
    pip install --upgrade pip

    # PyTorch CUDA 12.4 — versione fissa, evita che pip prenda torch con CUDA 13.0
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

    # Dipendenze principali (torch/torchvision già installati sopra)
    pip install \
        opencv-python==4.9.0.80 \
        diffusers==0.31.0 \
        "huggingface_hub[cli]" \
        accelerate==1.1.1 \
        einops==0.7.0 \
        loguru==0.7.2 \
        imageio==2.34.0 \
        imageio-ffmpeg==0.5.1 \
        safetensors==0.4.3 \
        peft==0.13.2 \
        pyexr==0.5.0 \
        pandas==2.0.3 \
        scipy \
        mmengine \
        timm \
        trimesh \
        transformers==4.44.2 \
        "git+https://github.com/openai/CLIP.git" \
        "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

    # MoGe (depth estimator)
    pip install git+https://github.com/microsoft/MoGe.git

    # hf_transfer — richiesto da RunPod (HF_HUB_ENABLE_HF_TRANSFER=1 di default)
    pip install hf_transfer

    # Forza numpy<2 — opencv 4.9 non è compatibile con numpy 2.x
    pip install "numpy<2"

    pip install tensorboard==2.19.0

    touch "$WORKSPACE/.venv/setup_complete"
    echo "==> venv pronto."
else
    echo "==> venv già esistente, skip installazione."
fi

source "$WORKSPACE/.venv/bin/activate"

# Scarica i pesi del modello (~40 GB)
echo "==> Download pesi HunyuanWorld-Voyager ..."
huggingface-cli download tencent/HunyuanWorld-Voyager --local-dir "$REPO_DIR/ckpts"

# Pre-scarica pesi MoGe (~1 GB) nella cache HuggingFace
echo "==> Download pesi MoGe ..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Ruicheng/moge-vitl')
print('MoGe pronto.')
"

echo ""
echo "==> Setup CPU completato. Puoi terminare il pod CPU."
echo "    Modelli in: $REPO_DIR/ckpts"
