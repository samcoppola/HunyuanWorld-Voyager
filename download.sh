#!/bin/bash
# Scarica i pesi del modello e MoGe sul Network Volume.
# Esegui sul pod CPU dopo aver attivato il venv.
set -e

REPO_DIR=/workspace/HunyuanWorld-Voyager

source /workspace/.venv/bin/activate
cd "$REPO_DIR"

echo "==> [1/2] Download pesi HunyuanWorld-Voyager (~40 GB)..."
hf download tencent/HunyuanWorld-Voyager --local-dir ./ckpts

echo "==> [2/2] Download pesi MoGe (~1 GB)..."
python3 download_moge.py

echo ""
echo "==> Download completo."
ls -lh ckpts/
