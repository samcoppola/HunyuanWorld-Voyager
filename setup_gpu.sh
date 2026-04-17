#!/bin/bash
# ============================================================
# Esegui UNA VOLTA sul pod GPU dopo setup_cpu.sh.
# Installa flash_attn (wheel pre-compilato, nessuna compilazione).
# ============================================================
set -e

source /workspace/.venv/bin/activate

if python3 -c "import flash_attn" 2>/dev/null; then
    echo "==> flash_attn già installata, skip."
else
    echo "==> Installazione flash_attn..."
    python3 /workspace/HunyuanWorld-Voyager/install_flash_attn.py
fi
