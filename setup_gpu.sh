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
    echo "==> Installazione flash_attn dal wheel pre-compilato..."
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    echo "==> flash_attn installata."
fi
