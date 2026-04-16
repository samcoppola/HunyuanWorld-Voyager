#!/bin/bash
# ============================================================
# Esegui UNA VOLTA sul pod GPU dopo setup_cpu.sh.
# Installa flash_attn (richiede compilazione CUDA).
# ============================================================
set -e

source /workspace/.venv/bin/activate

if python3 -c "import flash_attn" 2>/dev/null; then
    echo "==> flash_attn già installata, skip."
else
    echo "==> Installazione flash_attn (compilazione ~5 min)..."
    export PIP_CACHE_DIR=/workspace/.pip_cache
    pip install flash_attn --no-build-isolation
    echo "==> flash_attn installata."
fi
