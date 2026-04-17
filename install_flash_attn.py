#!/usr/bin/env python3
"""
Trova e installa il wheel pre-compilato di flash_attn corretto
per il Python/CUDA/torch attuale. Scarica su /workspace per
evitare errori cross-device link tipici di RunPod.
"""
import subprocess, json, sys, os
from urllib.request import urlopen, Request, urlretrieve

print("==> Cerco wheel flash_attn compatibile...")

import torch
torch_major_minor = ".".join(torch.__version__.split(".")[:2])  # es. "2.5"
torch_tag = f"torch{torch_major_minor}"
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"  # es. "cp311"

print(f"    Python: {py_tag}  |  Torch: {torch_tag}")

req = Request(
    "https://api.github.com/repos/Dao-AILab/flash-attention/releases",
    headers={"User-Agent": "python/flash-attn-installer"}
)
releases = json.loads(urlopen(req).read())

for release in releases:
    for asset in release["assets"]:
        name = asset["name"]
        url  = asset["browser_download_url"]
        if (py_tag in name and torch_tag in name and
                "linux_x86_64" in name and name.endswith(".whl")):
            print(f"==> Trovato: {name}")
            local = f"/workspace/{name}"
            print(f"    Download in {local} ...")
            urlretrieve(url, local)
            print("    Installazione...")
            subprocess.run([sys.executable, "-m", "pip", "install", local], check=True)
            os.remove(local)
            print("==> flash_attn installata.")
            sys.exit(0)

print("ERRORE: nessun wheel compatibile trovato.")
sys.exit(1)
