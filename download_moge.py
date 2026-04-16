from huggingface_hub import snapshot_download

print("Download MoGe...")
snapshot_download('Ruicheng/moge-vitl')
print("MoGe pronto.")
