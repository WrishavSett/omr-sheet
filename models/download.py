from huggingface_hub import snapshot_download

# Download Qwen2.5-VL-3B-Instruct into ./models/
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    local_dir="models/models--Qwen--Qwen2.5-VL-3B-Instruct",
    local_dir_use_symlinks=False  # Store actual files instead of symlinks
)