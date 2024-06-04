from huggingface_hub import snapshot_download
snapshot_download(repo_id="sentence-transformers/distiluse-base-multilingual-cased-v2", ignore_patterns=["*.msgpack", "*.h5", '*.safetensors'], cache_dir='workspace/plm/sbertv2')