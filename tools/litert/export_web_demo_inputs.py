"""Exports one real held-out pinecone image's (z, t, labels, cached_y_emb) inputs
and the real PyTorch-GPU reference output as raw float32 binary files, for the
LiteRT.js browser demo to fetch() directly (no numpy/npz parsing needed in JS)."""
import numpy as np
from pathlib import Path

GPU_RESULTS = "tools/litert/models/bench_pytorch_gpu_results.npz"
OUT_DIR = Path("tools/litert/web_demo/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

gpu_data = np.load(GPU_RESULTS)
image_name = str(gpu_data["image_names"][0])
print(f"Using held-out image: {image_name}")

ref_output = gpu_data[f"ref_{image_name}"].astype(np.float32)
cached_y_emb = gpu_data[f"yemb_{image_name}"].astype(np.float32)

from PIL import Image
IMAGE_DIR = Path("assets/pinecone_bench")
img = Image.open(IMAGE_DIR / image_name).convert("RGB").resize((512, 512), Image.BILINEAR)
arr = np.asarray(img, dtype=np.float32) / 255.0
labels = (arr.transpose(2, 0, 1)[None] * 2 - 1).astype(np.float32)  # [1,3,512,512], [-1,1]

z = np.random.RandomState(0).randn(1, 3, 512, 512).astype(np.float32)
t = np.array([0.3], dtype=np.float32)

for name, arr in [("z", z), ("t", t), ("labels", labels),
                  ("cached_y_emb", cached_y_emb), ("ref_output", ref_output)]:
    arr.tofile(OUT_DIR / f"{name}.bin")
    print(f"{name}: shape={arr.shape} dtype={arr.dtype} -> {OUT_DIR / f'{name}.bin'} "
          f"({arr.nbytes / 1e6:.2f} MB)")

import json
with open(OUT_DIR / "shapes.json", "w") as f:
    json.dump({
        "z": list(z.shape), "t": list(t.shape), "labels": list(labels.shape),
        "cached_y_emb": list(cached_y_emb.shape), "ref_output": list(ref_output.shape),
        "image_name": image_name,
    }, f, indent=2)
print("Wrote shapes.json")
