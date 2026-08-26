"""Exports real (z, t, labels, cached_y_emb) inputs + real PyTorch reference
output for PointDiT-B/16, for the LiteRT.js browser demo."""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, ".")
from main import get_args_parser
from denoiser import Denoiser

CHECKPOINT = "pretrained/pointditb-512-mixdata-nodinov3-1d42aacf.pth"
IMAGE_DIR = Path("assets/pinecone_bench")
OUT_DIR = Path("tools/litert/web_demo/assets_b16")
OUT_DIR.mkdir(parents=True, exist_ok=True)

args = get_args_parser().parse_args([
    "--model", "PointDiT-B/16", "--feature_embedding_type", "dinov3_vitb16",
    "--proj_dropout", "0.0", "--evaluate_gen", "--img_size", "512",
    "--num_sampling_steps", "3", "--pretrained", CHECKPOINT,
])
model = Denoiser(args)
model.eval()
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
incompatible = model.load_state_dict(checkpoint["model"], strict=False)
missing = [k for k in incompatible.missing_keys if not k.startswith("net.y_embedder.")]
assert not missing and not incompatible.unexpected_keys
ema_state_dict1 = checkpoint["model_ema1"]
sd = dict(model.state_dict())
for name, _ in model.named_parameters():
    if name in ema_state_dict1:
        sd[name] = ema_state_dict1[name]
model.load_state_dict(sd)

image_name = "IMG_7261.png"  # same held-out image used for the L/16 web demo
from PIL import Image
img = Image.open(IMAGE_DIR / image_name).convert("RGB").resize((512, 512), Image.BILINEAR)
arr = np.asarray(img, dtype=np.float32) / 255.0
labels = torch.from_numpy((arr.transpose(2, 0, 1)[None] * 2 - 1).astype(np.float32))

h_patches = w_patches = 512 // model.net.patch_size
with torch.no_grad():
    cached_y_emb = model.net.extract_y_embedding(labels, h_patches, w_patches)

z = torch.from_numpy(np.random.RandomState(0).randn(1, 3, 512, 512).astype(np.float32))
t = torch.tensor([0.3])
with torch.no_grad():
    ref_output = model.net(z, t, labels, cached_y_emb=cached_y_emb)

for name, arr in [("z", z.numpy()), ("t", t.numpy()), ("labels", labels.numpy()),
                  ("cached_y_emb", cached_y_emb.numpy()), ("ref_output", ref_output.numpy())]:
    arr = arr.astype(np.float32)
    arr.tofile(OUT_DIR / f"{name}.bin")
    print(f"{name}: shape={arr.shape} -> {OUT_DIR / f'{name}.bin'} ({arr.nbytes / 1e6:.2f} MB)")

import json
with open(OUT_DIR / "shapes.json", "w") as f:
    json.dump({
        "z": list(z.shape), "t": list(t.shape), "labels": list(labels.shape),
        "cached_y_emb": list(cached_y_emb.shape), "ref_output": list(ref_output.shape),
        "image_name": image_name,
    }, f, indent=2)
print("Wrote shapes.json")
