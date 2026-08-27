"""Exports the exact DINOv3-vitb16 feature extraction PointDiT-B/16 itself was trained against
(4 equally-spaced intermediate layers, normed, concatenated -- see model.py's extract_y_embedding)
to ONNX, for running client-side in the browser.

**This ONNX file is never committed or hosted anywhere -- it embeds Meta's gated DINOv3 weights,
same as pretrained/dinov3/*.pth, just in a different serialized format. Re-serializing a gated
model into ONNX does not relicense it.** Run this yourself, locally, only if you already have
legitimate access to the DINOv3 weights (see the main README's Installation section) -- the
output stays in web_demo/hosted/ (gitignored) for your own local use.

Real reason a generic ONNX DINOv3 export (e.g. the public onnx-community/dinov3-vitb16-...-ONNX
model) doesn't work here: checked its actual graph.output list directly (`onnx.load(...,
load_external_data=False)`) -- it only exposes `last_hidden_state` (the final layer) and
`pooler_output`, not the 4 intermediate layers this exact checkpoint was trained to condition on.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from main import get_args_parser
from denoiser import Denoiser

OUT_DIR = Path("tools/litert/web_demo/hosted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

args = get_args_parser().parse_args([
    "--model", "PointDiT-B/16", "--feature_embedding_type", "dinov3_vitb16",
    "--proj_dropout", "0.0", "--evaluate_gen", "--img_size", "512",
    "--num_sampling_steps", "3", "--pretrained", "pretrained/pointditb-512-mixdata-nodinov3-1d42aacf.pth",
])
model = Denoiser(args)
model.eval()
model.net.y_embedder.load_state_dict(
    torch.load("pretrained/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", map_location="cpu"),
    strict=True,
)
dinov3 = model.net.y_embedder
indices = model.net.dinov3_intermediate_layer_indices
print(f"intermediate layer indices: {indices}")


class Dinov3IntermediateFeatures(nn.Module):
    """ImageNet-normalize + run DINOv3's own get_intermediate_layers(n=indices, norm=True) +
    concatenate -- exactly the DINOv3 portion of model.py's extract_y_embedding, stopping right
    before the pos_embed_y addition (that part is a deterministic sin-cos function of the patch
    grid, not a DINOv3 weight -- computed separately, see get_2d_sincos_pos_embed_web.py)."""
    def __init__(self, dinov3, indices):
        super().__init__()
        self.dinov3 = dinov3
        self.indices = indices
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, y):
        # y: [B, 3, H, W] in [-1, 1] -- identical preprocessing to extract_y_embedding.
        y = (y + 1.0) / 2.0
        y = (y - self.mean) / self.std
        feats = self.dinov3.get_intermediate_layers(y, n=self.indices, norm=True)
        return torch.cat(feats, dim=-1)


wrapper = Dinov3IntermediateFeatures(dinov3, indices).eval()

dummy = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    real_out = wrapper(dummy)
print(f"real output shape: {tuple(real_out.shape)}")

onnx_path = OUT_DIR / "dinov3_vitb16_intermediate.onnx"
# dynamo=False: the legacy TorchScript-based exporter -- more conservative ONNX ops than the
# newer dynamo exporter's default output, which emitted a Split variant (num_outputs attribute)
# this environment's installed onnxruntime couldn't load. Matters doubly here since the same
# file needs to run in a browser via onnxruntime-web, not just this local onnxruntime install.
torch.onnx.export(
    wrapper, (dummy,), str(onnx_path),
    input_names=["pixel_values"], output_names=["y_emb"],
    opset_version=17, do_constant_folding=True, dynamo=False,
)
print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

# Verify immediately: real ONNX Runtime output vs. the real PyTorch output, same input.
import onnxruntime as ort

sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
onnx_out = sess.run(["y_emb"], {"pixel_values": dummy.numpy()})[0]
max_diff = np.abs(onnx_out - real_out.numpy()).max()
print(f"ONNX vs PyTorch max abs diff (random input): {max_diff:.4e}")
assert max_diff < 1e-3, "ONNX export diverges from the real PyTorch DINOv3 computation"
print("Export verified.")

# pos_embed_y is a fixed (requires_grad=False), deterministic sin-cos function of the patch grid
# size -- not a DINOv3 weight, and not something PointDiT was "trained" to produce (it's
# initialized once and frozen). Export the real tensor directly (via the model's own
# interpolate_pos_encoding_y) rather than reimplementing get_2d_sincos_pos_embed by hand in JS --
# a hand reconstruction of this exact formula was tried while building this script and was
# subtly wrong (max diff 86.7 against the real value, vs. 2.4e-4 using the real tensor) --
# shipping the real computed values sidesteps that whole class of bug. Freely
# committable/shippable: it embeds no DINOv3 (or any gated) weights at all.
pos_embed_y = model.net.interpolate_pos_encoding_y(32, 32).detach().numpy().astype(np.float32)
pos_embed_path = OUT_DIR / "pos_embed_y.bin"
pos_embed_y.tofile(pos_embed_path)
print(f"wrote {pos_embed_path} ({pos_embed_y.nbytes / 1e6:.1f} MB, shape {pos_embed_y.shape})")
