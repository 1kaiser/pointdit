"""Real LiteRT conversion attempt + verification for PointDiT-B/16's per-step denoiser
network -- the actual hot-loop call (`net.forward(z, t, y, cached_y_emb=...)`) traced from
denoiser.py's own `generate()`/`_forward_sample()`, not a guessed input shape.

Scoping rationale (traced from the real sampling loop, not assumed): during real generation,
the frozen DINOv3 image embedding is computed ONCE via `net.extract_y_embedding(...)` and then
reused every euler step via `cached_y_emb` -- passing it means `net.forward()` takes the
`if cached_y_emb is not None:` branch and skips the `F.interpolate(..., mode='bilinear')`
image-downsample entirely (model.py line 583). The ONLY interpolate left on this exact path is
`interpolate_pos_encoding`'s bicubic resize (line 447), which runs unconditionally every call.
This is the real, per-step hot loop worth LiteRT-converting for a latency benchmark -- the
one-time DINOv3 encoding is a separate, much smaller share of total generation time (with
num_sampling_steps=3, DINOv3 runs once, the denoiser net runs 3x).

Follows this machine's own established verification precedent (damnet_litert_attempt.py):
litert_torch.convert() completing without raising is NOT sufficient evidence of a correct
conversion -- that same pattern silently produced a 0.18 max-abs-diff wrong result on a model
using bilinear-interpolate/transposed-conv. Real parity is only established by diffing the
LiteRT output against the real PyTorch output on the SAME real input.
"""
import sys
import torch

sys.path.insert(0, ".")
from main import get_args_parser
from denoiser import Denoiser

CHECKPOINT = "pretrained/pointditb-512-mixdata-nodinov3-1d42aacf.pth"

args = get_args_parser().parse_args([
    "--model", "PointDiT-B/16", "--feature_embedding_type", "dinov3_vitb16",
    "--proj_dropout", "0.0", "--evaluate_gen", "--img_size", "512",
    "--num_sampling_steps", "3", "--pretrained", CHECKPOINT,
])

device = "cpu"  # LiteRT conversion traces on CPU -- matches this project's own established
                 # JAX->LiteRT precedent (jax_litert_conversion_gotchas memory note) of always
                 # converting on CPU, not just carried over blindly: litert_torch's own
                 # torch.export-based tracing has no CUDA-specific requirement here either way,
                 # and CPU avoids any GPU-memory contention with the tracing/conversion process.
model = Denoiser(args)
model.eval()

print("Loading checkpoint...")
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
incompatible = model.load_state_dict(checkpoint["model"], strict=False)
missing = [k for k in incompatible.missing_keys if not k.startswith("net.y_embedder.")]
assert not missing and not incompatible.unexpected_keys, (
    f"unexpected state_dict mismatch: missing={missing} unexpected={incompatible.unexpected_keys}"
)
# Restore EMA1 weights the same way main.py does for evaluation, since that's what the real
# benchmark should reflect.
ema_state_dict1 = checkpoint["model_ema1"]
sd = dict(model.state_dict())
for name, _ in model.named_parameters():
    if name in ema_state_dict1:
        sd[name] = ema_state_dict1[name]
model.load_state_dict(sd)
model.to(device)
print("Checkpoint loaded (EMA1 weights).")

# ---- Real example input, same shapes/ranges as the real sampling loop ----
torch.manual_seed(0)
H = W = 512
labels = torch.rand(1, 3, H, W, device=device) * 2 - 1  # matches generate()'s [-1, 1] image range
h_patches = H // model.net.patch_size
w_patches = W // model.net.patch_size
with torch.no_grad():
    cached_y_emb = model.net.extract_y_embedding(labels, h_patches, w_patches)
z = torch.randn(1, 3, H, W, device=device)
t = torch.tensor([0.3], device=device)

net = model.net
net.eval()

with torch.no_grad():
    torch_out = net(z, t, labels, cached_y_emb=cached_y_emb)
print("PyTorch forward OK, output shape:", tuple(torch_out.shape), "dtype:", torch_out.dtype)
print("output stats: mean", torch_out.mean().item(), "std", torch_out.std().item(),
      "max abs", torch_out.abs().max().item())

print("\nAttempting litert_torch.convert() on net.forward(z, t, labels, cached_y_emb=cached_y_emb)...")
try:
    import litert_torch
    with torch.no_grad():
        edge_model = litert_torch.convert(net, (z, t, labels), {"cached_y_emb": cached_y_emb})
    out_path = "tools/litert/models/pointdit_b16_denoiser_step.tflite"
    edge_model.export(out_path)
    print(f"CONVERSION SUCCEEDED -> {out_path}")

    from ai_edge_litert.interpreter import Interpreter
    import numpy as np
    interp = Interpreter(model_path=out_path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()
    out_d = interp.get_output_details()
    print("LiteRT input details:", [(d["name"], d["shape"], d["dtype"]) for d in in_d])
    print("LiteRT output details:", [(d["name"], d["shape"], d["dtype"]) for d in out_d])

    # Match inputs to the interpreter's own declared order by shape, not by assumption.
    real_inputs = {"z": z.numpy().astype(np.float32), "t": t.numpy().astype(np.float32),
                   "labels": labels.numpy().astype(np.float32),
                   "cached_y_emb": cached_y_emb.numpy().astype(np.float32)}
    for d in in_d:
        matched = [v for v in real_inputs.values() if list(v.shape) == list(d["shape"])]
        assert matched, f"no real input matches LiteRT's expected shape {d['shape']}"
        interp.set_tensor(d["index"], matched[0])
    interp.invoke()
    litert_out = interp.get_tensor(out_d[0]["index"])

    torch_out_np = torch_out.numpy()
    diff = np.abs(litert_out - torch_out_np)
    print(f"\nLiteRT vs PyTorch: shape litert={litert_out.shape} torch={torch_out_np.shape}")
    print(f"max abs diff = {diff.max():.4e} (vs. output magnitude ~{np.abs(torch_out_np).mean():.4e})")
    print(f"mean abs diff = {diff.mean():.4e}")
    REL_TOL = 0.02  # 2% of typical output magnitude -- matches the "usable" bar the DAM-Net
                     # precedent's own tiny-ViT comparison (~5e-6) vastly exceeds; this is a
                     # deliberately loose bar for a first pass, tightened once/if this passes.
    ok = diff.max() < REL_TOL * np.abs(torch_out_np).mean()
    print("USABLE CONVERSION" if ok else "NOT USABLE -- real numerical divergence, do not benchmark this")
except Exception as e:
    import traceback
    print("CONVERSION FAILED")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    traceback.print_exc()
