"""Verifies convert_hf_dinov3_to_hub.py's remapping is actually correct, not
just plausible: loads the SAME source checkpoint two ways -- via
transformers' own DINOv3ViTModel (guaranteed-correct, since transformers
wrote this exact format) and via the converter's output loaded into the
facebookresearch hub model -- runs both on an identical, fixed random input,
and compares the final normed patch-token features. Real agreement here means
the qkv-fusion/renaming/key_bias=False handling in the converter is right;
it is not assumed from the mapping looking reasonable.
"""
import sys
import torch

sys.path.insert(0, "scripts")
from convert_hf_dinov3_to_hub import convert_hf_to_hub_state_dict

SRC_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dinov3_cam_test"

torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224)

# ---- HF reference ----
from transformers import AutoModel
hf_model = AutoModel.from_pretrained(SRC_DIR)
hf_model.eval()
with torch.no_grad():
    hf_out = hf_model(pixel_values=x).last_hidden_state
# HF token order: [CLS, register_0..3, patch_0..N] (config: num_register_tokens=4)
hf_patch_tokens = hf_out[:, 1 + 4:, :]
hf_cls_token = hf_out[:, 0, :]

# ---- hub model with converted weights ----
hub_model = torch.hub.load("third_party/dinov3", "dinov3_vitl16", source="local", pretrained=False)
hub_model.eval()
hub_sd = convert_hf_to_hub_state_dict(hf_model.state_dict())
missing, unexpected = hub_model.load_state_dict(hub_sd, strict=False)
print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
print("missing (expected: only rope_embed.periods, a deterministic non-learned buffer):", missing)
print("unexpected:", unexpected)
assert unexpected == [], f"converter produced keys the hub model doesn't recognize: {unexpected}"
assert missing == ["rope_embed.periods"], (
    f"expected ONLY rope_embed.periods to be missing (deterministic, not learned); got {missing}"
)

with torch.no_grad():
    hub_feats = hub_model.forward_features(x)
hub_patch_tokens = hub_feats["x_norm_patchtokens"]
hub_cls_token = hub_feats["x_norm_clstoken"]

patch_diff = (hf_patch_tokens - hub_patch_tokens).abs()
cls_diff = (hf_cls_token - hub_cls_token).abs()
print(f"patch tokens: shape {tuple(hub_patch_tokens.shape)} vs HF {tuple(hf_patch_tokens.shape)}, "
      f"max abs diff {patch_diff.max().item():.6e}, mean abs diff {patch_diff.mean().item():.6e}")
print(f"cls token: max abs diff {cls_diff.max().item():.6e}, mean abs diff {cls_diff.mean().item():.6e}")

TOL = 1e-4
ok = patch_diff.max().item() < TOL and cls_diff.max().item() < TOL
print("CONVERSION VERIFIED CORRECT" if ok else "CONVERSION MISMATCH -- DO NOT TRUST")
sys.exit(0 if ok else 1)
