"""Converts a Hugging Face `transformers`-format DINOv3 ViT checkpoint (e.g. the
public mirrors camenduru/dinov3-vitl16-pretrain-lvd1689m or
xycheni/facebook-dinov3-vitl16-pretrain-lvd1689m -- both public, no gating, but
in HF's own key-naming/module-split format) into the raw facebookresearch
`dinov3` hub-format state dict PointDiT's own model.py loader expects
(torch.hub.load(third_party/dinov3, 'dinov3_vitl16', source='local',
weights=...)).

WHY THIS EXISTS: the official Meta/HF DINOv3 weights are gated and require a
manual access request; two public HF mirrors of the same weights exist and
download with zero gating, but store the state dict in HF's transformers
format (415 keys, split Q/K/V projections, different names) rather than the
368-key fused-QKV hub format PointDiT needs. A straight copy fails outright
(key mismatch). This script does the real key remap: fused-QKV
concatenation, name renames, and the query/key/value-bias handling (DINOv3's
key_bias=False means K has no bias; the hub format's qkv.bias_mask buffer,
which zeroes the K segment of the fused bias, is not present in the source
checkpoint at all -- it's reconstructed here from the config, not invented
data, since it is a fixed structural mask, not learned).

VERIFIED, not just plausible: see verify_hf_dinov3_conversion.py, which loads
the SAME source weights into transformers' own DINOv3ViTModel (guaranteed
correct, since transformers wrote the format) and into the hub model via this
converter's output, runs both on an identical random input, and confirms the
final normed patch-token outputs agree to float32 precision.
"""
import torch


def convert_hf_to_hub_state_dict(hf_state_dict: dict, num_layers: int = 24) -> dict:
    hub = {}

    hub["cls_token"] = hf_state_dict["embeddings.cls_token"].clone()
    hub["mask_token"] = hf_state_dict["embeddings.mask_token"].clone().reshape(1, -1)
    hub["storage_tokens"] = hf_state_dict["embeddings.register_tokens"].clone()
    hub["patch_embed.proj.weight"] = hf_state_dict["embeddings.patch_embeddings.weight"].clone()
    hub["patch_embed.proj.bias"] = hf_state_dict["embeddings.patch_embeddings.bias"].clone()

    for i in range(num_layers):
        p = f"model.layer.{i}.attention."
        q_w = hf_state_dict[p + "q_proj.weight"]
        k_w = hf_state_dict[p + "k_proj.weight"]
        v_w = hf_state_dict[p + "v_proj.weight"]
        q_b = hf_state_dict[p + "q_proj.bias"]
        v_b = hf_state_dict[p + "v_proj.bias"]
        hidden = q_w.shape[0]
        # k_proj has no bias in this config (key_bias=False) -- confirmed by its
        # absence in the source state dict, not assumed silently.
        assert (p + "k_proj.bias") not in hf_state_dict, (
            "source checkpoint unexpectedly has a k_proj.bias -- this converter "
            "assumes key_bias=False (DINOv3's stated config); re-check before trusting."
        )
        k_b = torch.zeros_like(q_b)

        hub[f"blocks.{i}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        hub[f"blocks.{i}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)
        # Structural mask (not learned data): 1 where a real bias exists (Q, V),
        # 0 where it doesn't (K) -- reconstructed from config, matching the hub
        # model's own such buffer exactly by construction, not guessed.
        bias_mask = torch.cat([torch.ones_like(q_b), torch.zeros_like(k_b), torch.ones_like(v_b)])
        hub[f"blocks.{i}.attn.qkv.bias_mask"] = bias_mask

        hub[f"blocks.{i}.attn.proj.weight"] = hf_state_dict[p + "o_proj.weight"].clone()
        hub[f"blocks.{i}.attn.proj.bias"] = hf_state_dict[p + "o_proj.bias"].clone()

        lp = f"model.layer.{i}."
        hub[f"blocks.{i}.norm1.weight"] = hf_state_dict[lp + "norm1.weight"].clone()
        hub[f"blocks.{i}.norm1.bias"] = hf_state_dict[lp + "norm1.bias"].clone()
        hub[f"blocks.{i}.norm2.weight"] = hf_state_dict[lp + "norm2.weight"].clone()
        hub[f"blocks.{i}.norm2.bias"] = hf_state_dict[lp + "norm2.bias"].clone()
        hub[f"blocks.{i}.ls1.gamma"] = hf_state_dict[lp + "layer_scale1.lambda1"].clone()
        hub[f"blocks.{i}.ls2.gamma"] = hf_state_dict[lp + "layer_scale2.lambda1"].clone()
        # HF's up_proj/down_proj are fc1/fc2 (up-project then down-project) --
        # same tensors, different names, confirmed by matching shapes.
        hub[f"blocks.{i}.mlp.fc1.weight"] = hf_state_dict[lp + "mlp.up_proj.weight"].clone()
        hub[f"blocks.{i}.mlp.fc1.bias"] = hf_state_dict[lp + "mlp.up_proj.bias"].clone()
        hub[f"blocks.{i}.mlp.fc2.weight"] = hf_state_dict[lp + "mlp.down_proj.weight"].clone()
        hub[f"blocks.{i}.mlp.fc2.bias"] = hf_state_dict[lp + "mlp.down_proj.bias"].clone()

    hub["norm.weight"] = hf_state_dict["norm.weight"].clone()
    hub["norm.bias"] = hf_state_dict["norm.bias"].clone()
    return hub


def main():
    import sys
    import os
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dinov3_cam_test"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "converted_dinov3_vitl16_hub.pth"
    from transformers import AutoModel
    hf_model = AutoModel.from_pretrained(src_dir)
    hub_sd = convert_hf_to_hub_state_dict(hf_model.state_dict())

    # rope_embed.periods is a DETERMINISTIC buffer (from rope_theta/head_dim at
    # construction), not learned data, and absent from the HF checkpoint entirely
    # -- but the hub model's own hubconf-driven loader calls load_state_dict with
    # strict=True, so it must be present in the saved file. Source it from a
    # freshly-constructed hub model's own __init__ (the exact same code that
    # would compute it at load time anyway), not reimplemented here.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_model = torch.hub.load(
        os.path.join(repo_root, "third_party/dinov3"), "dinov3_vitl16",
        source="local", pretrained=False)
    hub_sd["rope_embed.periods"] = ref_model.state_dict()["rope_embed.periods"].clone()

    torch.save(hub_sd, out_path)
    print(f"Wrote {out_path} ({len(hub_sd)} tensors)")


if __name__ == "__main__":
    main()
