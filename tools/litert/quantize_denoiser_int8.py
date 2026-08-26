"""Real int8 static (activation+weight) quantization of the PointDiT-L/16 per-step
denoiser TFLite model (pointdit_l16_denoiser_step.tflite, exported by
convert_denoiser_litert.py), calibrated on REAL sampling trajectories from the
NeRF-360 `pinecone` scene (99 real images, 1kaiser/NERF_360 on Hugging Face).

Calibration data is built by actually running the real 3-step euler ODE sampler
(replicating denoiser.py's own generate()/`_euler_step` logic exactly, not a
synthetic/random substitute) on the first CALIB_N images, capturing the real
(z, t, labels, cached_y_emb) tuple the network actually sees at each of the 3
real steps -- 3 * CALIB_N calibration samples, all in-distribution.
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, ".")
from main import get_args_parser
from denoiser import Denoiser

CHECKPOINT = "pretrained/pointditl-512-mixdata-nodinov3-240c1a4f.pth"
FLOAT_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step.tflite"
INT8_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step_int8.tflite"
IMAGE_DIR = Path("assets/pinecone_bench")
CALIB_N = 20  # first 20 images for calibration; remaining 79 held out for benchmark/verify

device = "cpu"


def load_model():
    args = get_args_parser().parse_args([
        "--model", "PointDiT-L/16", "--feature_embedding_type", "dinov3_vitl16",
        "--proj_dropout", "0.0", "--evaluate_gen", "--img_size", "512",
        "--num_sampling_steps", "3", "--pretrained", CHECKPOINT,
    ])
    model = Denoiser(args)
    model.eval()
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith("net.y_embedder.")]
    assert not missing and not incompatible.unexpected_keys
    ema_state_dict1 = checkpoint["model_ema1"]
    sd = dict(model.state_dict())
    for name, _ in model.named_parameters():
        if name in ema_state_dict1:
            sd[name] = ema_state_dict1[name]
    model.load_state_dict(sd)
    model.to(device)
    return model


def load_image_512(path) -> torch.Tensor:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((512, 512), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [0, 1], matches WildImagesDataset
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 512, 512]
    return t * 2 - 1  # to [-1, 1], matches denoiser.generate()'s expected `labels` range


def real_step_trajectory(model, labels, steps=3):
    """Replicates denoiser.py's Denoiser.generate() euler loop EXACTLY (same
    timesteps schedule, same zero-init since generate_noise_scale defaults to
    0.0, same cached_y_emb reuse), returning the real (z, t, labels,
    cached_y_emb) tuple fed to net() at each of the `steps` real calls."""
    net = model.net
    H = W = 512
    bsz = 1
    z = model.args.generate_noise_scale * torch.randn(bsz, 3, H, W)
    timesteps = torch.linspace(0.0, 1.0, steps + 1)
    h_patches, w_patches = H // net.patch_size, W // net.patch_size
    with torch.no_grad():
        cached_y_emb = net.extract_y_embedding(labels, h_patches, w_patches)

    real_calls = []
    with torch.no_grad():
        for i in range(steps - 1):
            t, t_next = timesteps[i], timesteps[i + 1]
            t_batch = t.expand(bsz)
            real_calls.append((z.clone(), t_batch.clone(), labels.clone(), cached_y_emb.clone()))
            x_cond = net(z, t_batch, labels, cached_y_emb=cached_y_emb)
            v_pred = (x_cond - z) / (1.0 - t).clamp_min(model.args.sample_t_eps)
            z = z + (t_next - t) * v_pred
        if steps > 0:
            t, t_next = timesteps[-2], timesteps[-1]
            t_batch = t.expand(bsz)
            real_calls.append((z.clone(), t_batch.clone(), labels.clone(), cached_y_emb.clone()))
            x_cond = net(z, t_batch, labels, cached_y_emb=cached_y_emb)
            v_pred = (x_cond - z) / (1.0 - t).clamp_min(model.args.sample_t_eps)
            z = z + (t_next - t) * v_pred
    return real_calls, z


def main():
    model = load_model()
    image_paths = sorted(IMAGE_DIR.glob("*.png"))
    print(f"Found {len(image_paths)} pinecone images; using first {CALIB_N} for calibration.")
    assert len(image_paths) >= CALIB_N

    calibration_samples = []
    for p in image_paths[:CALIB_N]:
        labels = load_image_512(p)
        real_calls, _ = real_step_trajectory(model, labels)
        for z, t, y, cached_y_emb in real_calls:
            calibration_samples.append({
                "args_0": z.numpy().astype(np.float32),
                "args_1": t.numpy().astype(np.float32),
                "args_2": y.numpy().astype(np.float32),
                "cached_y_emb": cached_y_emb.numpy().astype(np.float32),
            })
    print(f"Built {len(calibration_samples)} real calibration samples "
          f"({CALIB_N} images x 3 real euler steps each).")

    from ai_edge_quantizer import quantizer, qtyping

    qt = quantizer.Quantizer(FLOAT_TFLITE)
    qt.add_static_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=8, weight_num_bits=8,
    )
    print("Recipe:", qt.get_quantization_recipe())

    assert qt.need_calibration, "expected this int8-static recipe to require calibration"
    calibration_data = {"serving_default": calibration_samples}
    print("Calibrating...")
    calibration_result = qt.calibrate(calibration_data)

    print("Quantizing...")
    result = qt.quantize(calibration_result, serialize_to_path=INT8_TFLITE)
    print(f"Wrote {INT8_TFLITE}")

    import os
    fp32_size = os.path.getsize(FLOAT_TFLITE)
    int8_size = os.path.getsize(INT8_TFLITE)
    print(f"fp32 size: {fp32_size / 1e6:.1f} MB, int8 size: {int8_size / 1e6:.1f} MB "
          f"({fp32_size / int8_size:.2f}x smaller)")


if __name__ == "__main__":
    main()
