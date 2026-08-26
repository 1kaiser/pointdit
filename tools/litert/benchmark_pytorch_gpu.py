"""PyTorch-GPU half of the backend benchmark -- run in the `pointdit` conda env
(torch==2.7.0+cu128, the build that actually supports this machine's Blackwell
GPU; the `pointdit_litert` env's torch==2.12.1+cu126 does not -- confirmed live,
"no kernel image is available for execution on the device"). Saves timing +
real reference outputs to disk so benchmark_litert_backends.py (run in the
other env) can do the accuracy comparison without needing GPU-capable torch.
"""
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, ".")
from main import get_args_parser
from denoiser import Denoiser

CHECKPOINT = "pretrained/pointditl-512-mixdata-nodinov3-240c1a4f.pth"
IMAGE_DIR = Path("assets/pinecone_bench")
CALIB_N = 20
N_BENCH_IMAGES = 20
N_WARMUP = 3
OUT_PATH = "tools/litert/models/bench_pytorch_gpu_results.npz"


def load_model(device):
    args = get_args_parser().parse_args([
        "--model", "PointDiT-L/16", "--feature_embedding_type", "dinov3_vitl16",
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
    model.to(device)
    return model


def load_image_512(path, device):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((512, 512), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2 - 1


def main():
    image_paths = sorted(IMAGE_DIR.glob("*.png"))
    held_out = image_paths[CALIB_N:CALIB_N + N_BENCH_IMAGES]
    print(f"{len(held_out)} held-out images (not used for int8 calibration)")

    assert torch.cuda.is_available(), "expected a working CUDA device in the pointdit env"
    gpu = "cuda"
    model_gpu = load_model(gpu)

    ref_outputs = {}
    ref_cached_y_emb = {}
    times_gpu = []
    for i, p in enumerate(held_out):
        labels = load_image_512(p, gpu)
        h_patches = w_patches = 512 // model_gpu.net.patch_size
        with torch.no_grad():
            cached_y_emb = model_gpu.net.extract_y_embedding(labels, h_patches, w_patches)
        z = torch.from_numpy(np.random.RandomState(0).randn(1, 3, 512, 512).astype(np.float32)).to(gpu)
        t = torch.tensor([0.3], device=gpu)

        if i < N_WARMUP:
            with torch.no_grad():
                model_gpu.net(z, t, labels, cached_y_emb=cached_y_emb)
            torch.cuda.synchronize()
            continue

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model_gpu.net(z, t, labels, cached_y_emb=cached_y_emb)
        torch.cuda.synchronize()
        times_gpu.append(time.perf_counter() - t0)
        ref_outputs[p.name] = out.cpu().numpy()
        ref_cached_y_emb[p.name] = cached_y_emb.cpu().numpy()

    print(f"Timed {len(times_gpu)} calls after {N_WARMUP} warmup.")
    times_gpu = np.array(times_gpu) * 1000
    print(f"pytorch_cuda: mean={times_gpu.mean():.2f} ms  median={np.median(times_gpu):.2f} ms  "
          f"std={times_gpu.std():.2f} ms")

    np.savez(OUT_PATH, times_ms=times_gpu,
             image_names=np.array(list(ref_outputs.keys())),
             **{f"ref_{k}": v for k, v in ref_outputs.items()},
             **{f"yemb_{k}": v for k, v in ref_cached_y_emb.items()})
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
