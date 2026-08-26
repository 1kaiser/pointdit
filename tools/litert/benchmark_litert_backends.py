"""CPU half of the backend benchmark -- PyTorch-CPU, LiteRT-fp32 (XNNPACK CPU),
LiteRT-int8 (XNNPACK CPU) -- run in the `pointdit_litert` conda env. Loads the
PyTorch-GPU reference outputs/timing from bench_pytorch_gpu_results.npz
(produced by benchmark_pytorch_gpu.py in the `pointdit` env, which has the
Blackwell-compatible torch build) for the accuracy comparison and the final
combined report.
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
FLOAT_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step.tflite"
INT8_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step_int8.tflite"
DYNAMIC_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step_dynamic.tflite"
WEIGHTONLY_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step_weightonly.tflite"
DYNAMIC_INT4_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step_dynamic_int4.tflite"
WEIGHTONLY_INT4_TFLITE = "tools/litert/models/pointdit_l16_denoiser_step_weightonly_int4.tflite"
IMAGE_DIR = Path("assets/pinecone_bench")
CALIB_N = 20
N_BENCH_IMAGES = 20
N_WARMUP = 3
GPU_RESULTS = "tools/litert/models/bench_pytorch_gpu_results.npz"


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


def load_image_512(path):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((512, 512), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t * 2 - 1


def main():
    image_paths = sorted(IMAGE_DIR.glob("*.png"))
    gpu_data = np.load(GPU_RESULTS)
    ref_outputs = {name: gpu_data[f"ref_{name}"] for name in gpu_data["image_names"]}
    # The GPU run's own N_WARMUP images were never timed/saved there, so only the
    # named images actually present in ref_outputs have a real cached_y_emb to
    # reuse -- restrict held_out to exactly those (this run's own warmup uses one
    # of them repeated, since warmup doesn't need to match any particular image).
    held_out = [p for p in image_paths[CALIB_N:CALIB_N + N_BENCH_IMAGES] if p.name in ref_outputs]
    print(f"{len(held_out)} held-out images with a saved GPU reference "
          f"(of {N_BENCH_IMAGES} candidates, {N_BENCH_IMAGES - len(held_out)} were the GPU run's own warmup)")
    # Reuse the EXACT SAME cached_y_emb the GPU reference computed, rather than
    # recomputing it fresh on CPU in this different torch build (2.12.1 vs the
    # reference's 2.7.0) -- confirmed live this matters: a first pass that
    # recomputed cached_y_emb independently per backend got mean_max_diff up to
    # 0.36 (vs. output magnitude ~0.09, i.e. dominated by noise), because a
    # cross-env/cross-device DINOv3-feature discrepancy was silently getting
    # amplified through 24 transformer layers and conflated with the actual
    # thing under test (conversion/quantization error). Holding this one real
    # input fixed isolates that.
    ref_cached_y_emb = {name: gpu_data[f"yemb_{name}"] for name in gpu_data["image_names"]}
    results = {"pytorch_cuda": gpu_data["times_ms"].tolist()}
    print(f"Loaded {len(ref_outputs)} PyTorch-GPU reference outputs from {GPU_RESULTS}")

    # ---- PyTorch CPU ----
    model_cpu = load_model("cpu")
    times_cpu = []
    cpu_diffs = []
    for i, p in enumerate(held_out):
        labels = load_image_512(p)
        cached_y_emb = torch.from_numpy(ref_cached_y_emb[p.name])
        z = torch.from_numpy(np.random.RandomState(0).randn(1, 3, 512, 512).astype(np.float32))
        t = torch.tensor([0.3])
        if i < N_WARMUP:
            with torch.no_grad():
                model_cpu.net(z, t, labels, cached_y_emb=cached_y_emb)
            continue
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model_cpu.net(z, t, labels, cached_y_emb=cached_y_emb)
        times_cpu.append((time.perf_counter() - t0) * 1000)
        if p.name in ref_outputs:
            cpu_diffs.append(np.abs(out.numpy() - ref_outputs[p.name]).max())
    results["pytorch_cpu"] = times_cpu
    print(f"pytorch_cpu done, {len(times_cpu)} timed calls")

    # ---- LiteRT fp32 and int8, both CPU via XNNPACK ----
    from ai_edge_litert.interpreter import Interpreter

    accuracy_summary = {"pytorch_cpu": (float(np.mean(cpu_diffs)), float(np.max(cpu_diffs)), len(cpu_diffs))}
    for tag, tflite_path in [("litert_fp32", FLOAT_TFLITE), ("litert_int8", INT8_TFLITE),
                             ("litert_dynamic", DYNAMIC_TFLITE), ("litert_weightonly", WEIGHTONLY_TFLITE),
                             ("litert_dynamic_int4", DYNAMIC_INT4_TFLITE),
                             ("litert_weightonly_int4", WEIGHTONLY_INT4_TFLITE)]:
        interp = Interpreter(model_path=tflite_path, num_threads=4)
        interp.allocate_tensors()
        in_d = {d["name"]: d for d in interp.get_input_details()}
        out_d = interp.get_output_details()[0]
        times = []
        diffs = []
        for i, p in enumerate(held_out):
            labels_np = load_image_512(p).numpy().astype(np.float32)
            cached_y_emb_np = ref_cached_y_emb[p.name].astype(np.float32)
            z_np = np.random.RandomState(0).randn(1, 3, 512, 512).astype(np.float32)
            t_np = np.array([0.3], dtype=np.float32)

            real_inputs = {"args_0": z_np, "args_1": t_np, "args_2": labels_np,
                           "cached_y_emb": cached_y_emb_np}
            # The int8-quantized model's own re-export renames signature inputs with a
            # "serving_default_" prefix (confirmed live -- the fp32 export used bare
            # names, KeyError'd here on first run); match by suffix, not exact name, so
            # this works for either convention.
            for name, d in in_d.items():
                matched_key = next(k for k in real_inputs if name == k or name.endswith(f"_{k}") or name.endswith(k))
                value = real_inputs[matched_key]
                # The int8 model's own input tensors are genuinely int8 I/O (confirmed
                # live -- "Got value of type FLOAT32 but expected type INT8" on the
                # first attempt), not float-in/float-out with int8 internal compute
                # only. `args_2`/labels stayed float32 in the quantized model (its own
                # real scale/zero_point are 0/0) -- torch.export prunes the whole
                # cached_y_emb-is-None branch that would have consumed `labels` as dead
                # code when cached_y_emb is provided, so it's an unused phantom input
                # with nothing to calibrate; fed as-is either way.
                if d["dtype"] == np.int8:
                    scale, zero_point = d["quantization"]
                    value = np.round(value / scale + zero_point).clip(-128, 127).astype(np.int8)
                interp.set_tensor(d["index"], value)

            if i < N_WARMUP:
                interp.invoke()
                continue
            t0 = time.perf_counter()
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000)
            out = interp.get_tensor(out_d["index"])
            if out_d["dtype"] == np.int8:
                out_scale, out_zero_point = out_d["quantization"]
                out = (out.astype(np.float32) - out_zero_point) * out_scale
            if p.name in ref_outputs:
                diffs.append(np.abs(out - ref_outputs[p.name]).max())
        results[tag] = times
        accuracy_summary[tag] = (float(np.mean(diffs)), float(np.max(diffs)), len(diffs))
        print(f"{tag}: {len(times)} timed calls")

    print("\n=== Accuracy: max abs diff vs PyTorch-GPU reference, held-out pinecone images ===")
    for tag, (mean_d, max_d, n) in accuracy_summary.items():
        print(f"{tag:16s}: mean_max_diff={mean_d:.4e}  worst_case={max_d:.4e}  n={n}")

    print("\n=== Real per-step denoiser forward latency (ms), held-out pinecone images ===")
    for tag, times in results.items():
        times = np.array(times)
        print(f"{tag:16s}: mean={times.mean():7.2f} ms  median={np.median(times):7.2f} ms  "
              f"std={times.std():6.2f} ms  n={len(times)}")

    ref = np.array(results["pytorch_cuda"])
    for tag in ["pytorch_cpu", "litert_fp32", "litert_int8"]:
        t = np.array(results[tag])
        print(f"speedup {tag} vs pytorch_cuda: {ref.mean() / t.mean():.3f}x "
              f"({'faster' if t.mean() < ref.mean() else 'slower'})")


if __name__ == "__main__":
    main()
