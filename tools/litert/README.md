# LiteRT / on-device inference (community addition)

Converts PointDiT's per-step denoiser network to [LiteRT](https://ai.google.dev/edge/litert)
(`.tflite`), benchmarks 8 quantization strategies against the real PyTorch GPU/CPU baselines, and
runs the converted model both from Python and in a browser via [LiteRT.js](https://ai.google.dev/edge/litert/web).

**What's converted, and why it's redistributable.** Only the denoiser transformer (the
Apache-2.0-licensed, HF-hosted part of this repo) is converted. The frozen DINOv3 image encoder is
never traced into the exported graph — `cached_y_emb` (its output) is passed as a separate,
explicit input tensor, confirmed by every script here treating it as a named graph input, not a
baked-in constant. So none of the `.tflite` files in this directory's
[Release assets](../../../../releases) contain any DINOv3 weights; DINOv3 remains gated exactly as
in the rest of this repo (see the main [README](../../README.md#installation)) and you still need
your own copy of it locally to run anything in this directory.

## Layout

| Path | What it is |
|---|---|
| `convert_denoiser_litert.py` / `_b16.py` | fp32 PyTorch → `.tflite` conversion + verification, for PointDiT-L/16 and B/16 |
| `quantize_denoiser_*.py` | the 6 quantized variants (int8 static/dynamic/weight-only, int4 dynamic/weight-only, plus a B/16 weight-only export) |
| `benchmark_pytorch_gpu.py` / `benchmark_litert_backends.py` | the real measured comparison (split across two conda envs — see script docstrings for why) |
| `export_web_demo_inputs.py` / `_b16.py` | dumps real per-image tensors for the browser demo |
| `run_litert_inference.py` (+ `.ipynb`) | **run this to actually use a converted model** — full image → depth/point-cloud generation with the denoiser routed through a `.tflite` interpreter, verified against the real PyTorch output on the same image |
| `run_multiview_merge.py` (+ `.ipynb`) | multi-image inference + cross-image matching ([LoMa](https://github.com/davnords/LoMa)) + similarity-transform depth-map merging — see below |
| `download_pinecone_bench.py` | fetches the real benchmark/calibration image set used above |
| `web_demo/` | in-browser LiteRT.js demo (Playwright-driven headless-Chrome runner + a plain-HTML page) |
| `models/` | **not in git** — converted `.tflite` files + the GPU-reference `.npz` go here locally; download pre-converted copies from the [Release](../../../../releases) or regenerate with the scripts above |

## Quickstart: run inference with a pre-converted model

```bash
# from the repo root, in an env with ai-edge-litert installed (see below)
mkdir -p tools/litert/models
# download the 8 .tflite files from this fork's Release page into tools/litert/models/
python tools/litert/download_pinecone_bench.py     # or point image_path at any image you have

jupytext --to notebook tools/litert/run_litert_inference.py -o /tmp/run.ipynb --set-kernel pointdit_litert
papermill /tmp/run.ipynb /tmp/run_out.ipynb --kernel pointdit_litert \
  -p image_path assets/pinecone_bench/IMG_7261.png \
  -p tflite_path tools/litert/models/pointdit_l16_denoiser_step_weightonly.tflite
```

Outputs land in `generation/litert_inference/`: a side-by-side depth PNG (input | PyTorch
reference | LiteRT) and both backends' point clouds as `.ply`.

## Environment

The conversion/quantization/LiteRT-inference tooling needs `ai-edge-litert`, `litert-torch`, and
`torchao`, which need `torch>=2.11.0` — incompatible with this repo's own `torch==2.7.0` pin.
Build a **separate** env for everything in this directory:

```bash
mamba create -y -p /path/to/envs/pointdit_litert python=3.11
conda activate /path/to/envs/pointdit_litert
uv pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install ai-edge-litert ai-edge-quantizer litert-torch torchao "typing-extensions>=4.16.0" \
  accelerate einops trimesh Pillow numpy huggingface_hub ipykernel
python -m ipykernel install --user --name pointdit_litert --display-name "PointDiT LiteRT"
```

`benchmark_pytorch_gpu.py` is the one script that needs the *original* `pointdit` env instead — its
GPU reference numbers require this repo's own `torch==2.7.0+cu128` build (a newer torch may lack
kernels for your specific GPU architecture). It writes `models/bench_pytorch_gpu_results.npz`,
which `benchmark_litert_backends.py`/`export_web_demo_inputs*.py` (run in `pointdit_litert`) then
read — that's why the benchmark is split into two scripts instead of one.

## The real, measured results

See the main [README](../../README.md#litert--on-device-inference) for the full 8-way
quantization comparison table and conclusions (short version: weight-only int8 is the best
accuracy/robustness tradeoff; none of the 8 configurations beat plain PyTorch on this project's
own architecture; single-step accuracy understates full-generation accuracy by roughly 10x —
measure at the granularity you'll actually deploy at).

## Multi-image merging with cross-image matching (LoMa)

PointDiT predicts each image's point map independently and zero-centered — there's no shared
frame across images by construction. `run_multiview_merge.py` tests whether adding
[LoMa](https://github.com/davnords/LoMa) (ECCV 2026, ungated Apache-2.0/MIT local feature matcher)
+ a robust similarity-transform fit (Umeyama + RANSAC) on the matched pixels' predicted 3D points
recovers a real shared frame across a sequence of images.

```bash
uv pip install --python "$(which python)" -e third_party/LoMa   # after cloning it there
jupytext --to notebook tools/litert/run_multiview_merge.py -o /tmp/run.ipynb --set-kernel pointdit_litert
CUDA_VISIBLE_DEVICES="" papermill /tmp/run.ipynb /tmp/run_out.ipynb --kernel pointdit_litert \
  -p stride 1 -p num_images 6   # stride=1: adjacent frames; try a larger stride for a wider baseline
```

**Real, measured result — it depends on baseline, and don't trust the residual metric alone:**
adjacent-frame sequences (`stride=1`) align well (median matched-point residual drops 5-12x,
plausible 4-8° per-hop rotations, both merges visually coherent). Widely-spaced sequences
(`stride=16` across the same 99-image set) report an equally good-looking residual but recover
**implausible 38-143° per-hop rotations** — a low residual on a small, possibly-degenerate RANSAC
inlier set does not mean the transform is correct, and the resulting chain-composed merge is
visibly *worse* than not aligning at all. See the notebook's own section 10 for the full
before/after table and root-cause discussion (this is a real limitation of independent pairwise
fits + naive chain composition, not a bug — a rotation-plausibility gate or genuine global
refinement across all pairs would be the real fix, not attempted here).

## Browser demo

```bash
cd tools/litert/web_demo
npm install
# symlink or copy a weight-only .tflite export to ./model.tflite (L/16) and/or ./model_b16.tflite (B/16)
python -m http.server 8974 &
node run_demo.js       # PointDiT-L/16: hits a real WASM32 size ceiling in this runtime build
node run_demo_b16.js   # PointDiT-B/16: runs correctly (see README/skill for the full writeup)
```

### `hosted/index.html` -- no local setup needed, with a fast Web-Worker path

The two demos above need a local `npm install` and a locally-placed `.tflite` file.
[`hosted/index.html`](web_demo/hosted/index.html) doesn't: `@litertjs/core` loads from jsdelivr's
CDN and the model is fetched at runtime from
[`huggingface.co/1kaiser/pointdit-litert`](https://huggingface.co/1kaiser/pointdit-litert) instead
of a local file -- the same real pattern this repo's own prior project
([`1kaiser/astro`'s `moge-jax-lite/webgpu_demo`](https://github.com/1kaiser/astro/tree/main/moge-jax-lite/webgpu_demo))
uses, and for the same concrete reason: GitHub Release assets send no `Access-Control-Allow-Origin`
header at all (`curl -I` on this repo's own `litert-v1` release shows none), so a page fetching one
from anywhere other than github.com is blocked by CORS -- Hugging Face's CDN sends
`access-control-allow-origin: *`, confirmed the same way on the model uploaded there for this demo.

It also detects and uses LiteRT.js's **threaded WASM runtime** (real Web Workers, not just async
`fetch()`) whenever it's actually available, falling back to the portable single-threaded CDN
runtime otherwise -- again mirroring the same prior project:

```bash
cd tools/litert/web_demo/hosted
python3 serve_threaded.py 8974 &   # sends the two headers the threaded path needs -- see below
node run_demo.js "http://127.0.0.1:8974/index.html"
# or: python3 -m http.server 8974 &   -- also works, just uses the slower portable path
```
The page runs the real 3-step euler generation loop (not just one denoiser call) and renders the
actual reconstructed point cloud via `THREE.js` + `GLTFExporter` + `<model-viewer>` -- the same
real visualization approach the reference project uses, not just a numeric accuracy check.

**Real, measured result**: the threaded runtime is **10.4x faster** than the portable one for the
full 3-step generation (16172ms -> 1560ms), building the identical 65,536-point cloud either way
(numeric accuracy against the PyTorch reference is already established by
`run_litert_inference.py`'s own checks, using the same math). `tools/litert/verify_hosted_web_demo.py`
tests both configurations for real, not just the default -- see there and the `research-repo-bringup`
skill for the full writeup of why the threaded path needs two specific things (COOP/COEP response
headers, and same-origin WASM files since the threaded build spawns real cross-origin-rejecting
`Worker()`s) and why `serve_threaded.py`
provides them.
