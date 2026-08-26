<p align="center">
  <h1 align="center">PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation</h1>
  <p align="center">
    <a href="https://haofeixu.github.io/">Haofei Xu</a><sup>1,2,3</sup> &middot;
    <a href="https://rundiwu.github.io/">Rundi Wu</a><sup>1</sup> &middot;
    <a href="https://henzler.github.io/">Philipp Henzler</a><sup>1</sup> &middot;
    <a href="https://scholar.google.com/citations?user=XwzlnZoAAAAJ">Nikolai Kalischek</a><sup>1</sup> &middot;
    <a href="https://scholar.google.com/citations?user=vMcsUNAAAAAJ">Michael Oechsle</a><sup>1</sup>
    <br>
    <a href="https://scholar.google.com/citations?user=bERItx8AAAAJ">Fabian Manhardt</a><sup>1</sup> &middot;
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a><sup>2,4</sup> &middot;
    <a href="http://www.cvlibs.net/">Andreas Geiger</a><sup>3,5</sup> &middot;
    <a href="https://federicotombari.github.io/">Federico Tombari</a><sup>1,6</sup> &middot;
    <a href="https://m-niemeyer.github.io/">Michael Niemeyer</a><sup>1</sup>
  </p>
  <p align="center">
    <sup>1</sup>Google &nbsp; <sup>2</sup>ETH Zurich &nbsp; <sup>3</sup>University of Tübingen, Tübingen AI Center &nbsp; <sup>4</sup>Microsoft &nbsp; <sup>5</sup>KE:SAI &nbsp; <sup>6</sup>TUM
  </p>
  <h3 align="center">ICML 2026</h3>
  <h3 align="center">
    <a href="https://arxiv.org/abs/2607.02515">Paper</a> | <a href="https://haofeixu.github.io/pointdit/">Project Page</a> | <a href="MODELS.md">Models</a>
  </h3>
</p>

<p align="center">
  <a href="https://haofeixu.github.io/assets/pointdit.mp4">
    <img src="assets/teaser.gif" alt="PointDiT teaser" width="100%">
  </a>
</p>

PointDiT is a minimalist pixel-space Diffusion Transformer for monocular geometry estimation. It is a plain ViT that denoises raw 3D point map patches directly, conditioned on image tokens from a frozen pre-trained DINOv3: no point map tokenizer, no latent diffusion, no hybrid architecture, and no intricate loss formulation.

## Installation

This codebase is developed with Python 3.12, PyTorch 2.7.0, and CUDA 12.8.

We recommend setting up a virtual environment (e.g., [conda](https://docs.anaconda.com/miniconda/) or [venv](https://docs.python.org/3/library/venv.html)) before installation:

```bash
# conda
conda create -y -n pointdit python=3.12
conda activate pointdit

# or venv
# python -m venv /path/to/venv/pointdit
# source /path/to/venv/pointdit/bin/activate

# torch 2.7.0, cuda 12.8
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

# DINOv3 encoder code (not vendored; built with torch.hub from a local checkout)
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
```

All commands are run from the repository root, and the launch scripts in [scripts/](scripts).

The DINOv3 weights are gated and cannot be redistributed, so they are not part of the released
checkpoints. Request access on the
[official DINOv3 repository](https://github.com/facebookresearch/dinov3) and place the weights
under `pretrained/dinov3/` with their exact upstream filenames, or point `DINOV3_WEIGHTS_DIR` at
the directory holding them. They are needed for evaluation and the demo as well as for training;
see [MODELS.md](MODELS.md) for the file list.

## Model Zoo

Pre-trained models are available in the [Model Zoo](MODELS.md), at `256×256` and `512×512` for
each of PointDiT-B / L / H. All weights are hosted on
[Hugging Face](https://huggingface.co/haofeixu/pointdit).

Download the weights and place (or symlink) them in the `pretrained` directory:

```bash
ln -s YOUR_MODEL_PATH pretrained
```

The checkpoints ship without the frozen DINOv3 encoder, which is downloaded separately (see
[Installation](#installation)).

## Demo

Check [scripts/demo_l_512.sh](scripts/demo_l_512.sh) for running our pre-trained models on your own images, with no ground truth and no camera intrinsics required. Set `IMAGE_DIR` at the top of the script and run it:

```bash
bash scripts/demo_l_512.sh
```

Any `.jpg`/`.png` files work, and nested folders are fine, since the loader globs recursively. Results are written to `generation/`.

Also see [`notebooks/`](notebooks) for parameterized, re-runnable Jupyter versions of this demo
(one driving `main.py` via subprocess, one with the full setup/generation logic inlined into the
notebook itself).

## LiteRT / on-device inference

*Community addition — not part of the original release.* [`tools/litert/`](tools/litert)
converts the per-step denoiser network to [LiteRT](https://ai.google.dev/edge/litert) (`.tflite`)
and measures 8 quantization strategies against the real PyTorch GPU/CPU baselines, on 14 held-out
images from the `pinecone` scene of the [mip-NeRF 360](https://huggingface.co/datasets/1kaiser/NERF_360)
dataset. Only the denoiser transformer is converted — the frozen DINOv3 encoder's output is a
separate graph *input*, never baked into the exported model, so it stays gated exactly as
elsewhere in this repo; nothing gated is redistributed by any `.tflite` file here.

| Backend | Latency (mean) | vs PyTorch-GPU | Accuracy (max abs diff, single step) |
|---|---:|---:|---:|
| PyTorch GPU | 99.75 ms | reference | reference |
| PyTorch CPU | 460 ms | 4.6x slower | 8.3e-6 (float noise) |
| LiteRT fp32 | 2201 ms | 22x slower | 8.3e-6 (clean conversion) |
| LiteRT int8 static (act+weight) | 7763 ms | 78x slower | 0.345 (broken) |
| LiteRT int8 dynamic | 1213 ms | 12x slower | 0.020 |
| LiteRT int8 weight-only | 2200 ms | 22x slower | **0.007 (best accuracy)** |
| LiteRT int4 dynamic | 1214 ms | 12x slower | 0.103 |
| LiteRT int4 weight-only | 17753 ms | 178x slower | 0.104 |

None of the 8 configurations beat plain PyTorch-GPU (or even PyTorch-CPU) on this architecture on
the machine this was measured on — XNNPACK's kernels are CNN/mobile-tuned, and this is a
large, matmul-heavy ViT. Weight-only int8 is the recommended variant when a `.tflite` deployment
is the actual goal (e.g. mobile/browser, not a speed win on desktop CPU): best accuracy of the six
usable configs, and the only one that also ran correctly end-to-end in a browser via
[LiteRT.js](https://ai.google.dev/edge/litert/web) (PointDiT-B/16; PointDiT-L/16 hits a real
WASM32 model-size ceiling in that runtime). **Full-generation accuracy is worse than the
single-step numbers above** — errors compound across the euler sampling loop (weight-only int8
measured 0.0999 max abs diff over a real 3-step generation, vs. 0.007 for one isolated call); see
[`tools/litert/run_litert_inference.py`](tools/litert/run_litert_inference.py) to reproduce this
directly, and [`tools/litert/README.md`](tools/litert/README.md) for the full writeup, environment
setup, and the browser demo.

Pre-converted `.tflite` files for all 8 configurations (both PointDiT-L/16 and B/16) are attached
to this fork's [Releases](../../releases) — download rather than reconverting from scratch.

[`tools/litert/run_multiview_merge.py`](tools/litert/run_multiview_merge.py) additionally tests
multi-image merging: matching images pairwise with [LoMa](https://github.com/davnords/LoMa) and
fitting a similarity transform between each pair's matched 3D points to recover a shared frame
across PointDiT's independent, zero-centered per-image predictions. Real result: this works well
for adjacent-frame sequences, but is measurably unreliable at wider baselines (see the notebook
and `tools/litert/README.md` for the full honest writeup, including a case where it makes the
merge worse than doing nothing).

## Dataset Preparation

See [DATASETS.md](DATASETS.md) for detailed instructions on preparing the 12 training datasets and the 7 zero-shot evaluation datasets. We have provided the download link for the 7 evaluation datasets in [DATASETS.md](DATASETS.md).

Symlink the prepared datasets to the `datasets` directory:

```bash
ln -s YOUR_DATASET_PATH datasets
```

Point clouds are never stored on disk: the training loader reads `(RGB, depth, intrinsics)` and back-projects on the fly.


## Evaluation

Evaluation scripts are also provided in [scripts/](scripts) for reproducing the results in our paper:

```bash
bash scripts/eval_l_512.sh
```

Swap `l` for `b` or `h`, and `512` for `256`. Each script's header comment lists the numbers it should reproduce.

## Training

PointDiT is trained in two stages: (1) `256×256` pre-training on SceneNet-RGBD and (2) `512×512` fine-tuning on the 11-dataset mixture.

```bash
bash scripts/train_stage1_256_b.sh
bash scripts/train_stage2_512_b.sh
```

The training scripts in [scripts/](scripts) contain the exact commands and hyperparameters used for the experiments in our paper. Please refer to them for detailed configurations. Before training, you need to download the DINOv3 weights (see [Installation](#installation)) and prepare the datasets per [DATASETS.md](DATASETS.md).


## License

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

This codebase is released under the Apache License 2.0 (see the `LICENSE` file at the root of
the repository). It also contains code derived from other open-source projects, which
remains under its original licenses; see [THIRD_PARTY_NOTICES](third_party/THIRD_PARTY_NOTICES) for the
full list and notices.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{xu2026pointdit,
  title={PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation},
  author={Xu, Haofei and Wu, Rundi and Henzler, Philipp and Kalischek, Nikolai and Oechsle, Michael and Manhardt, Fabian and Pollefeys, Marc and Geiger, Andreas and Tombari, Federico and Niemeyer, Michael},
  booktitle={ICML},
  year={2026}
}
```

## Acknowledgements

Our codebase builds upon several excellent open-source projects: [JiT](https://github.com/LTH14/JiT), [DINOv3](https://github.com/facebookresearch/dinov3), [MoGe](https://github.com/microsoft/MoGe), [Depth Pro](https://github.com/apple/ml-depth-pro) and [utils3d](https://github.com/EasternJournalist/utils3d). We thank all the authors for their great work.
