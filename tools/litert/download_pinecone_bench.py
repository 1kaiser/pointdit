"""Fetches the real 'pinecone' scene (mip-NeRF 360 dataset) used as the benchmark/calibration
set throughout tools/litert/, from the public HF dataset `1kaiser/NERF_360`, and lays out
`assets/pinecone_bench/*.png` the way every script in this directory expects
(`Path("assets/pinecone_bench").glob("*.png")`).

Real, verified structure (checked directly against the dataset, not assumed): the dataset repo
holds a single top-level `nerf_real_360.zip`, which itself contains one nested zip per scene
(`pinecone/pinecone.zip`), which in turn holds `images/IMG_*.JPG` -- the original mip-NeRF 360
JPEGs. This script downloads once, extracts the pinecone scene only, and converts each JPEG to a
PNG (no resize -- the conversion/quantization/benchmark scripts each resize to their own working
resolution independently) under `assets/pinecone_bench/`.

Run from the repo root: `python tools/litert/download_pinecone_bench.py`
"""
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "1kaiser/NERF_360"
ZIP_FILENAME = "nerf_real_360.zip"
SCENE = "pinecone"
OUT_DIR = Path("assets/pinecone_bench")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ZIP_FILENAME} from {REPO_ID} (dataset repo)...")
    top_zip_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=ZIP_FILENAME)

    work_dir = Path("tools/litert/_pinecone_dl_tmp")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    print(f"Extracting {SCENE}/{SCENE}.zip from {top_zip_path}...")
    with zipfile.ZipFile(top_zip_path) as z:
        scene_zip_name = f"{SCENE}/{SCENE}.zip"
        z.extract(scene_zip_name, work_dir)

    scene_dir = work_dir / SCENE
    with zipfile.ZipFile(work_dir / scene_zip_name) as z:
        z.extractall(scene_dir)

    jpgs = sorted((scene_dir / "images").glob("*.JPG"))
    assert jpgs, f"no images/*.JPG found inside {scene_zip_name}"
    print(f"Converting {len(jpgs)} JPEGs -> {OUT_DIR}/*.png ...")
    for jpg in jpgs:
        Image.open(jpg).convert("RGB").save(OUT_DIR / f"{jpg.stem}.png")

    shutil.rmtree(work_dir)
    n_png = len(list(OUT_DIR.glob("*.png")))
    print(f"Done: {n_png} PNGs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
