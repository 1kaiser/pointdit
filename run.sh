#!/bin/bash

#
# Minimal end-to-end smoke test for PointDiT: CPU-only, no GPU, no dataset
# download and no pre-trained checkpoint. It trains a randomly initialised model
# for two steps on synthetic data, then samples from the resulting checkpoint.
#
# Run it from the repository root:
#
#     bash run.sh
#
# The virtualenv it builds is deleted again on exit, so a full run leaves nothing
# behind. Set POINTDIT_SKIP_VENV=1 to reuse the active interpreter instead of
# building one (much faster when iterating locally, since the wheels below are
# reinstalled from scratch every time otherwise).
#
set -e
set -x

cd "$(dirname "$0")"

# Everything this script creates is torn down here, on failure as well as on
# success. Both paths are guarded so the trap can be installed up front, before
# either exists, and a single handler keeps one trap from replacing the other.
SMOKE_DIR=""
CLEAN_VENV=""
cleanup() {
  if [ -n "${SMOKE_DIR}" ]; then rm -rf "${SMOKE_DIR}"; fi
  if [ -n "${CLEAN_VENV}" ]; then rm -rf "${CLEAN_VENV}"; fi
}
trap cleanup EXIT

if [ -z "${POINTDIT_SKIP_VENV}" ]; then
  # Only remove a virtualenv we built ourselves: a pointdit_env that was already
  # there belongs to the user, and is theirs to keep.
  if [ ! -d pointdit_env ]; then
    CLEAN_VENV="${PWD}/pointdit_env"
  fi
  python3 -m venv pointdit_env
  source ./pointdit_env/bin/activate
  # CPU wheels keep this test small; see README.md for CUDA builds.
  pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  pip install -r requirements.txt
fi

# The DINOv3 encoder code is not vendored: torch.hub builds it from a local
# checkout. Only the architecture is needed here, not the gated weights.
if [ ! -d third_party/dinov3 ]; then
  git clone --depth 1 https://github.com/facebookresearch/dinov3.git third_party/dinov3
fi

# Keep the test single-machine and CPU-only regardless of the host.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

SMOKE_DIR=$(mktemp -d)

# Self-tests of the evaluation metrics (random tensors, CPU, no data needed).
python -m metrics.compute_metrics
python -m metrics.masked_resize

# A few synthetic frames in the layout the Hypersim loader expects.
python tools/make_smoke_data.py --out "${SMOKE_DIR}"

# Stage 1: two training steps from scratch, at a tiny resolution.
# --warmup_epochs 0 is required, else the LR schedule yields 0 for step 0-1.
python main.py \
  --model PointDiT-B/16 --feature_embedding_type dinov3_vits16 --proj_dropout 0.0 \
  --device cpu --img_size 64 --batch_size 2 --num_workers 0 --no_pin_mem \
  --dataset_name hypersim --data_path "${SMOKE_DIR}/data" \
  --depth_intrinsics_resize_height 64 \
  --epochs 1 --warmup_epochs 0 --lr 1e-4 --debug_train_steps 2 \
  --output_dir "${SMOKE_DIR}/out"

# Stage 2: sample from the checkpoint Stage 1 just wrote, on the wild-image path
# (no ground truth and no intrinsics required).
python main.py \
  --model PointDiT-B/16 --feature_embedding_type dinov3_vits16 --proj_dropout 0.0 \
  --device cpu --img_size 64 --num_sampling_steps 2 --gen_bsz 1 \
  --num_workers 0 --no_pin_mem \
  --evaluate_gen --eval_wild_images \
  --eval_wild_images_dir "${SMOKE_DIR}/wild" \
  --pretrained "${SMOKE_DIR}/out/checkpoint-last.pth" \
  --gen_output_root "${SMOKE_DIR}/generation" \
  --output_dir "${SMOKE_DIR}/out"

# The run is only a success if it actually wrote predictions.
OUT_DIR="${SMOKE_DIR}/generation"
test -n "$(find "${OUT_DIR}" -name '*_depth.png' -print -quit)"
test -n "$(find "${OUT_DIR}" -name '*_pointcloud.ply' -print -quit)"
# Wild-image outputs are named after the input, with no index prefix, no doubled extension,
# and no sky-mask panels.
test -n "$(find "${OUT_DIR}" -name 'wild_01_depth.png' -print -quit)"
test -n "$(find "${OUT_DIR}" -name 'wild_01_pointcloud.ply' -print -quit)"
test -z "$(find "${OUT_DIR}" -name '*skymask*' -print -quit)"

# We won't land here if there are errors due to set -e.
echo "*** Success!"
