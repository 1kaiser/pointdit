# Dataset Preparation

This document describes the exact on-disk layout every dataset loader in this
codebase expects, so you can prepare the data for training and evaluation.

Training is **two-stage**:

- **Stage 1, pre-training** at `256×256` on a single dataset (**SceneNet-RGBD**).
  Driver scripts: `scripts/train_stage1_256_{b,l,h}.sh`.
- **Stage 2, fine-tuning** at `512×512` on an **11-dataset mixture**.
  Driver scripts: `scripts/train_stage2_512_{b,l,h}.sh`,
  which use the mixture config `dataloader/configs/res512mix.yaml`.

Evaluation is **zero-shot** on **7 real-captured datasets** (disjoint from training).

All paths below are relative to the repository root. In this repo `datasets/` is a
symlink to bulk storage; when releasing, replace it with a real directory (or your
own symlink) and place the data underneath as shown.

---

## Contents

- [Common conventions](#common-conventions)
- [Training datasets (12)](#training-datasets-12)
  - [Stage 1: SceneNet-RGBD](#1-scenenet-rgbd-stage-1)
  - [Stage 2 mixture (11 datasets)](#stage-2-mixture-11-datasets)
- [Evaluation datasets (7)](#evaluation-datasets-7)
- [Dataset mixture config format](#dataset-mixture-config-format)

---

## Common conventions

These hold across most datasets; per-dataset exceptions are called out below.

**Point clouds are computed on the fly.** No dataset stores point clouds. The
loader back-projects a depth map with camera intrinsics `K` into a camera-frame
point map at load time. So each sample needs three things: an **RGB image**, a
**depth map**, and **intrinsics**.

**Depth semantics, two flavors (do not mix them up):**
- **z-depth** (planar depth along the optical axis). Back-projected as
  `X=(u-cx)/fx·Z, Y=(v-cy)/fy·Z, Z=depth`. This is the common case.
- **ray-depth** (Euclidean distance along the pixel ray). Only **SceneNet-RGBD**
  uses this; it is back-projected with per-pixel unit rays.

**Depth storage, two families:**
- **Preprocessed `.npy`** float32, already in **meters**, z-depth. Used by most
  Stage-2 synthetic datasets (hypersim, urbansyn, synscapes, tartanair, eden,
  irs, dynamic_replica, mvssynth). You must convert the dataset's native depth
  (EXR / disparity / etc.) offline into per-frame `.npy` meters. Non-finite
  values are tolerated (zeroed at load).
- **Encoded `.png`** decoded at load time with a per-dataset scale/format:
  SceneNet (uint16 mm), VKITTI2 (uint16 cm), OmniWorld-Game (uint16 nonlinear
  disparity), TartanAirV2 (RGBA-packed float32), eval (MoGe log-16bit).

**Intrinsics, three sources:**
- **Per-frame `.npz`** with a 3×3 matrix under the key **`intrinsics`**
  (hypersim, urbansyn, synscapes, eden, irs, dynamic_replica, mvssynth).
- **Hardcoded constant** derived from a fixed camera model
  (SceneNet FOV; TartanAir `fx=fy=320,cx=320,cy=240` @640×480;
  TartanAirV2 `fx=fy=cx=cy=320` @640×640).
- **Per-sequence / per-split text or json** (VKITTI2 `intrinsic.txt`;
  OmniWorld-Game `camera/split_*.json`).

**No camera poses / extrinsics are ever read.** Everything is done in the camera
frame. Files like VKITTI2 `extrinsic.txt` or TartanAir pose files may exist but
are ignored.

**No sky-mask files.** Sky/far regions are handled numerically from depth
(thresholding + optional sky-dome), configured per dataset in the mixture YAML
(`handle_sky`, `use_sky_dome`, `sky_far_plane_value`). No segmentation file is
loaded (a `segmentation.png` present in some eval datasets is ignored).

**Discovery is cached.** The loader pickles the discovered file list per
`(dataset, split, subsample)` config. If you change files on disk, clear the
cache to force re-discovery.

**Train/test splits.** No curated holdout lists ship with the code. Every
mixture entry sets `num_test_scenes: 0` in its config, and the training scripts
pass `--split all`, so **all data is used for training**. To hold scenes out,
give a loader a `test_scenes_file` (one scene or sequence name per line, path
resolved relative to `dataloader/`) and run with `--split train`. Splitting is by
scene/sequence, never by frame except where noted.

---

## Training datasets (12)

Reproducing the paper's Table 4. **Note: the paper's "TartanGround" is named
`tartanairv2` in the code and configs.** Weights are the Stage-2 sampling
probabilities (independent of corpus size).

| # | Dataset (code name) | Domain | #Samples | Weight | `data_path` (config) |
|---|---|---|---:|---:|---|
| 1 | SceneNet-RGBD (`scenenet`) | indoor | 5,359,500 | 1.00¹ | `datasets/train/scenenet-rgbd` |
| 2 | Hypersim (`hypersim`) | indoor | 70,647 | 0.12 | `datasets/train/hypersim` |
| 3 | Virtual KITTI 2 (`vkitti2`) | driving | 42,520 | 0.14 | `datasets/train/vkitti2` |
| 4 | UrbanSyn (`urbansyn`) | urban driving | 7,539 | 0.05 | `datasets/train/urbansyn` |
| 5 | Synscapes (`synscapes`) | urban driving | 25,000 | 0.09 | `datasets/train/synscapes` |
| 6 | TartanAir (`tartanair`) | diverse | 306,637 | 0.10 | `datasets/train/tartanair` |
| 7 | OmniWorld-Game (`omniworldgame`) | diverse (game) | 1,024,252 | 0.19 | `datasets/train/omniworldgame` |
| 8 | EDEN (`eden`) | garden/outdoor | 368,663 | 0.05 | `datasets/train/eden` |
| 9 | IRS (`irs`) | indoor | 39,342 | 0.02 | `datasets/train/irs` |
| 10 | Dynamic Replica (`dynamic_replica`) | indoor | 150,900 | 0.03 | `datasets/train/dynamic_replica` |
| 11 | MVS-Synth (`mvssynth`) | urban | 12,000 | 0.06 | `datasets/train/mvssynth` |
| 12 | TartanGround (`tartanairv2`) | diverse | 4,170,178 | 0.15 | `datasets/train/tartanairv2` |

¹ SceneNet is Stage-1 only (single-dataset pre-training, weight 1.00). Datasets
2 to 12 form the Stage-2 mixture (weights sum to 1.00).

All 12 sources are **synthetic** and provide dense depth with known intrinsics.

---

### 1. SceneNet-RGBD (Stage 1)

- **Root:** `datasets/train/scenenet-rgbd`
- **Depth:** 16-bit PNG in **millimeters**, **ray-depth** (÷1000 → meters).
- **Intrinsics:** hardcoded from fixed FOV (hfov=60°, vfov=45°); no calib file.
- **Split:** directory-based (`train/` vs `val/`); no split file.

```
scenenet-rgbd/
├── train/
│   └── {scene}/                 # e.g. 0, 1, ...
│       └── {trajectory}/        # e.g. 123
│           ├── photo/           # literal, required
│           │   └── {frame}.jpg  # stem = frame id (0, 25, 50, ...)
│           └── depth/           # literal, required
│               └── {frame}.png  # same stem; uint16 PNG, mm
└── val/
    └── {scene}/{trajectory}/{photo,depth}/...
```

- A trajectory is skipped unless **both** `photo/` and `depth/` exist and there
  is a matching `depth/{stem}.png` for each `photo/{stem}.jpg`.
- `split=all` scans both `train/` and `val/`. The Stage-1 script uses
  `--split all --dataset_name scenenet`.
- Native resolution 320×240. The SceneNet protobufs / `instances/` dirs are
  ignored, and only the `photo/` and `depth/` image folders matter.

---

### Stage 2 mixture (11 datasets)

Two structural sub-families recur; scan them once, then the per-dataset trees:

- **Flat "triplet" layout** (`rgb/`, `depth/`, `cam/` matched by filename stem):
  UrbanSyn, Synscapes, EDEN, IRS, MVS-Synth, and (per-camera) Dynamic Replica.
  RGB `.png`/`.jpg` + depth `.npy` (meters) + intrinsics `.npz` (`intrinsics`
  key). A frame is used **only if all three files exist**.
- **Sequence trees with fixed/text intrinsics:** TartanAir, TartanAirV2,
  VKITTI2, Hypersim, OmniWorld-Game.

---

#### 2. Hypersim

- **Root:** `datasets/train/hypersim`
- **Depth:** `.npy` float32 z-depth, **meters** (native ray-distance already
  converted during preprocessing).
- **Intrinsics:** per-frame `.npz`, key `intrinsics` (3×3). Variable per frame.
- **Split:** by top-level scene, if a `test_scenes_file` of scene names
  (e.g. `ai_001_004`) is supplied. None ships, so all scenes are used.

```
hypersim/
└── {scene}/                     # ai_XXX_YYY, e.g. ai_001_001
    └── {cam}/                   # cam_00, cam_01, ...
        ├── {frame:06d}_rgb.png
        ├── {frame:06d}_depth.npy   # z-depth, meters
        └── {frame:06d}_cam.npz     # key 'intrinsics' = 3x3 K
```

- Exactly two directory levels (scene → cam) before frame files. A frame is kept
  only if all three `_rgb.png` / `_depth.npy` / `_cam.npz` exist.
- Expects **preprocessed** data in the layout above, not raw Hypersim `.hdf5`. The
  conversion (tone-mapped RGB, OpenGL projection matrix → pinhole `K`, distance →
  planar z-depth) has to be reproduced from the official Hypersim release; the
  conversion scripts are not part of this release.

---

#### 3. Virtual KITTI 2 (`vkitti2`)

- **Root:** `datasets/train/vkitti2` (official tars already
  extracted and merged: rgb + depth + textgt).
- **Depth:** 16-bit PNG in **centimeters**, z-depth (÷100 → meters), read with
  `cv2.IMREAD_ANYDEPTH`. Sky encoded as 65535 cm (655.35 m), removed by
  `max_depth`.
- **Intrinsics:** per `(scene, variation)` `intrinsic.txt` (header +
  `frame cameraID fx fy cx cy` rows).
- **Split:** camera-sequence level, if a `test_scenes_file` of
  `{scene}/{variation}/{camera}` names is supplied. None ships, so all are used.

```
vkitti2/
└── Scene{NN}/                   # Scene01, Scene02, Scene06, Scene18, Scene20
    └── {variation}/             # clone, fog, rain, morning, sunset, overcast,
        │                        # 15-deg-left/right, 30-deg-left/right
        ├── intrinsic.txt        # per-frame per-camera K table
        ├── extrinsic.txt        # present but UNUSED
        └── frames/
            ├── rgb/Camera_{0,1}/rgb_{frame:05d}.jpg
            └── depth/Camera_{0,1}/depth_{frame:05d}.png   # uint16, cm
```

- Scene dirs must start with `Scene`. The depth path is derived from the RGB
  path by swapping `/rgb/`→`/depth/`, so the depth tree must **mirror** the rgb
  tree exactly (same `Camera_{id}` folders and frame ids).

---

#### 4. UrbanSyn

- **Root:** `datasets/train/urbansyn` (preprocessed to 512×1024).
- **Depth:** `.npy` float32 z-depth, **meters** (already scaled).
- **Intrinsics:** per-frame `.npz`, key `intrinsics`.
- **Split:** frame-level uniform sampling (`num_test_scenes` = #test frames).

```
urbansyn/
├── rgb/   rgb_{frame:04d}.png
├── depth/ rgb_{frame:04d}.npy      # z-depth, meters
└── cam/   rgb_{frame:04d}.npz      # key 'intrinsics'
```

- **Filename coupling:** depth/cam reuse the **full** rgb stem including the
  `rgb_` prefix (`rgb_0001.png` → `rgb_0001.npy` → `rgb_0001.npz`).
- The runtime loader reads only the precomputed `.npz`, and the
  the intrinsics are baked into the per-frame `.npz` by the preprocessing script.

---

#### 5. Synscapes

- **Root:** `datasets/train/synscapes`
- **Depth:** `.npy` float32 z-depth, **meters** (preprocessed from EXR).
- **Intrinsics:** per-frame `.npz`, key `intrinsics`.
- **Split:** frame-level (uniform sampling, or a text list of frame ids).

```
synscapes/
├── rgb/   {frame}.png             # bare stem, e.g. 1.png ... 25000.png
├── depth/ {frame}.npy             # z-depth, meters
└── cam/   {frame}.npz             # key 'intrinsics'
```

- Flat triplet layout; stems must match exactly across `rgb/`, `depth/`, `cam/`.
  Only `.png` RGB is discovered.

---

#### 6. TartanAir

- **Root:** `datasets/train/tartanair` (already unzipped).
- **Depth:** `.npy` float32 z-depth, **meters**.
- **Intrinsics:** **hardcoded** `fx=fy=320, cx=320, cy=240` for 640×480.
- **Split:** sequence level, if a `test_scenes_file` of
  `{env}/{difficulty}/{trajectory}` names is supplied. None ships, so all are used.

```
tartanair/
└── {scene}/                       # abandonedfactory, amusement, ...
    └── {difficulty}/              # Easy, Hard
        └── {sequence}/            # P000, P001, ...
            ├── image_left/{frame:06d}_left.png
            └── depth_left/{frame:06d}_left_depth.npy   # z-depth, meters
```

- Strictly 3 directory levels (scene → difficulty → sequence). Only the **left**
  camera is used; `image_right/`/`depth_right/` are ignored. Every RGB frame must
  have its `_left_depth.npy` twin.

---

#### 7. OmniWorld-Game (`omniworldgame`)

- **Root:** `datasets/train/omniworldgame`
- **Depth:** 16-bit PNG, **nonlinear disparity-like** encoding decoded to metric
  z-depth (near=1.0, far=1000.0; see `load_omniworldgame_depth`). Not a plain
  uint16/1000 map.
- **Intrinsics:** per-split json (`camera/split_{idx}.json`) with a per-frame
  `focals` list plus scalar `cx`, `cy` (fx=fy=focals[i]).
- **Split:** scene level (manual list, or uniform sampling).

```
omniworldgame/
├── videos/OmniWorld-Game/        # literal "OmniWorld-Game" (case + hyphen)
│   └── {scene}/color/{frame:06d}.png          # RGB
└── annotations/OmniWorld-Game/
    └── {scene}/
        ├── split_info.json       # REQUIRED: {'split_num': N, 'split': [[frame ids], ...]}
        ├── depth/{frame:06d}.png # uint16 nonlinear disparity
        └── camera/split_{idx}.json   # {'focals': [...], 'cx': ..., 'cy': ...}
```

- RGB lives under `videos/`, everything else under `annotations/`. The inner
  literal segment `OmniWorld-Game` is required. Frame filenames are the integer
  indices listed in `split_info.json`, formatted `%06d`, shared by color and depth.

---

#### 8. EDEN

- **Root:** `datasets/train/eden`
- **Depth:** `.npy` float32 z-depth, **meters**.
- **Intrinsics:** per-frame `.npz`, key `intrinsics`.
- **Split:** sequence level (a "sequence" = one `{scene_id}_{lighting}` dir).

```
eden/
└── {scene_id}_{lighting}/        # lighting ∈ {clear, cloudy, overcast, sunset, twilight}
    ├── rgb/   {frame}.png        # e.g. A_0001.png
    ├── depth/ {frame}.npy        # z-depth, meters
    └── cam/   {frame}.npz        # key 'intrinsics'
```

- Each lighting variant is its own sequence. A dir is recognized only if all of
  `rgb/`, `depth/`, `cam/` exist; stems must match across the three.

---

#### 9. IRS

- **Root:** `datasets/train/irs`
- **Depth:** `.npy` float32 z-depth, **meters** (preprocessed from disparity EXR;
  depth-edge pixels pre-filtered, hence `filterdepthedge`).
- **Intrinsics:** per-frame `.npz`, key `intrinsics` (`fx=fy=480`, principal
  point at image center).
- **Split:** sequence level; default config uses all sequences (`num_test_scenes: 0`).

```
irs/
└── {sequence}/                   # e.g. ArchVizInterior03Data, ConvenienceStore
    ├── rgb/   {frame:05d}.png     # 5-digit ids, e.g. 00000.png
    ├── depth/ {frame:05d}.npy     # z-depth, meters
    └── cam/   {frame:05d}.npz     # key 'intrinsics'
```

- Expects **preprocessed** IRS (not the raw disparity EXR release).

---

#### 10. Dynamic Replica (`dynamic_replica`)

- **Root:** `datasets/train/dynamic_replica` (CUT3R-style processed data).
- **Depth:** `.npy` float32 z-depth, **meters**.
- **Intrinsics:** per-frame `.npz`, key `intrinsics` (≈`fx=fy=700, cx=640,
  cy=360` @1280×720).
- **Split:** by top-level `train`/`valid`/`test` subdir.

```
dynamic_replica/
└── {split}/                      # train | valid | test
    └── {sequence}/
        └── left/                 # only 'left' is used ('right' ignored)
            ├── rgb/   {frame_id}.png     # frame_id = float timestamp, e.g. 0.0, 0.03333...
            ├── depth/ {frame_id}.npy     # z-depth, meters
            └── cam/   {frame_id}.npz     # key 'intrinsics'
```

- Frame ids are **float-timestamp strings** (sorted numerically). Only `left/` is
  consumed. A frame is kept only if depth `.npy` and cam `.npz` both exist.

---

#### 11. MVS-Synth (`mvssynth`)

- **Root:** `datasets/train/mvssynth` (the former `GTAV_720`
  contents sit directly under this root)
- **Depth:** `.npy` float32 z-depth in **centimeters** (÷100 → meters).
  *(Note: unlike the other `.npy` datasets, this one is in cm, not meters.)*
- **Intrinsics:** per-frame `.npz`, key `intrinsics`.
- **Split:** sequence level (manual list or uniform sampling).

```
mvssynth/
└── {sequence}/                   # e.g. 0000, 0001, ...
    ├── rgb/   {frame}.jpg         # RGB is .jpg here
    ├── depth/ {frame}.npy         # z-depth, CENTIMETERS
    └── cam/   {frame}.npz         # key 'intrinsics'
```

- RGB glob is literal `*.jpg`. Preprocessed from MVS-Synth EXR into per-frame
  cm-depth `.npy` + intrinsics `.npz`.

---

#### 12. TartanGround / TartanAirV2 (`tartanairv2`)

- **Root:** `datasets/train/tartanairv2` (already unzipped).
- **Depth:** **4-channel RGBA 8-bit PNG whose 4 bytes per pixel are a packed
  little-endian float32** (meters), z-depth. Decode with
  `cv2.imread(..., IMREAD_UNCHANGED).view("<f4")`. Standard uint16 readers will
  **not** work.
- **Intrinsics:** **hardcoded** `fx=fy=cx=cy=320` for 640×640.
- **Split:** `(trajectory, camera)` level; default config uses all data
  (`num_test_scenes: 0`).

```
tartanairv2/
└── {environment}/                # e.g. AbandonedFactory
    └── Data_{difficulty}/        # Data_easy, Data_hard
        └── {trajectory}/         # P000, P001, ...
            ├── image_{cam}/{frame:06d}_{cam}.png
            └── depth_{cam}/{frame:06d}_{cam}_depth.png   # RGBA-packed float32, meters
```

- 12 camera views `{cam}`: `lcam_{front,back,left,right,top,bottom}` and
  `rcam_{front,back,left,right,top,bottom}`.
- A `(trajectory, camera)` is enumerated only if both `image_{cam}/` and
  `depth_{cam}/` exist. Depth is matched by string construction, so every RGB
  frame must have its exact `{frame}_{cam}_depth.png` twin.

---

## Evaluation datasets (7)

Reproducing the paper's Table 5. All are **real-captured** and disjoint from
training. "Boundary" = scale-invariant boundary-F1 additionally reported.

| Dataset | Domain | #Samples | Boundary |
|---|---|---:|:---:|
| DIODE | indoor + outdoor | 771 | |
| KITTI | outdoor (driving) | 652 | |
| NYUv2 | indoor | 654 | |
| ETH3D | indoor + outdoor | 454 | |
| HAMMER | indoor (multi-modal) | 775 | ✓ |
| iBims-1 | indoor | 100 | ✓ |
| Booster | indoor (transparent/specular) | 38 | ✓ |
| **Total** | | **3,444** | |

**All 7 sets are available pre-processed**, one zip per dataset, 2.7 GB total:

> **[PointDiT evaluation sets (Google Drive)](https://drive.google.com/drive/folders/1RklLC1VFCBtFra_y29mp_m3eVvy7C9bg?usp=sharing)**

Nothing needs converting. Download the zips into `datasets/eval/` and unpack:

```bash
mkdir -p datasets/eval
# download Booster.zip DIODE.zip ETH3D.zip HAMMER.zip iBims-1.zip KITTI.zip
# NYUv2.zip into datasets/eval/ from the link above, then:
cd datasets/eval && unzip '*.zip'   # one zip per dataset
```

---

## Dataset mixture config format

The Stage-2 mixture is a YAML file (`dataloader/configs/res512mix.yaml`)
passed via `--dataset_config`. Each entry configures one dataset:

```yaml
finetune_highres_mode: true
finetune_target_height: 512
finetune_target_crop: 512
img_size: 512

datasets:
  - name: hypersim                 # selects the loader (must match a code name above)
    data_path: datasets/train/hypersim
    max_depth: 30.0                # depth validity threshold (meters)
    weight: 0.12                   # sampling probability in the mixture
    handle_sky: false              # sky handling (outdoor scenes set true)
    use_sky_dome: false
    sky_loss_weight: 0.0
    sky_far_plane_value: 3.0       # (sky datasets) far-plane depth for sky dome
    remove_outliers: true          # geometric outlier removal
    outlier_threshold: 3.0
    compute_scale_factor_only_valid: true
    compute_scale_factor_use_std: false
    clamp_max_depth: null
    num_test_scenes: 0             # 0 = use all data for training
  # ... one block per dataset
```

- **`name`** must match one of the code names in the tables above; it selects the
  `_find_and_split_data_<name>` discovery routine.
- **`data_path`** points at that dataset's root (the trees documented above).
- **`weight`** is the mixture sampling probability (weights across datasets should
  sum to ~1.0). **`max_depth`**, sky handling, and outlier removal are training
  filters, not on-disk requirements.

For single-dataset Stage-1 (SceneNet) the script uses the CLI flags
`--dataset_name scenenet --data_path ... --split all` instead of a mixture YAML
(see `dataloader/configs/scenenet.yaml` for the equivalent single-entry config).

Note that the per-dataset keys in a mixture YAML take precedence: the matching CLI
flags (`--max_depth`, `--handle_sky`, `--remove_outliers`,
`--compute_scale_factor_only_valid`, …) are ignored whenever `--dataset_config` is
passed. `sky_loss_weight` is the one key that is parsed but not forwarded to the
loader. Set the sky loss from the CLI with `--sky_loss_weight`.
