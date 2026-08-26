# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: PointDiT
#     language: python
#     name: pointdit
# ---

# %% [markdown]
# # PointDiT demo -- fully self-contained (no subprocess to `main.py`)
#
# `run_pointdit_demo.py` reuses `main.py --evaluate_gen --eval_wild_images` via
# `subprocess.run(["python", "main.py", ...])`. This notebook instead inlines the *actual*
# setup/checkpoint-loading/generation logic from `main.py`'s `main(args)` (lines ~380-874) and
# `engine.py`'s `evaluate_img2point` (lines ~337-1650), trimmed to the one real, already-verified
# configuration this repo's demo scripts use: single/few wild images (no ground truth), model
# `PointDiT-L/16`, `feature_embedding_type dinov3_vitl16`, checkpoint
# `pretrained/pointditl-512-mixdata-nodinov3-240c1a4f.pth` (ships EMA weights, DINOv3-stripped),
# `img_size 512`, `num_sampling_steps 3`.
#
# **What's copied verbatim vs. imported.** `main.py`'s and `engine.py`'s own *control flow* --
# the branches that decide what actually happens for this configuration (DINOv3 loading, the
# `dinov3_stripped`/`has_ema` checkpoint branches, the EMA swap, the per-image generation loop,
# the depth-PNG/point-cloud-PLY saving code) -- is copied verbatim into this notebook's own
# cells, with dead branches trimmed and a one-line comment on each explaining why it never runs
# for this configuration. Library code that both `main.py`/`engine.py` themselves just import
# and call -- `Denoiser` (denoiser.py), `WildImagesDataset` (dataloader/eval_depth.py),
# `viz_depth_tensor` (util/viz_depth.py), `depth_flying_points`/`save_single_point_cloud`
# (util/viz_pointcloud.py), `misc.add_weight_decay`/`misc.get_world_size`/`misc.get_rank`
# (util/misc.py), `repo_path` (util/paths.py) -- is imported normally, exactly as those two files
# do, since it doesn't change per-configuration and isn't what "self-contained" is about here.
#
# One exception: `get_args_parser` is imported from `main.py` rather than hand-transcribing its
# ~150 `argparse.add_argument(...)` calls. It is pure, configuration-independent argparse
# scaffolding (no branches to trace/trim), and hand-copying it would risk exactly the kind of
# transcription error this notebook is meant to avoid. `args = get_args_parser().parse_args([...])`
# is called with the same command-line flags `run_pointdit_demo.py` passed to `main.py`'s
# subprocess, so every other flag lands on its real argparse default -- identical to the
# already-verified subprocess runs.
#
# **Critical real-code detail found while tracing `main.py`'s `evaluate_gen` call site (line
# 873):** `evaluate_img2point(model_without_ddp, args, 0, data_loader_test, log_writer=None)` is
# called *without* `accelerator=...`, so `accelerator` is `None` inside `evaluate_img2point` for
# this exact call path -- every `if accelerator is not None:` branch in that function takes its
# `else` (plain `builtins.print`, `misc.get_rank()`/`misc.get_world_size()` instead of
# `accelerator.print`/`.process_index`/`.num_processes`). This is traced and preserved exactly,
# not assumed.
#
# **Section order** follows `main.py`'s real execution order (dataset + model are built, and
# checkpoint loading happens, all before section headers rather than the reverse) rather than
# the section order of a written outline, since e.g. `accelerator.prepare(model, optimizer,
# data_loader_test)` genuinely requires `data_loader_test` to already exist.
#
# **Cross-cell context managers.** `main.py` wraps the whole `evaluate_gen` call in
# `with torch.random.fork_rng(): torch.manual_seed(seed); with torch.no_grad():`. A `with` block
# cannot span multiple notebook cells, so section 5 below enters these two context managers
# manually via `__enter__()` in its first cell and exits them via `__exit__()` in its last cell --
# functionally identical to the real nested `with` statements, just spread across cells for
# readability.

# %% [markdown]
# ## 1. Setup and imports

# %%
import builtins
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta

from util import misc
from util.paths import repo_path
from util.viz_depth import viz_depth_tensor
from util.viz_pointcloud import depth_flying_points, save_single_point_cloud
from denoiser import Denoiser
from dataloader.eval_depth import WildImagesDataset

# %% [markdown]
# ## 2. Configuration (parameters)

# %% tags=["parameters"]
# Directory of .jpg/.png images to run PointDiT on (recursive). Papermill overrides this.
image_dir = "assets/demo"
checkpoint = "pretrained/pointditl-512-mixdata-nodinov3-240c1a4f.pth"
img_size = 512
num_sampling_steps = 3
repo_root = str(Path(__file__).resolve().parent) if "__file__" in dir() else os.getcwd()

# %%
# Driver bookkeeping (not from main.py/engine.py) -- same as run_pointdit_demo.py: fix the
# working directory so every relative path in `args` (checkpoint, --eval_wild_images_dir,
# pretrained/dinov3/...) resolves the same way it did for the subprocess-based demo, and list
# the input images up front so the final verification cells can pair them with their outputs.
os.chdir(repo_root)
os.environ.setdefault("DINOV3_WEIGHTS_DIR", "pretrained/dinov3")
print("repo_root:", repo_root)
print("image_dir:", image_dir)
assert Path(image_dir).is_dir(), f"image_dir does not exist: {image_dir}"
images = sorted([p for p in Path(image_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
print(f"found {len(images)} image(s):", [str(p) for p in images])
assert images, f"no .jpg/.png files found under {image_dir}"

# %%
# main.py's own argparse builder (get_args_parser, lines 52-377) -- imported, not
# hand-transcribed; see markdown note above. Parsed with the same flags
# run_pointdit_demo.py's subprocess call used, so every other argument lands on its real
# argparse default.
from main import get_args_parser

argv = [
    "--model", "PointDiT-L/16",
    "--feature_embedding_type", "dinov3_vitl16",
    "--proj_dropout", "0.0",
    "--evaluate_gen",
    "--eval_wild_images",
    "--eval_wild_images_dir", image_dir,
    "--img_size", str(img_size),
    "--num_sampling_steps", str(num_sampling_steps),
    "--pretrained", checkpoint,
]
args = get_args_parser().parse_args(argv)
print(args)

# %% [markdown]
# ## 3. Build the accelerator, dataset, and model
#
# ### 3a. Accelerator + seeding
#
# main.py lines 384-403 (verbatim). The wandb/tensorboard logging block that follows in the
# real source (lines 405-451, guarded by `not args.evaluate_gen`) is trimmed: `args.evaluate_gen`
# is always `True` in this notebook, so `log_writer` stays `None` and that block never runs; the
# same is true of the `if not args.evaluate_gen:` argument-dump print at lines 449-451.

# %%
kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))  # 30 minutes
accelerator = Accelerator(cpu=(args.device == 'cpu'), kwargs_handlers=[kwargs])
device = accelerator.device
global_rank = accelerator.process_index
print = accelerator.print

if getattr(args, 'remove_sky', False) and args.remove_sky_metric == 'depth' and args.use_sky_dome:
    print("[Warning] --remove_sky_metric depth is unreliable with --use_sky_dome "
          "(dome sky has small z near the horizon); using 'norm' instead.")
    args.remove_sky_metric = 'norm'

seed = args.seed + global_rank
torch.manual_seed(seed)
np.random.seed(seed)

cudnn.benchmark = True

# %% [markdown]
# ### 3b. Wild-images test dataset and dataloader
#
# main.py lines 557-641 (verbatim for our config). The `if not args.evaluate_gen:` train-loader
# block (lines 453-555: `ImageDepthIntrinsicsDataset`/`MixedImageDepthIntrinsicsDataset`
# construction) is trimmed -- `args.evaluate_gen` is always `True`, so that branch never runs.
# Within the test-loader block, `args.eval_depth_dataset` is always `False` here (we always pass
# `--eval_wild_images`, never `--eval_depth_dataset`), so the `EvalDepthDataset`/
# `MultiEvalDepthDataset` branch (lines 569-596) is trimmed too; only the `elif
# args.eval_wild_images:` branch (597-626) and the shared DataLoader construction (630-639) run.

# %%
use_original_res = getattr(args, 'eval_depth_original_resolution', False)
wild_full_frame = args.eval_wild_images and args.eval_depth_resize_height is None
if use_original_res or wild_full_frame:
    if args.batch_size != 1:
        cause = '--eval_depth_original_resolution' if use_original_res else 'full-frame wild inference'
        print(f"[Warning] {cause} requires batch_size=1, overriding batch_size={args.batch_size} to 1")
        args.batch_size = 1

assert args.eval_wild_images_dir is not None, "--eval_wild_images_dir must be set when using --eval_wild_images"
if use_original_res or args.eval_depth_resize_height is None:
    eval_crop_size = None
    eval_resize_height = None
else:
    eval_crop_size = args.img_size
    eval_resize_height = args.eval_depth_resize_height
# Name the output subdirectory after the image folder, so results are easy to trace back to
# their input (normpath first, or a trailing slash yields '').
if args.eval_wild_images_name is None:
    args.eval_wild_images_name = os.path.basename(
        os.path.normpath(args.eval_wild_images_dir)) or 'wild'
dataset_test = WildImagesDataset(
    images_dir=args.eval_wild_images_dir,
    dataset_name=args.eval_wild_images_name,
    crop_size=eval_crop_size,
    resize_height=eval_resize_height,
    max_samples=args.eval_depth_max_samples,
)

print('Test dataset:', dataset_test)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    shuffle=False,
    batch_size=args.gen_bsz,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=True,
    persistent_workers=args.persistent_workers,
)

# %% [markdown]
# ### 3c. Denoiser model, optimizer, and `accelerator.prepare`
#
# main.py lines 643-681 (verbatim for our config). The `if accelerator.is_main_process and
# args.wandb: wandb.log(...)` call (649-650) and the two `if not args.evaluate_gen: print(...)`
# lr/batch-size prints (661-663, 669-670) are trimmed -- `args.wandb` is always `False` and
# `args.evaluate_gen` is always `True` here. `accelerator.prepare` is called with the
# 3-argument, `evaluate_gen`-only signature (674-676); the training 4-argument call (678-680)
# is trimmed since it requires `not args.evaluate_gen`.

# %%
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.optimize_ddp = False

model = Denoiser(args)
model.to(device)

eff_batch_size = args.batch_size * accelerator.num_processes
if args.lr is None:  # only base_lr (blr) is specified
    args.lr = args.blr * eff_batch_size / 256

param_groups = misc.add_weight_decay(model, args.weight_decay,
                                      feature_embedding_lr_scale=args.feature_embedding_lr_scale)
optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

model, optimizer, data_loader_test = accelerator.prepare(model, optimizer, data_loader_test)
model_without_ddp = accelerator.unwrap_model(model)

accelerator.wait_for_everyone()

# %% [markdown]
# ### 3d. Load the frozen DINOv3 encoder
#
# main.py lines 687-725 (verbatim). The `else: raise NotImplementedError(...)` arm (717-718,
# for a `--feature_embedding_type` that isn't DINOv3) never runs since ours always starts with
# `dinov3`; the `if not args.evaluate_gen: print(model)` (720-721) is trimmed since
# `args.evaluate_gen` is always `True`.

# %%
dinov3_weights_loaded = False
if args.feature_embedding_type.startswith('dinov3'):
    vit_type = args.feature_embedding_type.split('_')[-1]
    assert vit_type in ['vits16', 'vits16plus', 'vitb16', 'vitl16', 'vith16plus', 'vit7b16'], f'ViT type {vit_type} not supported for DINOv3 feature embedding'

    sha_dict = {
        'vits16': '08c60483',
        'vits16plus': '4057cbaa',
        'vitb16': '73cec8be',
        'vitl16': '8aa4cbdd',
        'vith16plus': '7c1da9a5',
        'vit7b16': 'a955f4ea',
    }
    sha = sha_dict[vit_type]
    dinov3_weights_dir = os.environ.get(
        'DINOV3_WEIGHTS_DIR', repo_path('pretrained/dinov3'))
    ckpt_path = os.path.join(
        dinov3_weights_dir, f'dinov3_{vit_type}_pretrain_lvd1689m-{sha}.pth')

    if os.path.exists(ckpt_path):
        ckpts = torch.load(ckpt_path, map_location='cpu')
        model_without_ddp.net.y_embedder.load_state_dict(ckpts, strict=True)
        dinov3_weights_loaded = True
        print(f'Loaded pretrained DINOv3 {vit_type} from {ckpt_path}')
    elif args.pretrained is None:
        print(f'[DINOv3] {ckpt_path} not found -> encoder left randomly initialised. '
              f'Download the gated LVD-1689M weights to train from scratch.')
else:
    raise NotImplementedError(f'Feature embedding type {args.feature_embedding_type} not supported')

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: {:.2f}M".format(n_params / 1e6))
n_params_non_train = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print("Number of non-trainable parameters: {:.2f}M".format(n_params_non_train / 1e6))

# %% [markdown]
# ## 4. Load the pretrained checkpoint
#
# main.py lines 731-865 (verbatim for our config). `args.resume` is always `False` here (we
# always use `--pretrained`, never `--resume`), so the `if args.resume:` block (lines 732-747,
# which looks for `checkpoint-last.pth`/`checkpoint-last-prev.pth` in `--output_dir`) is
# trimmed and `checkpoint` stays `None` going into the next branch -- collapsing
# `if checkpoint is not None: ... elif args.pretrained: ...` (748-857) to just the
# (Renamed to `loaded_checkpoint` in this notebook: main.py's local variable is called
# `checkpoint`, which collides with this notebook's own `checkpoint` *parameter* -- the
# pretrained-file path, matching run_pointdit_demo.py's parameterization -- so we keep that
# parameter name intact and rename only the loaded state-dict variable.)
# `args.pretrained` arm, since `args.pretrained` is always set. The final
# `else: ... "Training from scratch"` (859-862) is trimmed for the same reason.
#
# ### 4a. Handle the DINOv3-stripped checkpoint
#
# main.py's `--resize_posemb` (790-796) and `--resize_patch_embed` (803-807) blocks are
# trimmed: both default to off (`False` / `0`) and this notebook never sets them.

# %%
loaded_checkpoint = None  # main.py's args.resume branch (dead here) would otherwise populate this

assert os.path.exists(args.pretrained)
loaded_checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)

# The released checkpoints ship without the frozen DINOv3 encoder: those weights are
# gated and cannot be redistributed
dinov3_stripped = loaded_checkpoint.get('dinov3_stripped')
if dinov3_stripped is not None and not dinov3_weights_loaded:
    raise FileNotFoundError(
        f'"{args.pretrained}" was released without the frozen DINOv3 '
        f'{dinov3_stripped.get("variant", vit_type)} encoder, and the encoder weights '
        f'were not found at "{ckpt_path}".\n'
        f'Request access at https://github.com/facebookresearch/dinov3, then place '
        f'{dinov3_stripped.get("upstream_filename", "the weights")} in '
        f'pretrained/dinov3/, or set DINOV3_WEIGHTS_DIR to the directory holding it.')

# A checkpoint carries up to two EMA copies. The released ones keep only 'model_ema1'
ema_keys = [k for k in ('model_ema1', 'model_ema2') if k in loaded_checkpoint]
has_ema = 'model_ema1' in loaded_checkpoint

if dinov3_stripped is not None and not args.no_strict_load:
    # Tolerate exactly the encoder keys the release removed -- already loaded above --
    # and keep every other mismatch fatal, which a blanket --no_strict_load would not.
    prefix = dinov3_stripped.get('prefix', 'net.y_embedder.')
    incompatible = model_without_ddp.load_state_dict(loaded_checkpoint['model'], strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith(prefix)]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            'Error(s) in loading state_dict for {}:\n\tMissing keys: {}\n\t'
            'Unexpected keys: {}'.format(type(model_without_ddp).__name__,
                                         missing, incompatible.unexpected_keys))
else:
    model_without_ddp.load_state_dict(loaded_checkpoint['model'], strict=not args.no_strict_load)

if args.evaluate_gen and 'epoch' in loaded_checkpoint:
    args.start_epoch = loaded_checkpoint['epoch']

# %% [markdown]
# ### 4b. Restore EMA weights
#
# main.py lines 826-857. `has_ema` is always `True` for this checkpoint (it ships
# `model_ema1`), so the `if not has_ema:` fallback (826-832, which would leave
# `ema_params1`/`ema_params2` as `None` and evaluate the raw "model" weights directly) never
# runs; only the `else:` branch (833-855) executes.

# %%
ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))

print('resume pretrained ema')
ema_state_dict1 = loaded_checkpoint['model_ema1']
# Only ema_params1 is read at evaluation time, so the released checkpoints drop the
# second copy; the partial resume below then falls back to the loaded weights for it.
ema_state_dict2 = loaded_checkpoint.get('model_ema2', {})

# partial resume
model_without_ddp.ema_params1 = []
model_without_ddp.ema_params2 = []
for i, (name, _) in enumerate(model_without_ddp.named_parameters()):
    if name in ema_state_dict1:
        model_without_ddp.ema_params1.append(ema_state_dict1[name].to(device))
    else:
        model_without_ddp.ema_params1.append(ema_params1[i])

    if name in ema_state_dict2:
        model_without_ddp.ema_params2.append(ema_state_dict2[name].to(device))
    else:
        model_without_ddp.ema_params2.append(ema_params2[i])

print('Load pretrained model from', args.pretrained)

accelerator.wait_for_everyone()

# %% [markdown]
# ## 5. Generation loop
#
# engine.py's `evaluate_img2point(model_without_ddp, args, epoch, data_loader, log_writer=None,
# accelerator=None)`, inlined for the exact call main.py's `if args.evaluate_gen:` block makes
# (line 873): `evaluate_img2point(model_without_ddp, args, 0, data_loader_test,
# log_writer=None)` -- note **no** `accelerator=` kwarg, so `accelerator` is `None` inside the
# real function at this call site. Every `if accelerator is not None:` branch below therefore
# takes its `else` (plain `builtins.print`, `misc.get_rank()`/`misc.get_world_size()`); this is
# traced from the real call site, not assumed.
#
# main.py lines 867-871 wrap this whole call in
# `with torch.random.fork_rng(): torch.manual_seed(seed); with torch.no_grad():`. Since a `with`
# block can't span notebook cells, both context managers are entered manually here and exited
# manually at the end of section 5 -- functionally identical to the real nested `with`.

# %%
_fork_rng_ctx = torch.random.fork_rng()
_fork_rng_ctx.__enter__()
torch.manual_seed(seed)
_no_grad_ctx = torch.no_grad()
_no_grad_ctx.__enter__()

# %% [markdown]
# ### 5a. Helper, EMA switch, and output-folder setup
#
# `combine_keep_masks` is engine.py's own module-level helper (lines 43-55), copied verbatim
# since we inline `evaluate_img2point` rather than importing it. engine.py lines 337-475
# (trimmed): the per-dataset/per-sample/SI-boundary/depth-range metric accumulators (350-391)
# are dropped except the few sums actually read later, since `is_wild_images` is `True` for
# every sample in this loop and none of that ground-truth-metric bookkeeping is ever
# populated for wild images. `args.online_eval` and `args.wandb` are always `False` here, so
# the `if args.online_eval:` viz-counter setup (393-407) and `if args.wandb:` metric-definition
# block (409-417) are trimmed too.

# %%
def combine_keep_masks(*masks):
    """AND together optional (H, W) bool keep-masks (True = keep). None = no constraint."""
    result = None
    for m in masks:
        if m is None:
            continue
        result = m if result is None else (result & m)
    return result


model_without_ddp.eval()

world_size = misc.get_world_size()
local_rank = misc.get_rank()
print = builtins.print

total_affine_rel_depth = 0.
total_affine_delta1_depth = 0.
total_affine_rel_point = 0.
total_affine_delta1_point = 0.
total_si_boundary_f1 = 0.
num_si_boundary_samples = 0
num_valid_samples = 0

total_inference_time = 0.0
inference_sample_count = 0
warmup_samples = 5

epoch = 0  # matches the `epoch=0` positional argument at the real evaluate_gen call site
val_logs = {'epoch': epoch}

# Construct the folder name for saving generated results (engine.py lines 422-456).
if args.pretrained is not None and args.evaluate_gen and not args.eval_no_save_gen:
    parts = [p for p in args.pretrained.split('/') if p]
    parts[-1] = os.path.splitext(parts[-1])[0]
    target_dir = '-'.join(parts[-3:])
else:
    target_dir = 'online' if args.online_eval else 'randinit'
save_folder = os.path.join(args.gen_output_root or repo_path('generation'), target_dir)

if args.save_path_postfix is not None and not args.eval_no_save_gen:
    save_folder += f'-{args.save_path_postfix}'
if args.eval_wild_images:
    print("Save to:", os.path.join(save_folder, args.eval_wild_images_name))
else:
    print("Save to:", save_folder)

if misc.get_rank() == 0 and not os.path.exists(save_folder):
    os.makedirs(save_folder)

if not args.eval_no_ema and getattr(model_without_ddp, 'ema_params1', None) is not None:
    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

logged_resolutions = set()
device = next(model_without_ddp.parameters()).device

# %% [markdown]
# ### 5b. Per-image generation, saving depth PNGs and point-cloud PLYs
#
# engine.py lines 476-1271 (trimmed for wild images). The `eval_num_tokens`/
# `resize_input_nearest_res`/`eval_depth_original_resolution` resize branches (496-654) all
# require `is_eval_depth_loader` (`'intrinsics' in sample`), which is always `False` for
# `WildImagesDataset` samples -- only the `elif is_wild_images:` resize branch (655-684) ever
# runs. `args.distributed` is always `False`, so the post-generation
# `if args.distributed: torch.distributed.barrier()` (705-706) is trimmed. The entire
# ground-truth-comparison block (`if not is_wild_images:`, lines 748-909: `compute_affine_metrics`,
# per-dataset/per-sample metrics, DIODE indoor/outdoor split) is trimmed since it never runs for
# wild images (no ground truth). In the per-sample save loop, every branch gated on
# `not is_wild_images` (GT depth viz, GT/error-map/boundary-viz panels, the
# `save_generated_gt_point_cloud` / `EvalDepthDataset` point-cloud branches) is trimmed, as are
# `args.save_raw_depth_npy`, `args.save_separate_intermediate_plys`, and
# `args.save_separate_intermediate_depths` (all default `False` and never set here).

# %%
for i, sample in enumerate(data_loader_test):
    if i in [0, len(data_loader_test) // 2, len(data_loader_test) // 4, len(data_loader_test) * 3 // 4]:
        print("Generation step {}/{}".format(i, len(data_loader_test)))

    # Benchmark samples carry 'intrinsics'; wild images have neither 'intrinsics' nor
    # 'pointcloud' (inference-only, no GT).
    is_eval_depth_loader = 'intrinsics' in sample
    is_wild_images = 'pointcloud' not in sample and 'intrinsics' not in sample

    image = sample['image']  # [B, 3, H, W] in [0, 1]

    # image to [-1, 1]
    image_01 = image.to(device, non_blocking=True).to(torch.float32)
    image = image_01 * 2.0 - 1.0

    if not is_wild_images:
        pointcloud = sample['pointcloud'].to(device, non_blocking=True)
    else:
        pointcloud = None

    original_size = None
    eval_num_tokens = getattr(args, 'eval_num_tokens', 0)
    if is_wild_images:
        # Wild images arrive at their native resolution and have no GT to match, so the only
        # thing that matters is giving the model an input it generalizes well on. It only ever
        # saw one square resolution during training, so keep the token count close to that
        # budget: resize the whole frame with the aspect ratio preserved to the patch-aligned
        # resolution nearest the budget, predict there, and resize the prediction back to
        # native below. Aspect ratio is preserved rather than forced square, since a squashed
        # image is itself off-distribution.
        B, C, H, W = image.shape

        patch_size = getattr(model_without_ddp, 'patch_size', 16)
        target_tokens = getattr(args, 'eval_wild_target_tokens', 0) or eval_num_tokens
        if target_tokens <= 0:
            target_tokens = (args.img_size // patch_size) ** 2

        # Round rather than truncate, so the realized token count straddles the target
        # instead of always landing below it.
        resize_factor = ((target_tokens * patch_size ** 2) / (H * W)) ** 0.5
        new_H = max(patch_size, int(round(H * resize_factor / patch_size)) * patch_size)
        new_W = max(patch_size, int(round(W * resize_factor / patch_size)) * patch_size)

        # An input already at the training token count (a 512x512 image at --img_size 512)
        # needs no resize in either direction. Leaving original_size as None also skips the
        # resize-back below, instead of interpolating to the size it already has.
        if (new_H, new_W) != (H, W):
            original_size = (H, W)
            image = torch.nn.functional.interpolate(
                image, size=(new_H, new_W), mode='bilinear', align_corners=False
            )

    # generation
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if i >= warmup_samples:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_inference_start = time.perf_counter()

        sampled_pointcloud, intermediate_pointcloud = model_without_ddp.generate(image, return_intermediate_steps=True)

        if i >= warmup_samples:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_inference_end = time.perf_counter()
            batch_size = image.shape[0]
            total_inference_time += (t_inference_end - t_inference_start)
            inference_sample_count += batch_size

    sampled_pointcloud = sampled_pointcloud.float()

    # Resize prediction back to original resolution (use 'nearest' for depth/pointcloud)
    if original_size is not None:
        sampled_pointcloud = torch.nn.functional.interpolate(
            sampled_pointcloud, size=original_size, mode='nearest'
        )
        for k in intermediate_pointcloud.keys():
            intermediate_pointcloud[k] = torch.nn.functional.interpolate(
                intermediate_pointcloud[k].float(), size=original_size, mode='nearest'
            )

    # remove_sky: detect predicted sky on the RAW model output (normalized space), BEFORE the
    # radial-log inverse and scale/shift alignment. See engine.py lines 721-728 for the full
    # rationale.
    pred_sky_mask = None
    if getattr(args, 'remove_sky', False):
        if getattr(args, 'remove_sky_metric', 'norm') == 'depth':
            sky_metric = sampled_pointcloud[:, 2]            # [B, H, W]
        else:
            sky_metric = sampled_pointcloud.norm(dim=1)      # [B, H, W]
        pred_sky_mask = sky_metric > args.remove_sky_threshold  # [B, H, W] bool, True = sky

    raw_sampled_pointcloud = None
    align_outputs = {}
    affine_points_scale = torch.tensor(1.0, device=sampled_pointcloud.device)
    affine_points_shift = torch.zeros(1, 3, 1, 1, device=sampled_pointcloud.device)

    # distributed save depths
    for b_id in range(sampled_pointcloud.size(0)):
        img_id = i * sampled_pointcloud.size(0) * world_size + local_rank * sampled_pointcloud.size(0) + b_id

        if 'dataset_name' in sample:
            dataset_name = sample['dataset_name'][b_id] if isinstance(sample['dataset_name'], (list, tuple)) else sample['dataset_name']
            dataset_save_folder = os.path.join(save_folder, dataset_name)
            if not args.online_eval and not args.eval_no_save_gen:
                if misc.get_rank() == 0:
                    os.makedirs(dataset_save_folder, exist_ok=True)
        else:
            dataset_save_folder = save_folder

        # Use sample_id for filename if available (WildImagesDataset always provides one)
        if 'sample_id' in sample:
            sample_id_raw = sample['sample_id'][b_id] if isinstance(sample['sample_id'], list) else sample['sample_id']
            if args.eval_wild_images:
                # Mirror the input filename, so sample0.png yields sample0_depth.png and
                # sample0_pointcloud.ply.
                file_prefix = os.path.splitext(sample_id_raw)[0]
                sub = os.path.dirname(file_prefix)
                if sub and not args.online_eval and not args.eval_no_save_gen:
                    os.makedirs(os.path.join(dataset_save_folder, sub), exist_ok=True)
            else:
                file_prefix = f"{img_id:05d}_{sample_id_raw.replace('/', '_')}"
        else:
            file_prefix = f'{img_id:05d}'

        # remove_sky: per-sample keep-mask (True = keep, False = sky to drop)
        sky_keep_mask = None
        if pred_sky_mask is not None:
            sky_keep_mask = (~pred_sky_mask[b_id]).detach().cpu().numpy()  # (H, W) bool

        # depth from z. Wild-image model output is zero-centered (center_shift_point +
        # normalize_point_by_mean) with no affine alignment, so we must not clamp to 0 and must
        # use shifted_depth=True.
        sampled_depth = sampled_pointcloud[b_id, 2]  # [H, W], may be negative
        sampled_depth_viz = sampled_depth
        if pred_sky_mask is not None:
            sampled_depth_viz = torch.where(pred_sky_mask[b_id], torch.full_like(sampled_depth_viz, 1e6), sampled_depth_viz)
        depth_viz = viz_depth_tensor(sampled_depth_viz, shifted_depth=True, colormap=args.depth_colormap)  # [H, W, 3] np

        img_viz = (image_01[b_id].permute(1, 2, 0).detach().cpu() * 255.).to(torch.uint8).numpy()

        # remove_sky: [input | sky-mask | overlay] panel. Computed for parity with the real
        # code, but never saved for wild images (see the save gate below).
        skymask_concat = None
        if pred_sky_mask is not None:
            sky_b = pred_sky_mask[b_id].detach().cpu().numpy()  # (H, W) bool, True = sky
            mask_viz = np.repeat((sky_b[:, :, None].astype(np.uint8) * 255), 3, axis=2)  # (H, W, 3)
            overlay = img_viz.copy()
            overlay[sky_b] = (0.5 * overlay[sky_b].astype(np.float32)
                              + 0.5 * np.array([255, 0, 0], dtype=np.float32)).astype(np.uint8)
            skymask_concat = np.concatenate([img_viz, mask_viz, overlay], axis=1)

        concat_list = [img_viz]

        # save the intermediate generation results (computed for parity with the real code,
        # but engine.py only appends depth_k_viz to concat_list `if not is_wild_images` --
        # for wild images the panel is built and thrown away, exactly like the real run).
        for k, v in intermediate_pointcloud.items():
            v = v * affine_points_scale + affine_points_shift  # [B, 3, H, W]
            depth_k = v[b_id, 2].detach().cpu()  # [H, W], keep raw (zero-centered)
            depth_k_viz = viz_depth_tensor(depth_k, shifted_depth=True, colormap=args.depth_colormap)

        # final prediction
        concat_list.append(depth_viz)

        concat = np.concatenate(concat_list, axis=1)

        # save
        if not args.online_eval and not args.eval_no_save_gen:
            save_path = os.path.join(dataset_save_folder, f'{file_prefix}_depth.png')
            Image.fromarray(concat).save(save_path)
            # skymask_concat is only saved `and not args.eval_wild_images` in the real code --
            # we always pass --eval_wild_images, so it's never written here.

        # save point cloud
        save_path_pointcloud = os.path.join(dataset_save_folder, f'{file_prefix}_pointcloud.ply')

        if not args.online_eval and not args.eval_no_save_gen:
            # is_eval_depth_loader is always False for wild images, so only the
            # `if is_wild_images:` arm of engine.py's 3-way point-cloud save (1144-1165) ever
            # runs.
            de_mask = None
            if getattr(args, 'remove_depth_edge', False):
                pred_depth_np = sampled_pointcloud[b_id][2].detach().cpu().numpy()
                de_mask = ~depth_flying_points(pred_depth_np, rtol=args.depth_edge_rtol)
            # remove_sky: drop predicted sky points
            de_mask = combine_keep_masks(de_mask, sky_keep_mask)
            save_single_point_cloud(
                sampled_pointcloud[b_id].permute(1, 2, 0).detach().cpu().numpy(),
                image_01[b_id].permute(1, 2, 0).detach().cpu().numpy(),
                filename=save_path_pointcloud,
                mask=de_mask,
            )

# %% [markdown]
# ### 5c. Switch back from EMA and final report
#
# engine.py lines 1273-1650 (trimmed). The per-sample-metrics JSON gather (1276-1303,
# `args.eval_save_per_sample_metrics` defaults `False` and is never set) and the
# per-dataset-metrics cross-process aggregation (1306-1401, `accelerator is None` and
# `not args.distributed` here, and `per_dataset_metrics` was never populated for wild images
# anyway) are both trimmed. The stats-tensor reduction (1414-1469) takes the final
# `else:` arm (`accelerator is None`, `not args.distributed`) directly. The average-metrics
# computation (1476-1483), the `if not args.eval_wild_images:` metrics print (1488-1491,
# `args.eval_wild_images` is always `True`), the depth-range-metrics block (1494-1515,
# `args.eval_depth_range_metrics` defaults `False`), the SI-boundary-F1 print (1517-1522, no
# downstream reader once wandb/tensorboard are trimmed), the per-dataset-metrics print
# (1534-1573, `per_dataset_metrics` is always empty for wild images), the tensorboard block
# (1575-1611, `log_writer` is always `None`), the wandb block (1613-1645, `args.wandb` is
# always `False`), and the final `if args.distributed: barrier()` (1647-1648) are all trimmed
# for the same reasons.

# %%
global_affine_rel_depth_sum = total_affine_rel_depth
global_affine_delta1_depth_sum = total_affine_delta1_depth
global_affine_rel_point_sum = total_affine_rel_point
global_affine_delta1_point_sum = total_affine_delta1_point

global_num_valid_samples = num_valid_samples
global_si_boundary_f1_sum = total_si_boundary_f1
global_num_si_boundary_samples = num_si_boundary_samples

if not args.eval_no_ema and getattr(model_without_ddp, 'ema_params1', None) is not None:
    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

# Inference speed metrics (only prints once inference_sample_count exceeds the 5-sample
# warmup -- a 1-2 image demo folder never reaches that, matching the real runs).
if inference_sample_count > 0:
    avg_time_per_image = total_inference_time / inference_sample_count
    fps = inference_sample_count / total_inference_time
    print(f"Inference speed: {avg_time_per_image*1000:.2f} ms/image, {fps:.2f} FPS (total: {inference_sample_count} images in {total_inference_time:.2f}s)")

if args.eval_wild_images:
    # No metrics to report, so close the run with something unambiguous instead.
    print("Done!")

# %%
_no_grad_ctx.__exit__(None, None, None)
_fork_rng_ctx.__exit__(None, None, None);

# %% [markdown]
# ## 6. Verify the real output -- actually look at the result, don't just trust that the loop ran

# %%
ckpt_tag = Path(checkpoint).stem
out_dir = Path("generation") / f"pretrained-{ckpt_tag}" / Path(image_dir).name
print("output dir:", out_dir)
depth_pngs = sorted(out_dir.glob("*_depth.png"))
ply_files = sorted(out_dir.glob("*_pointcloud.ply"))
print(f"{len(depth_pngs)} depth PNG(s), {len(ply_files)} point-cloud PLY(s)")
assert depth_pngs and ply_files, f"expected outputs not found in {out_dir}"

# %%
import matplotlib.pyplot as plt
from PIL import Image

fig, axes = plt.subplots(len(depth_pngs), 2, figsize=(10, 5 * len(depth_pngs)), squeeze=False)
for i, depth_png in enumerate(depth_pngs):
    src_img = images[i] if i < len(images) else None
    if src_img is not None:
        axes[i][0].imshow(Image.open(src_img))
        axes[i][0].set_title(f"input: {src_img.name}")
        axes[i][0].axis("off")
    axes[i][1].imshow(Image.open(depth_png))
    axes[i][1].set_title(f"depth: {depth_png.name}")
    axes[i][1].axis("off")
plt.tight_layout()
plt.show()

# %%
import trimesh

for ply in ply_files:
    mesh = trimesh.load(ply)
    n_points = len(mesh.vertices) if hasattr(mesh, "vertices") else "?"
    print(f"{ply.name}: {n_points} points, bbox {mesh.bounds.tolist() if hasattr(mesh, 'bounds') else 'n/a'}")
