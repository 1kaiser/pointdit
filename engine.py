# coding=utf-8

# Derived from JiT (https://github.com/LTH14/JiT), MIT License,
# Copyright (c) 2025 Tianhong Li. See THIRD_PARTY_NOTICES.
import math
import os
import builtins
import time

import torch
import numpy as np
from PIL import Image

from util import misc
from util import lr_sched
from util.paths import repo_path
import copy
import wandb

import torch.distributed as dist

from util.viz_depth import viz_depth_tensor
from util.log_tb import add_text_border
from util.viz_pointcloud import (depth_flying_points,
                                         save_generated_gt_point_cloud,
                                         save_single_point_cloud)
from loss import rel_point_loss
from metrics.compute_metrics import compute_metrics as compute_affine_metrics


def combine_keep_masks(*masks):
    """AND together optional (H, W) bool keep-masks (True = keep). None = no constraint.

    Returns None if all inputs are None, otherwise the element-wise AND of the
    non-None masks. Used to merge the depth-edge keep-mask with the sky-removal
    keep-mask before saving point clouds.
    """
    result = None
    for m in masks:
        if m is None:
            continue
        result = m if result is None else (result & m)
    return result


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None, accelerator=None, global_step=0):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ",
                                      print_func=print if accelerator is None else accelerator.print)
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    # GPU compute timing (when dataloader_debug_timing is enabled)
    if args.dataloader_debug_timing:
        gpu_timing = {'forward': 0.0, 'backward': 0.0, 'optimizer': 0.0, 'ema': 0.0}
        gpu_timing_count = 0

    optimizer.zero_grad()

    # Monotonic step counter, seeded from the resumed value so logging/checkpoint
    # intervals stay continuous across resumes (independent of dataset length).
    start_global_step = global_step

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Start forward timing
        if args.dataloader_debug_timing:
            torch.cuda.synchronize()
            t_forward_start = time.perf_counter()

        # normalize image to [-1, 1]
        # NOTE: only need to divide by 255 for original JiT dataloader
        # otherwise already handled in the dataloader
        image = sample['image']  # [B, 3, H, W] in [0, 1]
        pointcloud = sample['pointcloud']  # [B, 3, H, W]

        # image to [-1, 1]
        image = image.to(device, non_blocking=True).to(torch.float32)
        image = image * 2.0 - 1.0

        pointcloud = pointcloud.to(device, non_blocking=True)

        # accelerate's prepared dataloader moves batches to the device for us, plain
        # torchrun/single-GPU runs do not -- move the masks explicitly so both work.
        valid_mask = sample.get('valid_mask', None)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device, non_blocking=True)
        sky_mask = sample.get('sky_mask', None)
        if sky_mask is not None:
            sky_mask = sky_mask.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, model_in_out = model(pointcloud, image, return_model_input_output=True,
                force_zero_t=args.force_zero_t,
                args=args,
                valid_mask=valid_mask,
                sky_mask=sky_mask,
                )

        denoise_loss_value = loss.item()

        # relative point loss
        if args.rel_point_loss_weight > 0:
            assert 'output' in model_in_out
            x_pred = model_in_out['output']  # [B, 3, H, W]

            rel_loss_val = rel_point_loss(x_pred, pointcloud)  # [B, H, W]

            # exclude invalid points
            if args.exclude_invalid_gt:
                if args.center_shift_point:
                    # use valid mask from dataloader
                    assert 'valid_mask' in sample
                    valid_mask = sample['valid_mask'].to(device, non_blocking=True)
                    rel_loss_val = rel_loss_val[valid_mask].mean()
                else:
                    valid_mask = pointcloud[:, 2:] > 0  # [B, 1, H, W]
                    rel_loss_val = rel_loss_val[valid_mask.squeeze(1)].mean()
            else:
                rel_loss_val = rel_loss_val.mean()

            loss = loss + args.rel_point_loss_weight * rel_loss_val

            rel_point_loss_value = rel_loss_val.item()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, skipping backward pass".format(loss_value))
            # Skip backward/optimizer but continue to maintain sync across nodes
            optimizer.zero_grad()
            model_without_ddp.update_ema()
            metric_logger.update(loss=0.0)  # Log 0 to indicate skipped iteration
            continue

        # End forward timing, start backward timing
        if args.dataloader_debug_timing:
            torch.cuda.synchronize()
            t_forward_end = time.perf_counter()
            gpu_timing['forward'] += t_forward_end - t_forward_start

        optimizer.zero_grad()

        # Start backward timing
        if args.dataloader_debug_timing:
            torch.cuda.synchronize()
            t_backward_start = time.perf_counter()

        if accelerator is not None:
            # Use accelerate backward
            accelerator.backward(loss)

            if args.grad_clip:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        else:
            loss.backward()

        # End backward timing, start optimizer timing
        if args.dataloader_debug_timing:
            torch.cuda.synchronize()
            t_backward_end = time.perf_counter()
            gpu_timing['backward'] += t_backward_end - t_backward_start
            t_optimizer_start = time.perf_counter()

        optimizer.step()

        # End optimizer timing, start EMA timing
        if args.dataloader_debug_timing:
            torch.cuda.synchronize()
            t_optimizer_end = time.perf_counter()
            gpu_timing['optimizer'] += t_optimizer_end - t_optimizer_start
            t_ema_start = time.perf_counter()

        model_without_ddp.update_ema()

        # End EMA timing and log
        if args.dataloader_debug_timing:
            torch.cuda.synchronize()
            t_ema_end = time.perf_counter()
            gpu_timing['ema'] += t_ema_end - t_ema_start
            gpu_timing_count += 1

            if gpu_timing_count % 100 == 0:
                n = gpu_timing_count
                total_gpu = gpu_timing['forward'] + gpu_timing['backward'] + gpu_timing['optimizer'] + gpu_timing['ema']
                print(f"[GPU Timing] steps={n}, "
                      f"forward={gpu_timing['forward']/n*1000:.1f}ms, "
                      f"backward={gpu_timing['backward']/n*1000:.1f}ms, "
                      f"optimizer={gpu_timing['optimizer']/n*1000:.1f}ms, "
                      f"ema={gpu_timing['ema']/n*1000:.1f}ms, "
                      f"total_gpu={total_gpu/n*1000:.1f}ms")

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Debug early exit
        if args.debug_train_steps > 0 and data_iter_step + 1 >= args.debug_train_steps:
            print(f"[DEBUG] Exiting training loop early after {data_iter_step + 1} steps")
            break

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if accelerator is not None:
            is_main_process = accelerator.is_main_process 
        else:
            is_main_process = misc.is_main_process()

        # Monotonic global step (continuous across resumes).
        global_step = start_global_step + data_iter_step

        # Save checkpoint at step intervals (sub-epoch checkpointing).
        # Must run on ALL ranks: the barrier is a collective and save_model
        # internally guards the disk write with is_main_process(). Gating this on
        # rank-0-only conditions (e.g. log_writer) deadlocks distributed training.
        if args.save_last_steps > 0 and global_step > 0 and global_step % args.save_last_steps == 0:
            if accelerator is not None:
                accelerator.wait_for_everyone()
            elif misc.is_dist_avail_and_initialized():
                torch.distributed.barrier()
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last",
                global_step=global_step,
                resume_epoch=epoch
            )

        # log with tensorboard
        if log_writer is not None or (args.wandb and is_main_process):
            if global_step % args.log_freq == 0:
                if args.wandb:
                    log_values = {
                        'train/loss': loss_value_reduce,
                        'train/lr': lr,
                    }

                    if args.rel_point_loss_weight > 0:
                        log_values['train/denoise_loss'] = denoise_loss_value
                        log_values['train/rel_point_loss'] = rel_point_loss_value

                    wandb.log(log_values, step=global_step)
                else:
                    log_writer.add_scalar('train_loss', loss_value_reduce, global_step)
                    log_writer.add_scalar('lr', lr, global_step)

                # TODO: log training image and pointcloud
                # TODO: can use make_grid to show multiple samples

                if args.center_shift_point:
                    # shift back the pointcloud for visualization
                    assert 'center_shift' in sample
                    center_shift = sample['center_shift'].to(device, non_blocking=True)  # [B, 3, 1, 1]

                    pointcloud = pointcloud + center_shift
                    model_in_out['input'] = model_in_out['input'] + center_shift
                    model_in_out['output'] = model_in_out['output'] + center_shift

                num_viz_samples = min(4, pointcloud.shape[0])
                for sample_idx in range(num_viz_samples):
                    # TODO: filter invalid point
                    if pointcloud[sample_idx, 2].max() > 0 and model_in_out['output'][sample_idx, 2].max() > 0:
                        # log image and depth
                        img_viz = image[sample_idx].detach().cpu() * 0.5 + 0.5  # [3, H, W] in 0-1
                        depth_viz = viz_depth_tensor(pointcloud[sample_idx, 2].detach().cpu(), return_numpy=False, shifted_depth=False, colormap=args.depth_colormap) / 255.  # [3, H, W] in 0-1
                        # TODO: concat model prediction at different timesteps

                        tmp_t = model_in_out['t'][sample_idx].squeeze(0).item()
                        model_in = model_in_out['input'][sample_idx, 2].detach().cpu().float()  # [H, W]
                        model_out = model_in_out['output'][sample_idx, 2].detach().cpu().float()  # [H, W]

                        model_in_viz = viz_depth_tensor(model_in, return_numpy=False, shifted_depth=False, colormap=args.depth_colormap) / 255.
                        model_out_viz = viz_depth_tensor(model_out, return_numpy=False, shifted_depth=False, colormap=args.depth_colormap) / 255.

                        concat = torch.cat((img_viz, model_in_viz, model_out_viz, depth_viz), dim=-1)

                        # viz point cloud error
                        point_pred_error = torch.abs(model_in_out['output'][sample_idx] - pointcloud[sample_idx]).mean(0)  # [H, W]
                        error_normalized = (point_pred_error - point_pred_error.min()) / (point_pred_error.max() - point_pred_error.min() + 1e-8)
                        # exclude invalid regions
                        error_normalized[pointcloud[sample_idx, 2] <= 0] = 0.
                        error_normalized = error_normalized[None, :, :].repeat(3, 1, 1)  # [3, H, W]
                        concat = torch.cat((concat, error_normalized.detach().cpu()), dim=-1)

                        # viz valid_mask
                        if 'valid_mask' in sample:
                            valid_mask_viz = sample['valid_mask'][sample_idx].detach().cpu().float()  # [H, W]
                            valid_mask_viz = valid_mask_viz[None, :, :].repeat(3, 1, 1)  # [3, H, W]
                            concat = torch.cat((concat, valid_mask_viz), dim=-1)

                        # add time step info
                        concat = add_text_border(concat, f't = {tmp_t:.3f}')

                        if global_step % (args.log_freq * args.log_img_stride) == 0:
                            # log images less frequently to save space
                            if args.wandb:
                                wandb.log({
                                    f'train_viz/sample{sample_idx}': wandb.Image(concat.permute(1, 2, 0).numpy()),
                                }, step=global_step)

                            else:
                                log_writer.add_image(f'train/sample{sample_idx}', concat, global_step)


        if (args.local_rank == 0 and epoch == 0 and data_iter_step == 5
                and torch.cuda.is_available()):
            if accelerator is not None:
                if accelerator.is_main_process:
                    os.system("nvidia-smi")
            else:
                os.system("nvidia-smi")

    # Return the monotonic step counter so the next epoch continues from here.
    return global_step + 1


def evaluate_img2point(model_without_ddp, args, epoch, data_loader, log_writer=None, accelerator=None):

    model_without_ddp.eval()

    if accelerator is not None:
        print = accelerator.print
        world_size = accelerator.num_processes
        local_rank = accelerator.process_index
    else:
        world_size = misc.get_world_size()
        local_rank = misc.get_rank()
        print = builtins.print

    # metrics
    # affine metrics - track per dataset
    # Use defaultdict to automatically initialize metrics for each dataset
    from collections import defaultdict

    # Parse which datasets should compute SI boundary metrics
    boundary_datasets = set()
    if hasattr(args, 'eval_boundary_datasets') and args.eval_boundary_datasets:
        boundary_datasets = set(args.eval_boundary_datasets.split(','))

    per_dataset_metrics = defaultdict(lambda: {
        'total_affine_rel_depth': 0.,
        'total_affine_delta1_depth': 0.,
        'total_affine_rel_point': 0.,
        'total_affine_delta1_point': 0.,
        'num_valid_samples': 0,
        'total_si_boundary_f1': 0.,
        'num_si_boundary_samples': 0,
        'depth_range': {b: {'total_rel_depth': 0., 'total_delta1_depth': 0., 'total_rel_point': 0., 'total_delta1_point': 0., 'count': 0} for b in ['near', 'medium', 'far']},
    })

    # Per-sample metrics for detailed comparison (only if enabled)
    per_sample_metrics = [] if getattr(args, 'eval_save_per_sample_metrics', False) else None

    # Global metrics (aggregate across all datasets)
    total_affine_rel_depth = 0.
    total_affine_delta1_depth = 0.
    total_affine_rel_point = 0.
    total_affine_delta1_point = 0.
    total_si_boundary_f1 = 0.
    num_si_boundary_samples = 0

    # Depth-range stratified metrics (near/mid/far)
    depth_range_bins = ['near', 'medium', 'far']
    depth_range_global = {b: {'total_rel_depth': 0., 'total_delta1_depth': 0., 'total_rel_point': 0., 'total_delta1_point': 0., 'count': 0} for b in depth_range_bins}

    num_valid_samples = 0  # Track samples with enough valid points for metrics

    # Inference speed tracking (skip first 5 samples for warmup)
    total_inference_time = 0.0
    inference_sample_count = 0
    warmup_samples = 5

    if args.online_eval:
        # Check if using multiple datasets for visualization
        # We'll determine this by checking if MultiEvalDepthDataset is being used
        # This is indicated by having multiple dataset names in the loader
        is_multi_dataset_viz = hasattr(data_loader.dataset, 'datasets') and len(data_loader.dataset.datasets) > 1

        if is_multi_dataset_viz:
            # Multiple datasets: visualize 1 samples per dataset
            per_dataset_viz_counters = {}
            samples_per_dataset = 1
        else:
            # Single dataset: visualize 4 samples total
            num_val_samples = min(4, len(data_loader))
            eval_counter = 0
            log_eval_freq = len(data_loader) // num_val_samples

    if args.wandb:
        if accelerator is not None:
            if accelerator.is_main_process:
                wandb.define_metric("test_viz/*", step_metric="epoch")
                wandb.define_metric("test_metric/*", step_metric="epoch")
        else:
            if args.local_rank == 0:
                wandb.define_metric("test_viz/*", step_metric="epoch")
                wandb.define_metric("test_metric/*", step_metric="epoch")

    # 1. Initialize an empty dictionary before the loop
    val_logs = {'epoch': epoch}

    # Construct the folder name for saving generated results.
    if args.pretrained is not None and args.evaluate_gen and not args.eval_no_save_gen:
        # Name the output folder after (up to) the last three components of the checkpoint
        # path, so runs from different checkpoint directories do not collide.
        parts = [p for p in args.pretrained.split('/') if p]
        parts[-1] = os.path.splitext(parts[-1])[0]
        target_dir = '-'.join(parts[-3:])
    else:
        # No checkpoint to name the folder after: online eval during training, or a
        # standalone run over a randomly initialised model. Kept distinct from each other.
        target_dir = 'online' if args.online_eval else 'randinit'
    # Defaults to <repo>/generation rather than ./generation, so results land next
    # to the code even when the command is launched from elsewhere. Deliberately not inside
    # the image directory: generation/ is gitignored, an --eval_wild_images_dir may not be.
    save_folder = os.path.join(
        args.gen_output_root or repo_path('generation'),
        target_dir,
    )

    if args.save_path_postfix is not None and not args.eval_no_save_gen:
        save_folder += f'-{args.save_path_postfix}'
    if args.eval_wild_images:
        # A wild run has exactly one output subdirectory, so report the full path the files
        # actually land in rather than its parent.
        print("Save to:", os.path.join(save_folder, args.eval_wild_images_name))
    else:
        print("Save to:", save_folder)

    if accelerator is not None:
        if accelerator.is_main_process:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
    else:
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

    # Track logged resolutions to avoid spam
    logged_resolutions = set()

    # Follow the model rather than assuming CUDA, so evaluation and the wild-image
    # demo also run on a CPU-only machine.
    device = next(model_without_ddp.parameters()).device

    # data iter
    for i, sample in enumerate(data_loader):
        if i in [0, len(data_loader) // 2, len(data_loader) // 4, len(data_loader) * 3 // 4]:
            print("Generation step {}/{}".format(i, len(data_loader)))

        # Benchmark samples carry 'intrinsics'; wild images do not.
        # WildImagesDataset has neither (inference-only, no GT)
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

        # Handle original resolution evaluation: resize input to div-16, then resize output back
        original_size = None
        eval_num_tokens = getattr(args, 'eval_num_tokens', 0)
        if eval_num_tokens > 0 and is_eval_depth_loader:
            # MoGe-style token-based resizing: resize to match target token count while preserving aspect ratio
            B, C, H, W = image.shape
            original_size = (H, W)

            # MoGe-style resize factor calculation
            # Formula: resize_factor = sqrt((num_tokens * patch_size^2) / (H * W))
            patch_size = 16  # PointDiT uses 16x16 patches (vs MoGe's 14x14)
            resize_factor = ((eval_num_tokens * patch_size ** 2) / (H * W)) ** 0.5
            new_H = int(H * resize_factor)
            new_W = int(W * resize_factor)

            # Round to nearest multiple of patch_size (16)
            new_H = (new_H // patch_size) * patch_size
            new_W = (new_W // patch_size) * patch_size

            # Ensure minimum size
            new_H = max(new_H, patch_size)
            new_W = max(new_W, patch_size)

            if (new_H, new_W) != (H, W):
                image = torch.nn.functional.interpolate(
                    image, size=(new_H, new_W), mode='bilinear', align_corners=False
                )
                dataset_name = sample['dataset_name'][0] if isinstance(sample.get('dataset_name'), (list, tuple)) else sample.get('dataset_name', 'unknown')
                if (dataset_name, H, W, new_H, new_W) not in logged_resolutions:
                    logged_resolutions.add((dataset_name, H, W, new_H, new_W))
                    print(f"[eval_num_tokens={eval_num_tokens}] Resized {dataset_name} input: ({H}, {W}) -> ({new_H}, {new_W})")
            else:
                # No resize needed, but still log the resolution info
                dataset_name = sample['dataset_name'][0] if isinstance(sample.get('dataset_name'), (list, tuple)) else sample.get('dataset_name', 'unknown')
                actual_tokens = (new_H // patch_size) * (new_W // patch_size)
                if (dataset_name, H, W, new_H, new_W) not in logged_resolutions:
                    logged_resolutions.add((dataset_name, H, W, new_H, new_W))
                    print(f"[eval_num_tokens={eval_num_tokens}] {dataset_name}: ({H}, {W}) -> no resize, tokens={actual_tokens}")

        elif getattr(args, 'resize_input_nearest_res', False) and is_eval_depth_loader:
            # Resize to nearest resolution divisible by 16, with optional token constraints
            B, C, H, W = image.shape
            original_size = (H, W)
            patch_size = 16

            # Round to NEAREST multiple of 16 (not floor)
            nearest_H = max(patch_size, int(round(H / patch_size)) * patch_size)
            nearest_W = max(patch_size, int(round(W / patch_size)) * patch_size)
            nearest_tokens = (nearest_H // patch_size) * (nearest_W // patch_size)

            eval_num_tokens = getattr(args, 'eval_num_tokens', 0)

            if eval_num_tokens > 0:
                # Override: use specific token count
                target_tokens = eval_num_tokens
                resize_factor = ((target_tokens * patch_size ** 2) / (nearest_H * nearest_W)) ** 0.5
                new_H = max(patch_size, int(round(nearest_H * resize_factor / patch_size)) * patch_size)
                new_W = max(patch_size, int(round(nearest_W * resize_factor / patch_size)) * patch_size)
            else:
                # Use min/max token logic
                min_tokens = getattr(args, 'eval_min_tokens', 0)
                max_tokens = getattr(args, 'eval_max_tokens', 0)

                if min_tokens > 0 and nearest_tokens < min_tokens:
                    target_tokens = min_tokens
                    resize_factor = ((target_tokens * patch_size ** 2) / (nearest_H * nearest_W)) ** 0.5
                    # Use ceil to ensure tokens >= min_tokens
                    new_H = max(patch_size, math.ceil(nearest_H * resize_factor / patch_size) * patch_size)
                    new_W = max(patch_size, math.ceil(nearest_W * resize_factor / patch_size) * patch_size)
                elif max_tokens > 0 and nearest_tokens > max_tokens:
                    target_tokens = max_tokens
                    resize_factor = ((target_tokens * patch_size ** 2) / (nearest_H * nearest_W)) ** 0.5
                    # Use floor to ensure tokens <= max_tokens
                    new_H = max(patch_size, int(nearest_H * resize_factor / patch_size) * patch_size)
                    new_W = max(patch_size, int(nearest_W * resize_factor / patch_size) * patch_size)
                else:
                    # Tokens within range, use nearest-16 resolution
                    new_H, new_W = nearest_H, nearest_W

            dataset_name = sample['dataset_name'][0] if isinstance(sample.get('dataset_name'), (list, tuple)) else sample.get('dataset_name', 'unknown')
            actual_tokens = (new_H // patch_size) * (new_W // patch_size)

            if (new_H, new_W) != (H, W):
                image = torch.nn.functional.interpolate(
                    image, size=(new_H, new_W), mode='bilinear', align_corners=False
                )
                if (dataset_name, H, W, new_H, new_W) not in logged_resolutions:
                    logged_resolutions.add((dataset_name, H, W, new_H, new_W))
                    print(f"[resize_input_nearest_res] {dataset_name}: ({H}, {W}) -> ({new_H}, {new_W}), tokens={actual_tokens}")
            else:
                # No resize needed, but still log the resolution info
                if (dataset_name, H, W, new_H, new_W) not in logged_resolutions:
                    logged_resolutions.add((dataset_name, H, W, new_H, new_W))
                    print(f"[resize_input_nearest_res] {dataset_name}: ({H}, {W}) -> no resize, tokens={actual_tokens}")

        elif getattr(args, 'eval_depth_original_resolution', False) and (is_eval_depth_loader or is_wild_images):
            B, C, H, W = image.shape
            original_size = (H, W)

            patch_size = 16
            min_tokens = getattr(args, 'eval_min_tokens', 0)
            max_tokens = getattr(args, 'eval_max_tokens', 0)

            if min_tokens > 0 or max_tokens > 0:
                # Token-based resizing: compute target from original aspect ratio
                nearest_H = max(patch_size, int(round(H / patch_size)) * patch_size)
                nearest_W = max(patch_size, int(round(W / patch_size)) * patch_size)
                nearest_tokens = (nearest_H // patch_size) * (nearest_W // patch_size)

                if min_tokens > 0 and nearest_tokens < min_tokens:
                    resize_factor = ((min_tokens * patch_size ** 2) / (nearest_H * nearest_W)) ** 0.5
                    new_H = max(patch_size, math.ceil(nearest_H * resize_factor / patch_size) * patch_size)
                    new_W = max(patch_size, math.ceil(nearest_W * resize_factor / patch_size) * patch_size)
                elif max_tokens > 0 and nearest_tokens > max_tokens:
                    resize_factor = ((max_tokens * patch_size ** 2) / (nearest_H * nearest_W)) ** 0.5
                    new_H = max(patch_size, int(nearest_H * resize_factor / patch_size) * patch_size)
                    new_W = max(patch_size, int(nearest_W * resize_factor / patch_size) * patch_size)
                else:
                    new_H, new_W = nearest_H, nearest_W
            else:
                # Pixel-based resizing fallback
                max_height = getattr(args, 'eval_depth_max_height', None)
                max_width = getattr(args, 'eval_depth_max_width', None)

                if max_height is not None and max_width is not None:
                    scale = min(max_height / H, max_width / W, 1.0)
                    new_H = int(H * scale)
                    new_W = int(W * scale)
                else:
                    max_res = getattr(args, 'eval_depth_max_resolution', 1024)
                    if max(H, W) > max_res:
                        scale = max_res / max(H, W)
                        new_H = int(H * scale)
                        new_W = int(W * scale)
                    else:
                        new_H, new_W = H, W

                new_H = (new_H // patch_size) * patch_size
                new_W = (new_W // patch_size) * patch_size
                new_H = max(new_H, patch_size)
                new_W = max(new_W, patch_size)

            if (new_H, new_W) != (H, W):
                image = torch.nn.functional.interpolate(
                    image, size=(new_H, new_W), mode='bilinear', align_corners=False
                )
                dataset_name = sample['dataset_name'][0] if isinstance(sample.get('dataset_name'), (list, tuple)) else sample.get('dataset_name', 'unknown')
                actual_tokens = (new_H // patch_size) * (new_W // patch_size)
                if (dataset_name, H, W, new_H, new_W) not in logged_resolutions:
                    logged_resolutions.add((dataset_name, H, W, new_H, new_W))
                    print(f"[Original Res] Resized {dataset_name} input: ({H}, {W}) -> ({new_H}, {new_W}), tokens={actual_tokens}")
            else:
                # No resize needed, but still log the resolution info
                dataset_name = sample['dataset_name'][0] if isinstance(sample.get('dataset_name'), (list, tuple)) else sample.get('dataset_name', 'unknown')
                actual_tokens = (new_H // patch_size) * (new_W // patch_size)
                if (dataset_name, H, W, new_H, new_W) not in logged_resolutions:
                    logged_resolutions.add((dataset_name, H, W, new_H, new_W))
                    print(f"[Original Res] {dataset_name}: ({H}, {W}) -> no resize, tokens={actual_tokens}")

        elif is_wild_images:
            # Wild images arrive at their native resolution and have no GT to match, so the
            # only thing that matters is giving the model an input it generalizes well on.
            # It only ever saw one square resolution during training, so keep the token
            # count close to that budget: resize the whole frame with the aspect ratio
            # preserved to the patch-aligned resolution nearest the budget, predict there,
            # and let the original_size block below resize the prediction back to native.
            # Aspect ratio is preserved rather than forced square, since a squashed image is
            # itself off-distribution.
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
            # needs no resize in either direction. Leaving original_size as None also skips
            # the resize-back below, instead of interpolating to the size it already has.
            if (new_H, new_W) != (H, W):
                original_size = (H, W)
                image = torch.nn.functional.interpolate(
                    image, size=(new_H, new_W), mode='bilinear', align_corners=False
                )

        # generation
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Start inference timing (skip warmup samples)
            if i >= warmup_samples:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_inference_start = time.perf_counter()

            sampled_pointcloud, intermediate_pointcloud = model_without_ddp.generate(image, return_intermediate_steps=True)

            # End inference timing (skip warmup samples)
            if i >= warmup_samples:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_inference_end = time.perf_counter()
                batch_size = image.shape[0]
                total_inference_time += (t_inference_end - t_inference_start)
                inference_sample_count += batch_size

        if args.distributed:
            torch.distributed.barrier()

        sampled_pointcloud = sampled_pointcloud.float()

        # Resize prediction back to original resolution (use 'nearest' for depth/pointcloud)
        if original_size is not None:
            sampled_pointcloud = torch.nn.functional.interpolate(
                sampled_pointcloud, size=original_size, mode='nearest'
            )
            # Also resize intermediate_pointcloud for visualization consistency
            for k in intermediate_pointcloud.keys():
                intermediate_pointcloud[k] = torch.nn.functional.interpolate(
                    intermediate_pointcloud[k].float(), size=original_size, mode='nearest'
                )

        # remove_sky: detect predicted sky on the RAW model output (normalized space),
        # BEFORE the radial-log inverse and scale/shift alignment. Training with
        # --handle_sky places sky pixels at norm ~sky_far_plane_value (default 3.0) in
        # this space; after the inverse transform / affine alignment the value becomes
        # scene-dependent (e.g. norm ~exp(3)-1 in linear space, then metric after
        # alignment), so a fixed threshold is only meaningful here. The per-pixel mask
        # is computed at the final (post-resize) resolution so it stays aligned with the
        # saved point clouds, which are spatially unchanged by the later affine transform.
        pred_sky_mask = None
        if getattr(args, 'remove_sky', False):
            if getattr(args, 'remove_sky_metric', 'norm') == 'depth':
                sky_metric = sampled_pointcloud[:, 2]            # [B, H, W]
            else:
                sky_metric = sampled_pointcloud.norm(dim=1)      # [B, H, W]
            pred_sky_mask = sky_metric > args.remove_sky_threshold  # [B, H, W] bool, True = sky

        # compute metrics (skipped for wild images with no GT)
        raw_sampled_pointcloud = None
        # Initialised per sample: the "<100 valid GT points" early exit below skips the
        # compute_affine_metrics() call that populates this, while the shared visualization
        # path still reads it. Without this, the first such sample raises UnboundLocalError
        # and later ones would splice the previous sample's boundary_viz into this panel.
        align_outputs = {}
        if is_wild_images:
            affine_points_scale = torch.tensor(1.0, device=sampled_pointcloud.device)
            affine_points_shift = torch.zeros(1, 3, 1, 1, device=sampled_pointcloud.device)
        with torch.no_grad():
            if not is_wild_images:
    
                # compute affine metrics
                assert sampled_pointcloud.shape[0] == 1  # only support batch size 1 for now
                
                # Get GT pointcloud based on dataloader type
                if is_eval_depth_loader:
                    # EvalDepthDataset: pointcloud is already unnormalized (raw GT)
                    gt_pointcloud = sample['pointcloud'].to(sampled_pointcloud.device, non_blocking=True)
                    # Use valid_mask from dataloader (handles NaN/Inf)
                    gt_valid_mask = sample['valid_mask'].to(sampled_pointcloud.device, non_blocking=True)
                else:
                    # fall back to the unnormalized point cloud
                    gt_pointcloud = sample['pointcloud_unnormalized'].to(sampled_pointcloud.device, non_blocking=True)
                    # Valid mask is depth > 0
                    gt_valid_mask = gt_pointcloud[:, 2] > 0
                
                # Check if there are enough valid points for metric computation
                num_valid_points = gt_valid_mask.sum().item()
                if num_valid_points < 100:
                    print(f"[Warning] Sample {i} has only {num_valid_points} valid points, skipping metrics computation")
                    # Skip this sample but still save visualization if needed
                    pointcloud = gt_pointcloud
                    # Use dummy values for visualization
                    affine_points_scale = torch.tensor(1.0, device=sampled_pointcloud.device)
                    affine_points_shift = torch.zeros(1, 3, 1, 1, device=sampled_pointcloud.device)
                else:
                    pred = {
                        'points_affine_invariant': sampled_pointcloud.permute(0, 2, 3, 1),  # [B, H, W, 3],
                        'depth_affine_invariant': sampled_pointcloud[:, 2],  # [B, H, W]
                    }
                    gt = {
                        'points': gt_pointcloud.permute(0, 2, 3, 1),  # [B, H, W, 3],
                        'depth': gt_pointcloud[:, 2],  # [B, H, W]
                        'depth_mask': gt_valid_mask,
                    }
                    # Determine if this sample's dataset should compute SI boundary F1
                    dataset_name = sample['dataset_name'][0] if 'dataset_name' in sample and isinstance(sample['dataset_name'], (list, tuple)) else sample.get('dataset_name', '')
                    compute_si_boundary = dataset_name in boundary_datasets
                    compute_depth_range = getattr(args, 'eval_depth_range_metrics', False)
                    affine_metrics, align_outputs = compute_affine_metrics(pred, gt, vis=True, compute_si_boundary=compute_si_boundary, compute_depth_range=compute_depth_range)
    
                    total_affine_rel_depth += affine_metrics['depth_affine_invariant']['rel']
                    total_affine_delta1_depth += affine_metrics['depth_affine_invariant']['delta1']
    
                    total_affine_rel_point += affine_metrics['points_affine_invariant']['rel']
                    total_affine_delta1_point += affine_metrics['points_affine_invariant']['delta1']
    
                    num_valid_samples += 1  # Increment valid sample counter
    
                    # Accumulate SI boundary F1 if computed
                    if 'si_boundary_f1' in affine_metrics:
                        total_si_boundary_f1 += affine_metrics['si_boundary_f1']
                        num_si_boundary_samples += 1
    
                    # Accumulate depth-range metrics
                    if 'depth_range' in affine_metrics:
                        for bin_name, bin_metrics in affine_metrics['depth_range'].items():
                            depth_range_global[bin_name]['total_rel_depth'] += bin_metrics['rel_depth']
                            depth_range_global[bin_name]['total_delta1_depth'] += bin_metrics['delta1_depth']
                            depth_range_global[bin_name]['count'] += 1
                            if 'rel_point' in bin_metrics:
                                depth_range_global[bin_name]['total_rel_point'] += bin_metrics['rel_point']
                                depth_range_global[bin_name]['total_delta1_point'] += bin_metrics['delta1_point']
    
                    # Track per-dataset metrics
                    if 'dataset_name' in sample:
                        # Extract dataset name (handle batch dimension)
                        dataset_name = sample['dataset_name'][0] if isinstance(sample['dataset_name'], (list, tuple)) else sample['dataset_name']
                        per_dataset_metrics[dataset_name]['total_affine_rel_depth'] += affine_metrics['depth_affine_invariant']['rel']
                        per_dataset_metrics[dataset_name]['total_affine_delta1_depth'] += affine_metrics['depth_affine_invariant']['delta1']
                        per_dataset_metrics[dataset_name]['total_affine_rel_point'] += affine_metrics['points_affine_invariant']['rel']
                        per_dataset_metrics[dataset_name]['total_affine_delta1_point'] += affine_metrics['points_affine_invariant']['delta1']
                        per_dataset_metrics[dataset_name]['num_valid_samples'] += 1
                        # Track SI boundary F1 per dataset
                        if 'si_boundary_f1' in affine_metrics:
                            per_dataset_metrics[dataset_name]['total_si_boundary_f1'] += affine_metrics['si_boundary_f1']
                            per_dataset_metrics[dataset_name]['num_si_boundary_samples'] += 1
    
                        # Track depth-range metrics per dataset
                        if 'depth_range' in affine_metrics:
                            for bin_name, bin_metrics in affine_metrics['depth_range'].items():
                                ds_bin = per_dataset_metrics[dataset_name]['depth_range'][bin_name]
                                ds_bin['total_rel_depth'] += bin_metrics['rel_depth']
                                ds_bin['total_delta1_depth'] += bin_metrics['delta1_depth']
                                ds_bin['count'] += 1
                                if 'rel_point' in bin_metrics:
                                    ds_bin['total_rel_point'] += bin_metrics['rel_point']
                                    ds_bin['total_delta1_point'] += bin_metrics['delta1_point']
    
                        # DIODE indoor/outdoor split - additionally track split metrics
                        if getattr(args, 'eval_diode_split_indoor_outdoor', False) and dataset_name == 'DIODE':
                            sample_id = sample['sample_id'][0] if isinstance(sample['sample_id'], (list, tuple)) else sample['sample_id']
                            if 'indoor' in sample_id.lower():
                                split_name = 'DIODE_indoor'
                            elif 'outdoor' in sample_id.lower():
                                split_name = 'DIODE_outdoor'
                            else:
                                split_name = None
    
                            if split_name:
                                per_dataset_metrics[split_name]['total_affine_rel_depth'] += affine_metrics['depth_affine_invariant']['rel']
                                per_dataset_metrics[split_name]['total_affine_delta1_depth'] += affine_metrics['depth_affine_invariant']['delta1']
                                per_dataset_metrics[split_name]['total_affine_rel_point'] += affine_metrics['points_affine_invariant']['rel']
                                per_dataset_metrics[split_name]['total_affine_delta1_point'] += affine_metrics['points_affine_invariant']['delta1']
                                per_dataset_metrics[split_name]['num_valid_samples'] += 1
    
                    # Collect per-sample metrics if enabled
                    if per_sample_metrics is not None and 'sample_id' in sample:
                        sample_id_str = sample['sample_id'][0] if isinstance(sample['sample_id'], (list, tuple)) else sample['sample_id']
                        dataset_name_str = dataset_name if 'dataset_name' in sample else 'default'
                        sample_record = {
                            'sample_id': sample_id_str,
                            'dataset_name': dataset_name_str,
                            'affine_rel_depth': affine_metrics['depth_affine_invariant']['rel'],
                            'affine_delta1_depth': affine_metrics['depth_affine_invariant']['delta1'],
                            'affine_rel_point': affine_metrics['points_affine_invariant']['rel'],
                            'affine_delta1_point': affine_metrics['points_affine_invariant']['delta1'],
                            'si_boundary_f1': affine_metrics.get('si_boundary_f1', None),
                        }
    
                        # Compute boundary F1 for each intermediate diffusion step
                        if (getattr(args, 'eval_intermediate_boundary', False)
                                and intermediate_pointcloud
                                and compute_si_boundary):
                            step_boundary_f1 = {}
                            for step_key in sorted(intermediate_pointcloud.keys()):
                                if step_key == -1:
                                    continue  # skip initial noise
                                step_pc = intermediate_pointcloud[step_key].float()
                                step_pred = {
                                    'points_affine_invariant': step_pc.permute(0, 2, 3, 1),
                                    'depth_affine_invariant': step_pc[:, 2],
                                }
                                step_metrics, _ = compute_affine_metrics(
                                    step_pred, gt, vis=False,
                                    compute_si_boundary=True,
                                    compute_depth_range=False
                                )
                                num_steps_done = step_key + 1
                                step_boundary_f1[f'step_{num_steps_done}'] = step_metrics.get('si_boundary_f1', None)
    
                            # Add final output as the last step
                            total_steps = len([k for k in intermediate_pointcloud if k != -1]) + 1
                            step_boundary_f1[f'step_{total_steps}'] = affine_metrics.get('si_boundary_f1', None)
                            sample_record['si_boundary_f1_by_step'] = step_boundary_f1
    
                        per_sample_metrics.append(sample_record)
    
                    # Save raw (unaligned) prediction for optional visualization
                    if getattr(args, 'viz_unaligned_depth', False):
                        raw_sampled_pointcloud = sampled_pointcloud.detach().clone()

                    # update viz
                    sampled_pointcloud = align_outputs['pred_points'].permute(0, 3, 1, 2)  # [B, 3, H, W]
                    sampled_pointcloud[:, 2][sampled_pointcloud[:, 2] < 0.] = 0.
    
                    affine_points_scale = align_outputs['affine_points_scale']  # [1]
                    affine_points_shift = align_outputs['affine_points_shift'].reshape(1, 3, 1, 1)  # [1, 3, 1, 1]
    
                # unnormalize gt pointcloud for visualization
                pointcloud = gt_pointcloud

        # distributed save depths
        for b_id in range(sampled_pointcloud.size(0)):
            img_id = i * sampled_pointcloud.size(0) * world_size + local_rank * sampled_pointcloud.size(0) + b_id

            # Determine save directory (per-dataset subdirectory for multi-dataset evaluation).
            # For wild images the name comes from --eval_wild_images_name.
            if 'dataset_name' in sample:
                dataset_name = sample['dataset_name'][b_id] if isinstance(sample['dataset_name'], (list, tuple)) else sample['dataset_name']
                dataset_save_folder = os.path.join(save_folder, dataset_name)
                # Create dataset-specific subdirectory if it doesn't exist
                if not args.online_eval and not args.eval_no_save_gen:
                    if accelerator is not None:
                        if accelerator.is_main_process:
                            os.makedirs(dataset_save_folder, exist_ok=True)
                    else:
                        if misc.get_rank() == 0:
                            os.makedirs(dataset_save_folder, exist_ok=True)
            else:
                # Single dataset or no dataset_name - save to root folder
                dataset_save_folder = save_folder

            # Use sample_id for filename if available (EvalDepthDataset or WildImagesDataset)
            if 'sample_id' in sample:
                sample_id_raw = sample['sample_id'][b_id] if isinstance(sample['sample_id'], list) else sample['sample_id']
                if args.eval_wild_images:
                    # Mirror the input filename, so sample0.png yields sample0_depth.png and
                    # sample0_pointcloud.ply. No index prefix, and the extension is dropped so
                    # it is not doubled. Any subdirectory the image sat in is preserved, which
                    # also stops same-named images in different folders overwriting each other.
                    file_prefix = os.path.splitext(sample_id_raw)[0]
                    sub = os.path.dirname(file_prefix)
                    if sub and not args.online_eval and not args.eval_no_save_gen:
                        os.makedirs(os.path.join(dataset_save_folder, sub), exist_ok=True)
                else:
                    # Replace / with _ for valid filename
                    file_prefix = f"{img_id:05d}_{sample_id_raw.replace('/', '_')}"
            else:
                file_prefix = f'{img_id:05d}'

            # remove_sky: per-sample keep-mask (True = keep, False = sky to drop), derived
            # from the raw-output sky detection above. Used both to hide sky in the depth
            # viz and to filter sky out of the saved point clouds.
            sky_keep_mask = None
            if pred_sky_mask is not None:
                sky_keep_mask = (~pred_sky_mask[b_id]).detach().cpu().numpy()  # (H, W) bool

            # depth_gt (only available when GT exists)
            if not is_wild_images:
                depth_gt = pointcloud[b_id, 2].detach().cpu()  # [H, W]
                if is_eval_depth_loader:
                    depth_gt_for_viz = depth_gt.clone()
                    depth_gt_for_viz[~torch.isfinite(depth_gt_for_viz)] = 0
                    depth_gt_viz = viz_depth_tensor(depth_gt_for_viz, shifted_depth=False, colormap=args.depth_colormap)  # [H, W, 3] np
                else:
                    depth_gt_viz = viz_depth_tensor(depth_gt, shifted_depth=False, colormap=args.depth_colormap)  # [H, W, 3] np

            # depth from z
            # For wild images the model output is zero-centered (center_shift_point + normalize_point_by_mean)
            # with no affine alignment, so we must not clamp to 0 and must use shifted_depth=True
            if is_wild_images:
                sampled_depth = sampled_pointcloud[b_id, 2]  # [H, W], may be negative
                sampled_depth_viz = sampled_depth
                # remove_sky: push predicted sky to far so it drops out of the depth viz
                if pred_sky_mask is not None:
                    sampled_depth_viz = torch.where(pred_sky_mask[b_id], torch.full_like(sampled_depth_viz, 1e6), sampled_depth_viz)
                depth_viz = viz_depth_tensor(sampled_depth_viz, shifted_depth=True, colormap=args.depth_colormap)  # [H, W, 3] np
            else:
                sampled_depth = sampled_pointcloud[b_id, 2].clamp(min=0)  # [H, W]
                sampled_depth_viz = sampled_depth
                # remove_sky: treat predicted sky as far/invalid in the depth viz
                if pred_sky_mask is not None:
                    sampled_depth_viz = torch.where(pred_sky_mask[b_id], torch.full_like(sampled_depth_viz, 1e6), sampled_depth_viz)
                depth_viz = viz_depth_tensor(sampled_depth_viz, shifted_depth=False, colormap=args.depth_colormap)  # [H, W, 3] np

            img_viz = (image_01[b_id].permute(1, 2, 0).detach().cpu() * 255.).to(torch.uint8).numpy()

            # remove_sky: build an [input | sky-mask | overlay] panel so the threshold can
            # be checked visually. mask is white where sky (removed); overlay tints sky red.
            skymask_concat = None
            if pred_sky_mask is not None:
                sky_b = pred_sky_mask[b_id].detach().cpu().numpy()  # (H, W) bool, True = sky
                mask_viz = np.repeat((sky_b[:, :, None].astype(np.uint8) * 255), 3, axis=2)  # (H, W, 3)
                overlay = img_viz.copy()
                overlay[sky_b] = (0.5 * overlay[sky_b].astype(np.float32)
                                  + 0.5 * np.array([255, 0, 0], dtype=np.float32)).astype(np.uint8)
                skymask_concat = np.concatenate([img_viz, mask_viz, overlay], axis=1)

            concat_list = [img_viz]

            # save the intermediate generation results
            for k, v in intermediate_pointcloud.items():
                # to the original scale
                v = v * affine_points_scale + affine_points_shift  # [B, 3, H, W]
                if is_wild_images:
                    depth_k = v[b_id, 2].detach().cpu()  # [H, W], keep raw (zero-centered)
                    depth_k_viz = viz_depth_tensor(depth_k, shifted_depth=True, colormap=args.depth_colormap)
                else:
                    depth_k = v[b_id, 2].detach().cpu().clamp(min=0)  # [H, W]
                    depth_k_viz = viz_depth_tensor(depth_k, shifted_depth=False, colormap=args.depth_colormap)
                if not is_wild_images:
                    concat_list.append(depth_k_viz)

            # final prediction (and GT + error map if available)
            concat_list.append(depth_viz)
            if not is_wild_images and raw_sampled_pointcloud is not None:
                unaligned_depth = raw_sampled_pointcloud[b_id, 2]  # [H, W], zero-centered
                unaligned_depth_viz = viz_depth_tensor(unaligned_depth, shifted_depth=True, colormap=args.depth_colormap)
                concat_list.append(unaligned_depth_viz)
            if not is_wild_images:
                concat_list.append(depth_gt_viz)

                # also visualize point error map
                point_pred_error = torch.abs(sampled_pointcloud[b_id] - pointcloud[b_id]).mean(0)  # [H, W]
                # For EvalDepthDataset, mask out NaN/Inf before computing min/max
                if is_eval_depth_loader:
                    valid_error_mask = gt_valid_mask[b_id]
                    point_pred_error_valid = point_pred_error.clone()
                    point_pred_error_valid[~valid_error_mask] = 0
                    valid_errors = point_pred_error[valid_error_mask]
                    if valid_errors.numel() > 0:
                        error_min, error_max = valid_errors.min(), valid_errors.max()
                    else:
                        error_min, error_max = 0, 1
                    error_normalized = (point_pred_error_valid - error_min) / (error_max - error_min + 1e-8)
                    error_normalized[~valid_error_mask] = 0.
                else:
                    error_normalized = (point_pred_error - point_pred_error.min()) / (point_pred_error.max() - point_pred_error.min() + 1e-8)
                    error_normalized[pointcloud[b_id, 2] <= 0] = 0.
                error_normalized = (error_normalized * 255.).to(torch.uint8).detach().cpu()
                error_normalized = error_normalized[:, :, None].repeat(1, 1, 3).numpy()  # [H, W, 3]
                concat_list.append(error_normalized)

            # Add boundary visualization if computed
            if 'boundary_viz' in align_outputs:
                concat_list.append(align_outputs['boundary_viz'])

            # concat all the results
            concat = np.concatenate(concat_list, axis=1)

            # save
            if not args.online_eval and not args.eval_no_save_gen:
                save_path = os.path.join(dataset_save_folder, f'{file_prefix}_depth.png')
                Image.fromarray(concat).save(save_path)

                # remove_sky: save the [input | mask | overlay] panel; threshold is in the
                # filename so multiple --remove_sky_threshold runs can be compared side by side.
                # Skipped for demo runs, which want only the depth map and the point cloud.
                if skymask_concat is not None and not args.eval_wild_images:
                    skymask_path = os.path.join(
                        dataset_save_folder,
                        f'{file_prefix}_skymask_th{args.remove_sky_threshold:.2f}.png')
                    Image.fromarray(skymask_concat).save(skymask_path)

            # save raw numpy arrays for cross-run comparison
            if getattr(args, 'save_raw_depth_npy', False) and not args.online_eval and not args.eval_no_save_gen:
                # Save predicted depth (after affine alignment)
                pred_depth_path = os.path.join(dataset_save_folder, f'{file_prefix}_depth_pred.npy')
                np.save(pred_depth_path, sampled_depth.detach().cpu().numpy())
                # Save predicted pointcloud [3, H, W]
                pred_pc_path = os.path.join(dataset_save_folder, f'{file_prefix}_pointcloud_pred.npy')
                np.save(pred_pc_path, sampled_pointcloud[b_id].detach().cpu().numpy())
                # Save input image for visualization
                input_img_path = os.path.join(dataset_save_folder, f'{file_prefix}_input.png')
                Image.fromarray(img_viz).save(input_img_path)

            # log to wandb
            if args.online_eval and b_id == 0:
                should_log = False
                viz_tag = None

                if is_multi_dataset_viz:
                    # Multiple datasets: log 2 samples per dataset
                    if 'dataset_name' in sample:
                        dataset_name = sample['dataset_name'][b_id] if isinstance(sample['dataset_name'], (list, tuple)) else sample['dataset_name']
                        if dataset_name not in per_dataset_viz_counters:
                            per_dataset_viz_counters[dataset_name] = 0

                        counter = per_dataset_viz_counters[dataset_name]
                        if counter < samples_per_dataset:
                            should_log = True
                            viz_tag = f'test_viz/{dataset_name}_sample{counter}'
                            per_dataset_viz_counters[dataset_name] += 1
                else:
                    # Single dataset: log 4 samples total
                    if i % log_eval_freq == 0 and eval_counter < num_val_samples:
                        should_log = True
                        viz_tag = f'test_viz/sample{eval_counter}'
                        eval_counter += 1

                if should_log and viz_tag is not None:
                    is_main = accelerator.is_main_process if accelerator is not None else (misc.get_rank() == 0)
                    if is_main:
                        val_logs[viz_tag] = wandb.Image(concat)
                        # remove_sky: also log the [input | mask | overlay] panel
                        if skymask_concat is not None:
                            val_logs[f'{viz_tag}_skymask'] = wandb.Image(skymask_concat)

            # save point cloud
            save_path_pointcloud = os.path.join(dataset_save_folder, f'{file_prefix}_pointcloud.ply')

            # save generated and gt point cloud
            if not args.online_eval and not args.eval_no_save_gen:
                # For EvalDepthDataset, filter out NaN/Inf points before saving
                if is_eval_depth_loader:
                    valid_mask_np = gt_valid_mask[b_id].detach().cpu().numpy()
                    
                    # Get point clouds and colors
                    sampled_pc_np = sampled_pointcloud[b_id].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
                    gt_pc_np = pointcloud[b_id].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
                    img_np = image_01[b_id].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
                    
                    # Set invalid points to 0 (or filter them out)
                    sampled_pc_clean = sampled_pc_np.copy()
                    gt_pc_clean = gt_pc_np.copy()
                    # sampled_pc_clean[~valid_mask_np] = 0
                    gt_pc_clean[~valid_mask_np] = 0
                    
                    de_mask = None
                    gt_de_mask = None
                    if getattr(args, 'remove_depth_edge', False):
                        de_mask = ~depth_flying_points(sampled_pc_clean[:, :, 2], rtol=args.depth_edge_rtol)
                        gt_de_mask = ~depth_flying_points(gt_pc_clean[:, :, 2], rtol=args.depth_edge_rtol)
                    # remove_sky: drop predicted sky points (prediction only, not GT)
                    de_mask = combine_keep_masks(de_mask, sky_keep_mask)
                    save_generated_gt_point_cloud(
                        sampled_pc_clean,
                        img_np,
                        gt_pc_clean,
                        img_np,
                        filename=save_path_pointcloud,
                        mask=de_mask,
                        gt_mask=gt_de_mask,
                    )
                else:
                    if is_wild_images:
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
                    else:
                        save_generated_gt_point_cloud(sampled_pointcloud[b_id].permute(1, 2, 0).detach().cpu().numpy(),
                            image_01[b_id].permute(1, 2, 0).detach().cpu().numpy(),
                            pointcloud[b_id].permute(1, 2, 0).detach().cpu().numpy(),
                            image_01[b_id].permute(1, 2, 0).detach().cpu().numpy(),
                            filename=save_path_pointcloud,
                            mask=sky_keep_mask,
                        )

            # save intermediate generated point cloud separately if flag is enabled
            if args.save_separate_intermediate_plys and not args.online_eval and not args.eval_no_save_gen:
                # Get base filename without extension
                base_path = save_path_pointcloud.replace('.ply', '')

                # Prepare point clouds and colors
                img_np = image_01[b_id].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)

                # Handle EvalDepthDataset with valid masks
                if is_eval_depth_loader:
                    valid_mask_np = gt_valid_mask[b_id].detach().cpu().numpy()
                    sampled_pc_np = sampled_pointcloud[b_id].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
                    gt_pc_np = pointcloud[b_id].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)

                    # Set invalid points to 0
                    sampled_pc_clean = sampled_pc_np.copy()
                    gt_pc_clean = gt_pc_np.copy()
                    # sampled_pc_clean[~valid_mask_np] = 0
                    gt_pc_clean[~valid_mask_np] = 0
                elif is_wild_images:
                    sampled_pc_clean = sampled_pointcloud[b_id].permute(1, 2, 0).detach().cpu().numpy()
                    gt_pc_clean = None
                else:
                    sampled_pc_clean = sampled_pointcloud[b_id].permute(1, 2, 0).detach().cpu().numpy()
                    gt_pc_clean = pointcloud[b_id].permute(1, 2, 0).detach().cpu().numpy()

                # Compute depth edge masks
                de_mask = None
                gt_de_mask = None
                if getattr(args, 'remove_depth_edge', False):
                    de_mask = ~depth_flying_points(sampled_pc_clean[:, :, 2], rtol=args.depth_edge_rtol)
                    if gt_pc_clean is not None:
                        gt_de_mask = ~depth_flying_points(gt_pc_clean[:, :, 2], rtol=args.depth_edge_rtol)
                # remove_sky: drop predicted sky points (applied to pred + intermediate steps,
                # which share the same sky pixels; GT is left untouched)
                de_mask = combine_keep_masks(de_mask, sky_keep_mask)

                # Save predicted point cloud
                pred_path = f'{base_path}_pred.ply'
                save_single_point_cloud(sampled_pc_clean, img_np, pred_path, transform_to_gl=True, mask=de_mask)

                # Save GT point cloud (only when GT exists)
                if gt_pc_clean is not None:
                    gt_path = f'{base_path}_gt.ply'
                    save_single_point_cloud(gt_pc_clean, img_np, gt_path, transform_to_gl=True, mask=gt_de_mask)

                # Save intermediate diffusion steps
                if intermediate_pointcloud:
                    for k, v in intermediate_pointcloud.items():
                        # Convert to original scale
                        v_scaled = v * affine_points_scale + affine_points_shift  # [B, 3, H, W]
                        point_k = v_scaled[b_id].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)

                        # Handle valid masks for EvalDepthDataset
                        if is_eval_depth_loader:
                            point_k_clean = point_k.copy()
                            # point_k_clean[~valid_mask_np] = 0
                        else:
                            point_k_clean = point_k

                        # Save with appropriate naming
                        if k == -1:
                            # Initial random noise
                            step_path = f'{base_path}_noise_init.ply'
                        else:
                            # Regular diffusion step
                            step_path = f'{base_path}_step{k}.ply'
                        save_single_point_cloud(point_k_clean, img_np, step_path, transform_to_gl=True, mask=de_mask)

            # save intermediate depth maps separately if flag is enabled
            if args.save_separate_intermediate_depths and not args.online_eval and not args.eval_no_save_gen:
                # Get base filename without extension
                base_path = save_path_pointcloud.replace('.ply', '')

                # Save input image
                img_h, img_w = image_01.shape[-2], image_01.shape[-1]
                res_suffix = f'_{img_h}x{img_w}' if getattr(args, 'save_resolution_in_filename', False) else ''
                input_img_path = f'{base_path}_input{res_suffix}.png'
                Image.fromarray((image_01[b_id].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save(input_img_path)

                # Save predicted depth
                pred_depth_path = f'{base_path}_depth_pred.png'
                Image.fromarray(depth_viz).save(pred_depth_path)

                # Save GT depth (only when GT exists)
                if not is_wild_images:
                    gt_depth_path = f'{base_path}_depth_gt.png'
                    Image.fromarray(depth_gt_viz).save(gt_depth_path)

                # Save intermediate diffusion steps
                if intermediate_pointcloud:
                    for k, v in intermediate_pointcloud.items():
                        # Convert to original scale
                        v_scaled = v * affine_points_scale + affine_points_shift  # [B, 3, H, W]
                        depth_k = v_scaled[b_id, 2].detach().cpu().clamp(min=0)  # [H, W]
                        depth_k_viz = viz_depth_tensor(depth_k, shifted_depth=False, colormap=args.depth_colormap)  # [H, W, 3] np

                        # Save with appropriate naming
                        if k == -1:
                            # Initial random noise
                            step_path = f'{base_path}_depth_noise_init.png'
                        else:
                            # Regular diffusion step
                            step_path = f'{base_path}_depth_step{k}.png'
                        Image.fromarray(depth_k_viz).save(step_path)

        # TODO: compute depth and pointcloud metrics

    # Save per-sample metrics to JSON (if enabled)
    if per_sample_metrics is not None:
        import json
        if accelerator is not None:
            # Gather all per-sample metrics from all processes
            gathered = accelerator.gather_for_metrics(per_sample_metrics)
            if accelerator.is_main_process:
                # Flatten the gathered list (it may be nested)
                all_per_sample_metrics = gathered if isinstance(gathered, list) else list(gathered)
                metrics_path = os.path.join(save_folder, 'per_sample_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(all_per_sample_metrics, f, indent=2)
                print(f"Saved per-sample metrics to: {metrics_path}")
        elif args.distributed:
            # For torch.distributed, gather to all processes then save on rank 0
            gathered = [None] * world_size
            dist.all_gather_object(gathered, per_sample_metrics)
            if misc.get_rank() == 0:
                all_per_sample_metrics = [m for sublist in gathered for m in sublist]
                metrics_path = os.path.join(save_folder, 'per_sample_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(all_per_sample_metrics, f, indent=2)
                print(f"Saved per-sample metrics to: {metrics_path}")
        else:
            # Single GPU
            metrics_path = os.path.join(save_folder, 'per_sample_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(per_sample_metrics, f, indent=2)
            print(f"Saved per-sample metrics to: {metrics_path}")

    # Aggregate per_dataset_metrics across all processes
    if accelerator is not None and per_dataset_metrics:
        # Aggregate each local dataset's metrics across all processes
        # No need to gather dataset names - just aggregate what each process has seen
        # accelerator.reduce() will sum metrics from all processes
        aggregated_per_dataset_metrics = {}

        for dataset_name, local_metrics in per_dataset_metrics.items():
            stats = [
                local_metrics['total_affine_rel_depth'],
                local_metrics['total_affine_delta1_depth'],
                local_metrics['total_affine_rel_point'],
                local_metrics['total_affine_delta1_point'],
                float(local_metrics['num_valid_samples']),
                local_metrics['total_si_boundary_f1'],
                float(local_metrics['num_si_boundary_samples']),
            ]
            # Add depth-range stats: 5 values per bin (rel_depth, delta1_depth, rel_point, delta1_point, count)
            for b in depth_range_bins:
                dr = local_metrics['depth_range'][b]
                stats.extend([dr['total_rel_depth'], dr['total_delta1_depth'], dr['total_rel_point'], dr['total_delta1_point'], float(dr['count'])])
            stats_tensor = torch.tensor(stats, dtype=torch.float32, device=accelerator.device)
            global_stats = accelerator.reduce(stats_tensor, reduction="sum")

            # Store aggregated metrics
            aggregated_per_dataset_metrics[dataset_name] = {
                'total_affine_rel_depth': global_stats[0].item(),
                'total_affine_delta1_depth': global_stats[1].item(),
                'total_affine_rel_point': global_stats[2].item(),
                'total_affine_delta1_point': global_stats[3].item(),
                'num_valid_samples': int(global_stats[4].item()),
                'total_si_boundary_f1': global_stats[5].item(),
                'num_si_boundary_samples': int(global_stats[6].item()),
                'depth_range': {},
            }
            for i, b in enumerate(depth_range_bins):
                base = 7 + i * 5
                aggregated_per_dataset_metrics[dataset_name]['depth_range'][b] = {
                    'total_rel_depth': global_stats[base].item(),
                    'total_delta1_depth': global_stats[base + 1].item(),
                    'total_rel_point': global_stats[base + 2].item(),
                    'total_delta1_point': global_stats[base + 3].item(),
                    'count': int(global_stats[base + 4].item()),
                }

        # Replace local per_dataset_metrics with aggregated version
        per_dataset_metrics = aggregated_per_dataset_metrics

    elif args.distributed and per_dataset_metrics:
        # Fallback to torch.distributed for aggregation
        # Aggregate each local dataset's metrics across all processes
        aggregated_per_dataset_metrics = {}

        for dataset_name, local_metrics in per_dataset_metrics.items():
            stats = [
                local_metrics['total_affine_rel_depth'],
                local_metrics['total_affine_delta1_depth'],
                local_metrics['total_affine_rel_point'],
                local_metrics['total_affine_delta1_point'],
                float(local_metrics['num_valid_samples']),
                local_metrics['total_si_boundary_f1'],
                float(local_metrics['num_si_boundary_samples']),
            ]
            for b in depth_range_bins:
                dr = local_metrics['depth_range'][b]
                stats.extend([dr['total_rel_depth'], dr['total_delta1_depth'], dr['total_rel_point'], dr['total_delta1_point'], float(dr['count'])])
            stats_tensor = torch.tensor(stats, dtype=torch.float32, device='cuda')

            # Barrier to ensure all processes are ready
            dist.barrier()

            # All-Reduce (Sum) across all GPUs
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

            # Store aggregated metrics
            aggregated_per_dataset_metrics[dataset_name] = {
                'total_affine_rel_depth': stats_tensor[0].item(),
                'total_affine_delta1_depth': stats_tensor[1].item(),
                'total_affine_rel_point': stats_tensor[2].item(),
                'total_affine_delta1_point': stats_tensor[3].item(),
                'num_valid_samples': int(stats_tensor[4].item()),
                'total_si_boundary_f1': stats_tensor[5].item(),
                'num_si_boundary_samples': int(stats_tensor[6].item()),
                'depth_range': {},
            }
            for i, b in enumerate(depth_range_bins):
                base = 7 + i * 5
                aggregated_per_dataset_metrics[dataset_name]['depth_range'][b] = {
                    'total_rel_depth': stats_tensor[base].item(),
                    'total_delta1_depth': stats_tensor[base + 1].item(),
                    'total_rel_point': stats_tensor[base + 2].item(),
                    'total_delta1_point': stats_tensor[base + 3].item(),
                    'count': int(stats_tensor[base + 4].item()),
                }

        # Replace local per_dataset_metrics with aggregated version
        per_dataset_metrics = aggregated_per_dataset_metrics

    # Debug: Print aggregated per-dataset sample counts
    if per_dataset_metrics:
        total_per_dataset_samples = sum(m['num_valid_samples'] for m in per_dataset_metrics.values())
        if accelerator is not None:
            if accelerator.is_main_process:
                print(f"[Aggregated] Per-dataset samples: {dict((k, v['num_valid_samples']) for k, v in per_dataset_metrics.items())}")
                print(f"[Aggregated] Total per-dataset samples: {total_per_dataset_samples}")
        elif misc.get_rank() == 0:
            print(f"[Aggregated] Per-dataset samples: {dict((k, v['num_valid_samples']) for k, v in per_dataset_metrics.items())}")
            print(f"[Aggregated] Total per-dataset samples: {total_per_dataset_samples}")

    if accelerator is not None:
        # 1. Wrap your values in a tensor (include num_valid_samples)
        stats_tensor = torch.tensor([
            total_affine_rel_depth,
            total_affine_delta1_depth,
            total_affine_rel_point,
            total_affine_delta1_point,
            float(num_valid_samples),  # Add sample count
            total_si_boundary_f1,
            float(num_si_boundary_samples),
        ], dtype=torch.float32, device=accelerator.device)

        # 2. Reduce across all processes (default op is "sum")
        # This replaces the barrier and the manual all_reduce
        reduced_stats = accelerator.reduce(stats_tensor, reduction="sum")

        # 3. Extract values (sum across all processes)
        global_affine_rel_depth_sum = reduced_stats[0].item()
        global_affine_delta1_depth_sum = reduced_stats[1].item()
        global_affine_rel_point_sum = reduced_stats[2].item()
        global_affine_delta1_point_sum = reduced_stats[3].item()
        global_num_valid_samples = int(reduced_stats[4].item())
        global_si_boundary_f1_sum = reduced_stats[5].item()
        global_num_si_boundary_samples = int(reduced_stats[6].item())

    elif args.distributed:
        # 1. Create a tensor to hold the data (include num_valid_samples)
        stats_tensor = torch.tensor([total_affine_rel_depth, total_affine_delta1_depth, total_affine_rel_point, total_affine_delta1_point,
                                     float(num_valid_samples),
                                     total_si_boundary_f1, float(num_si_boundary_samples)],
                                     dtype=torch.float32, device='cuda')

        # 2. Barrier to ensure all processes are ready
        torch.distributed.barrier()

        # 3. All-Reduce (Sum) across all GPUs
        # This sums the error and the sample count from Rank 0, Rank 1, etc.
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        # 4. Extract aggregated values (sum across all processes)
        global_affine_rel_depth_sum = stats_tensor[0].item()
        global_affine_delta1_depth_sum = stats_tensor[1].item()
        global_affine_rel_point_sum = stats_tensor[2].item()
        global_affine_delta1_point_sum = stats_tensor[3].item()
        global_num_valid_samples = int(stats_tensor[4].item())
        global_si_boundary_f1_sum = stats_tensor[5].item()
        global_num_si_boundary_samples = int(stats_tensor[6].item())
    else:
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

    # mean metrics
    # Use num_valid_samples instead of len(data_loader) to handle skipped samples
    num_samples_for_avg = max(1, global_num_valid_samples)  # Avoid division by zero

    avg_affine_rel_depth = (global_affine_rel_depth_sum / num_samples_for_avg) * 100.
    avg_affine_delta1_depth = (global_affine_delta1_depth_sum / num_samples_for_avg) * 100.
    avg_affine_rel_point = (global_affine_rel_point_sum / num_samples_for_avg) * 100.
    avg_affine_delta1_point = (global_affine_delta1_point_sum / num_samples_for_avg) * 100.

    # Wild images carry no ground truth, so every average above is 0 by construction. They are
    # still computed because the tensorboard/wandb logging below reads them, but printing zeros
    # invites reading them as scores.
    if not args.eval_wild_images:
        print(f"Valid samples for metrics: {global_num_valid_samples} / {len(data_loader)}")
        print(f"affine_real_point: {avg_affine_rel_point:.3f}, affine_delta1_point: {avg_affine_delta1_point:.3f}")
        print(f"affine_real_depth: {avg_affine_rel_depth:.3f}, affine_delta1_depth: {avg_affine_delta1_depth:.3f}")

    # Global depth-range metrics (aggregate from per-dataset if distributed, else use local accumulators)
    if getattr(args, 'eval_depth_range_metrics', False):
        # Recompute global depth-range from per-dataset aggregated data if available
        if per_dataset_metrics:
            for b in depth_range_bins:
                depth_range_global[b] = {'total_rel_depth': 0., 'total_delta1_depth': 0., 'total_rel_point': 0., 'total_delta1_point': 0., 'count': 0}
                for ds_metrics in per_dataset_metrics.values():
                    if 'depth_range' in ds_metrics:
                        dr = ds_metrics['depth_range'][b]
                        for k in ['total_rel_depth', 'total_delta1_depth', 'total_rel_point', 'total_delta1_point', 'count']:
                            depth_range_global[b][k] += dr[k]
        has_global_dr = any(depth_range_global[b]['count'] > 0 for b in depth_range_bins)
        if has_global_dr:
            print("\n=== Global Depth-Range Metrics ===")
            for b in depth_range_bins:
                dr = depth_range_global[b]
                if dr['count'] > 0:
                    n_dr = dr['count']
                    print(f"  [{b}] rel_depth: {dr['total_rel_depth']/n_dr*100:.3f}, "
                          f"delta1_depth: {dr['total_delta1_depth']/n_dr*100:.3f}, "
                          f"rel_point: {dr['total_rel_point']/n_dr*100:.3f}, "
                          f"delta1_point: {dr['total_delta1_point']/n_dr*100:.3f} (n={n_dr})")
            print("=" * 50)

    # SI Boundary F1 metric
    if global_num_si_boundary_samples > 0:
        avg_si_boundary_f1 = (global_si_boundary_f1_sum / global_num_si_boundary_samples) * 100.
        print(f"si_boundary_f1: {avg_si_boundary_f1:.3f}")
    else:
        avg_si_boundary_f1 = 0.

    # Inference speed metrics
    if inference_sample_count > 0:
        avg_time_per_image = total_inference_time / inference_sample_count
        fps = inference_sample_count / total_inference_time
        print(f"Inference speed: {avg_time_per_image*1000:.2f} ms/image, {fps:.2f} FPS (total: {inference_sample_count} images in {total_inference_time:.2f}s)")

    if args.eval_wild_images:
        # No metrics to report, so close the run with something unambiguous instead.
        print("Done!")

    # Compute and print per-dataset metrics
    per_dataset_avg_metrics = {}
    if per_dataset_metrics:
        print("\n=== Per-Dataset Metrics ===")
        for dataset_name, metrics in per_dataset_metrics.items():
            num_samples = max(1, metrics['num_valid_samples'])
            avg_metrics = {
                'affine_rel_depth': (metrics['total_affine_rel_depth'] / num_samples) * 100.,
                'affine_delta1_depth': (metrics['total_affine_delta1_depth'] / num_samples) * 100.,
                'affine_rel_point': (metrics['total_affine_rel_point'] / num_samples) * 100.,
                'affine_delta1_point': (metrics['total_affine_delta1_point'] / num_samples) * 100.,
                'num_samples': num_samples,
            }
            # Add SI boundary F1 if computed for this dataset
            if metrics['num_si_boundary_samples'] > 0:
                avg_metrics['si_boundary_f1'] = (metrics['total_si_boundary_f1'] / metrics['num_si_boundary_samples']) * 100.
            per_dataset_avg_metrics[dataset_name] = avg_metrics

            print(f"\n[{dataset_name}] (n={num_samples})")
            print(f"  affine_rel_point: {avg_metrics['affine_rel_point']:.3f}, "
                  f"affine_delta1_point: {avg_metrics['affine_delta1_point']:.3f}")
            print(f"  affine_rel_depth: {avg_metrics['affine_rel_depth']:.3f}, "
                  f"affine_delta1_depth: {avg_metrics['affine_delta1_depth']:.3f}")
            if 'si_boundary_f1' in avg_metrics:
                print(f"  si_boundary_f1: {avg_metrics['si_boundary_f1']:.3f}")

            # Print depth-range breakdown for this dataset
            if 'depth_range' in metrics:
                has_depth_range = any(metrics['depth_range'][b]['count'] > 0 for b in depth_range_bins)
                if has_depth_range:
                    print(f"  Depth-range breakdown:")
                    for b in depth_range_bins:
                        dr = metrics['depth_range'][b]
                        if dr['count'] > 0:
                            n_dr = dr['count']
                            print(f"    [{b}] rel_depth: {dr['total_rel_depth']/n_dr*100:.3f}, "
                                  f"delta1_depth: {dr['total_delta1_depth']/n_dr*100:.3f}, "
                                  f"rel_point: {dr['total_rel_point']/n_dr*100:.3f}, "
                                  f"delta1_point: {dr['total_delta1_point']/n_dr*100:.3f} (n={n_dr})")
        print("=" * 50)

    # log with tensorboard
    if accelerator is not None:
        if accelerator.is_main_process and log_writer is not None:

            log_writer.add_scalar('test/affine_rel_depth', avg_affine_rel_depth, epoch)
            log_writer.add_scalar('test/affine_delta1_depth', avg_affine_delta1_depth, epoch)
            log_writer.add_scalar('test/affine_rel_point', avg_affine_rel_point, epoch)
            log_writer.add_scalar('test/affine_delta1_point', avg_affine_delta1_point, epoch)
            if global_num_si_boundary_samples > 0:
                log_writer.add_scalar('test/si_boundary_f1', avg_si_boundary_f1, epoch)

            # Add per-dataset metrics to tensorboard
            for dataset_name, avg_metrics in per_dataset_avg_metrics.items():
                log_writer.add_scalar(f'test/{dataset_name}/affine_rel_depth', avg_metrics['affine_rel_depth'], epoch)
                log_writer.add_scalar(f'test/{dataset_name}/affine_delta1_depth', avg_metrics['affine_delta1_depth'], epoch)
                log_writer.add_scalar(f'test/{dataset_name}/affine_rel_point', avg_metrics['affine_rel_point'], epoch)
                log_writer.add_scalar(f'test/{dataset_name}/affine_delta1_point', avg_metrics['affine_delta1_point'], epoch)
                if 'si_boundary_f1' in avg_metrics:
                    log_writer.add_scalar(f'test/{dataset_name}/si_boundary_f1', avg_metrics['si_boundary_f1'], epoch)
    else:
        if misc.get_rank() == 0 and log_writer is not None:

            log_writer.add_scalar('test/affine_rel_depth', avg_affine_rel_depth, epoch)
            log_writer.add_scalar('test/affine_delta1_depth', avg_affine_delta1_depth, epoch)
            log_writer.add_scalar('test/affine_rel_point', avg_affine_rel_point, epoch)
            log_writer.add_scalar('test/affine_delta1_point', avg_affine_delta1_point, epoch)
            if global_num_si_boundary_samples > 0:
                log_writer.add_scalar('test/si_boundary_f1', avg_si_boundary_f1, epoch)

            # Add per-dataset metrics to tensorboard
            for dataset_name, avg_metrics in per_dataset_avg_metrics.items():
                log_writer.add_scalar(f'test/{dataset_name}/affine_rel_depth', avg_metrics['affine_rel_depth'], epoch)
                log_writer.add_scalar(f'test/{dataset_name}/affine_delta1_depth', avg_metrics['affine_delta1_depth'], epoch)
                log_writer.add_scalar(f'test/{dataset_name}/affine_rel_point', avg_metrics['affine_rel_point'], epoch)
                log_writer.add_scalar(f'test/{dataset_name}/affine_delta1_point', avg_metrics['affine_delta1_point'], epoch)
                if 'si_boundary_f1' in avg_metrics:
                    log_writer.add_scalar(f'test/{dataset_name}/si_boundary_f1', avg_metrics['si_boundary_f1'], epoch)

    # log with wandb
    if accelerator is not None:
        if accelerator.is_main_process and args.wandb:

            val_logs['test_metric/affine_rel_depth'] = avg_affine_rel_depth
            val_logs['test_metric/affine_delta1_depth'] = avg_affine_delta1_depth
            val_logs['test_metric/affine_rel_point'] = avg_affine_rel_point
            val_logs['test_metric/affine_delta1_point'] = avg_affine_delta1_point

            # Add per-dataset metrics to wandb
            for dataset_name, avg_metrics in per_dataset_avg_metrics.items():
                val_logs[f'test_metric_{dataset_name}/affine_rel_depth'] = avg_metrics['affine_rel_depth']
                val_logs[f'test_metric_{dataset_name}/affine_delta1_depth'] = avg_metrics['affine_delta1_depth']
                val_logs[f'test_metric_{dataset_name}/affine_rel_point'] = avg_metrics['affine_rel_point']
                val_logs[f'test_metric_{dataset_name}/affine_delta1_point'] = avg_metrics['affine_delta1_point']

            wandb.log(val_logs)
    else:
        if misc.get_rank() == 0 and args.wandb:

            val_logs['test_metric/affine_rel_depth'] = avg_affine_rel_depth
            val_logs['test_metric/affine_delta1_depth'] = avg_affine_delta1_depth
            val_logs['test_metric/affine_rel_point'] = avg_affine_rel_point
            val_logs['test_metric/affine_delta1_point'] = avg_affine_delta1_point

            # Add per-dataset metrics to wandb
            for dataset_name, avg_metrics in per_dataset_avg_metrics.items():
                val_logs[f'test_metric_{dataset_name}/affine_rel_depth'] = avg_metrics['affine_rel_depth']
                val_logs[f'test_metric_{dataset_name}/affine_delta1_depth'] = avg_metrics['affine_delta1_depth']
                val_logs[f'test_metric_{dataset_name}/affine_rel_point'] = avg_metrics['affine_rel_point']
                val_logs[f'test_metric_{dataset_name}/affine_delta1_point'] = avg_metrics['affine_delta1_point']

            wandb.log(val_logs)

    if args.distributed:
        torch.distributed.barrier()

    return
