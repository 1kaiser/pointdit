#!/usr/bin/env bash

# Stage 2: PointDiT-H fine-tuning at 512x512 on the 11-dataset mixture.
# Original run: 128 GPUs x batch 4 (effective batch 512, lr 2e-4); released at epoch 2.
# lr = blr * batch_size * num_gpus / 256, so keep that product at 512 or pass --lr.
# --noise_scale 2.0 is the Simple-Diffusion rule vs the 256 base: 1.0 * sqrt(512^2/256^2).
# --eval_depth_max_samples 256 keeps the interleaved --online_eval fast: 256 samples per
# dataset (evenly strided, not the first 256) instead of every sample. Drop it to score
# the full sets, which is what the eval_*.sh scripts do.

cd "$(dirname "$0")/.."

# Multi-node: raise --num_machines, set --machine_rank per node, and add
# --main_process_ip <rank-0 host> --main_process_port 29500 to the launcher.

DATASET_CONFIG=dataloader/configs/res512mix.yaml
EVAL_DATA=datasets/eval
STAGE1_CKPT=checkpoints/pointdit-h-256/checkpoint-last.pth
OUTPUT_DIR=checkpoints/pointdit-h-512
mkdir -p ${OUTPUT_DIR}

accelerate launch --num_processes 8 --num_machines 1 --machine_rank 0 \
--mixed_precision bf16 --dynamo_backend no \
main.py \
--model PointDiT-H/16 --feature_embedding_type dinov3_vith16plus --proj_dropout 0.2 \
--split all \
--img_size 512 --noise_scale 2.0 \
--dataset_config ${DATASET_CONFIG} \
--sky_loss_weight 0.01 --noise_fill_invalid \
--batch_size 4 --blr 1e-4 \
--epochs 3 --warmup_epochs 0 \
--save_ckpt_freq 2 --save_last_freq 1 \
--online_eval --eval_freq 1 --num_sampling_steps 1 \
--eval_depth_dataset --eval_depth_data_root ${EVAL_DATA} \
--eval_depth_dataset_name DIODE,KITTI,NYUv2,ETH3D,HAMMER \
--eval_depth_resize_height 512 --eval_depth_max_samples 256 \
--pretrained ${STAGE1_CKPT} --resize_posemb \
--output_dir ${OUTPUT_DIR} --resume \
2>&1 | tee -a ${OUTPUT_DIR}/train.log
