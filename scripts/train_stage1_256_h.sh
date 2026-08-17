#!/usr/bin/env bash

# Stage 1: PointDiT-H pre-training at 256x256 on SceneNet-RGBD.
# Original run: 64 GPUs x batch 16 (effective batch 1024, lr 2e-4), 30 epochs.
# lr = blr * batch_size * num_gpus / 256, so keep that product at 1024 or pass --lr.
# The original run used --attention_type flash3; torch is the portable default.
# --eval_depth_max_samples 64 keeps the interleaved --online_eval fast: 64 samples per
# dataset (evenly strided, not the first 64) instead of all 3,444. Drop it to score every
# sample, which is what the eval_*.sh scripts do.

cd "$(dirname "$0")/.."

# Multi-node: raise --num_machines, set --machine_rank per node, and add
# --main_process_ip <rank-0 host> --main_process_port 29500 to the launcher.

SCENENET_PATH=datasets/train/scenenet-rgbd
EVAL_DATA=datasets/eval
OUTPUT_DIR=checkpoints/pointdit-h-256
mkdir -p ${OUTPUT_DIR}

accelerate launch --num_processes 8 --num_machines 1 --machine_rank 0 \
--mixed_precision bf16 --dynamo_backend no \
main.py \
--model PointDiT-H/16 --feature_embedding_type dinov3_vith16plus --proj_dropout 0.2 \
--split all \
--img_size 256 --noise_scale 1.0 \
--dataset_name scenenet --data_path ${SCENENET_PATH} \
--depth_intrinsics_resize_height 256 \
--batch_size 16 --blr 5e-5 \
--epochs 30 --warmup_epochs 5 \
--save_ckpt_freq 5 --save_last_freq 1 \
--online_eval --eval_freq 1 --num_sampling_steps 1 \
--eval_depth_dataset --eval_depth_data_root ${EVAL_DATA} \
--eval_depth_dataset_name DIODE,KITTI,NYUv2,ETH3D,HAMMER,iBims-1,Booster \
--eval_depth_resize_height 256 --eval_depth_max_samples 64 \
--output_dir ${OUTPUT_DIR} --resume \
2>&1 | tee -a ${OUTPUT_DIR}/train.log
