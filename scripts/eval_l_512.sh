#!/usr/bin/env bash

# Zero-shot benchmark: PointDiT-L at 512x512, 3-step sampling, 7 datasets (3,444 samples).
# Expected (7 datasets, x100): Rel_p 4.85  d1_p 97.54  Rel_d 3.09  d1_d 98.25  BF1 10.36
# Drop --eval_no_save_gen to also write depth panels and point clouds

cd "$(dirname "$0")/.."

CHECKPOINT=pretrained/pointditl-512.pth
EVAL_DATA=datasets/eval

python main.py \
--model PointDiT-L/16 --feature_embedding_type dinov3_vitl16 --proj_dropout 0.0 \
--evaluate_gen --eval_no_save_gen \
--img_size 512 --eval_depth_resize_height 512 --num_sampling_steps 3 \
--eval_depth_dataset --eval_depth_data_root ${EVAL_DATA} \
--eval_depth_dataset_name DIODE,KITTI,NYUv2,ETH3D,HAMMER,iBims-1,Booster \
--eval_boundary_datasets HAMMER,iBims-1,Booster \
--pretrained ${CHECKPOINT}
