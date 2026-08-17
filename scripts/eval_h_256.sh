#!/usr/bin/env bash

# Zero-shot benchmark: PointDiT-H at 256x256, 1-step sampling, 7 datasets (3,444 samples).
# Drop --eval_no_save_gen to also write depth panels and point clouds
# --no_remove_sky is needed here: Stage 1 trains on SceneNet with handle_sky: false,
# so sky is never placed at the far plane the default threshold looks for.

cd "$(dirname "$0")/.."

CHECKPOINT=pretrained/pointdith-256-scenenet.pth
EVAL_DATA=datasets/eval

python main.py \
--model PointDiT-H/16 --feature_embedding_type dinov3_vith16plus --proj_dropout 0.2 \
--evaluate_gen --eval_no_save_gen \
--img_size 256 --eval_depth_resize_height 256 --num_sampling_steps 1 \
--eval_depth_dataset --eval_depth_data_root ${EVAL_DATA} \
--eval_depth_dataset_name DIODE,KITTI,NYUv2,ETH3D,HAMMER,iBims-1,Booster \
--eval_boundary_datasets HAMMER,iBims-1,Booster \
--no_remove_sky \
--pretrained ${CHECKPOINT}
