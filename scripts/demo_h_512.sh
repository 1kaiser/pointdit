#!/usr/bin/env bash

# PointDiT-H (512x512) on your own images. Set IMAGE_DIR below; .jpg/.png, nested dirs are fine.
# Results go to generation/pretrained-pointdith-512/<IMAGE_DIR folder name>/, so
# IMAGE_DIR=assets/demo writes to .../demo/. Each output is named after its input:
# sample0.png -> sample0_depth.png and sample0_pointcloud.ply.

# Any input resolution works: the full frame is resized (aspect ratio kept) to the
# resolution closest to the 32x32 tokens the model was trained on, the prediction runs
# there, then it is resized back to the input resolution for saving. Retarget the token
# budget with --eval_wild_target_tokens, or pass --eval_depth_resize_height 512 to go
# back to resizing the shortest side and center-cropping (which discards the edges).

cd "$(dirname "$0")/.."

IMAGE_DIR=assets/demo
CHECKPOINT=pretrained/pointdith-512.pth

python main.py \
--model PointDiT-H/16 --feature_embedding_type dinov3_vith16plus --proj_dropout 0.2 \
--evaluate_gen --eval_wild_images \
--eval_wild_images_dir ${IMAGE_DIR} \
--img_size 512 --num_sampling_steps 3 \
--pretrained ${CHECKPOINT}
