#!/usr/bin/env bash

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# PointDiT-H (512x512) on your own images. Set IMAGE_DIR below; .jpg/.png, nested dirs are fine.
# Results go to generation/pretrained-pointdith-512-mixdata-nodinov3-cb01dd3b/<IMAGE_DIR folder name>/, so
# IMAGE_DIR=assets/demo writes to .../demo/. Each output is named after its input:
# sample0.png -> sample0_depth.png and sample0_pointcloud.ply.

# Any input resolution works: the full frame is resized (aspect ratio kept) to the
# resolution closest to the 32x32 tokens the model was trained on, the prediction runs
# there, then it is resized back to the input resolution for saving. Retarget the token
# budget with --eval_wild_target_tokens, or pass --eval_depth_resize_height 512 to go
# back to resizing the shortest side and center-cropping (which discards the edges).

cd "$(dirname "$0")/.."

IMAGE_DIR=assets/demo
# The frozen DINOv3 encoder is not part of the checkpoint: its weights are gated, so
# they are downloaded separately into pretrained/dinov3/ (see MODELS.md).
CHECKPOINT=pretrained/pointdith-512-mixdata-nodinov3-cb01dd3b.pth
export DINOV3_WEIGHTS_DIR=${DINOV3_WEIGHTS_DIR:-pretrained/dinov3}

python main.py \
--model PointDiT-H/16 --feature_embedding_type dinov3_vith16plus --proj_dropout 0.2 \
--evaluate_gen --eval_wild_images \
--eval_wild_images_dir ${IMAGE_DIR} \
--img_size 512 --num_sampling_steps 3 \
--pretrained ${CHECKPOINT}
