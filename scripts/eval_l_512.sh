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


# Zero-shot benchmark: PointDiT-L at 512x512, 3-step sampling, 7 datasets (3,444 samples).
# Expected (7 datasets, x100): Rel_p 4.85  d1_p 97.54  Rel_d 3.09  d1_d 98.25  BF1 10.36
# Drop --eval_no_save_gen to also write depth panels and point clouds

cd "$(dirname "$0")/.."

# The frozen DINOv3 encoder is not part of the checkpoint: its weights are gated, so
# they are downloaded separately into pretrained/dinov3/ (see MODELS.md).
CHECKPOINT=pretrained/pointditl-512-mixdata-nodinov3-240c1a4f.pth
export DINOV3_WEIGHTS_DIR=${DINOV3_WEIGHTS_DIR:-pretrained/dinov3}
EVAL_DATA=datasets/eval

python main.py \
--model PointDiT-L/16 --feature_embedding_type dinov3_vitl16 --proj_dropout 0.0 \
--evaluate_gen --eval_no_save_gen \
--img_size 512 --eval_depth_resize_height 512 --num_sampling_steps 3 \
--eval_depth_dataset --eval_depth_data_root ${EVAL_DATA} \
--eval_depth_dataset_name DIODE,KITTI,NYUv2,ETH3D,HAMMER,iBims-1,Booster \
--eval_boundary_datasets HAMMER,iBims-1,Booster \
--pretrained ${CHECKPOINT}
