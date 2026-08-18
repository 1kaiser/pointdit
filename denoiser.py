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

# Derived from JiT (https://github.com/LTH14/JiT), MIT License,
# Copyright (c) 2025 Tianhong Li. See THIRD_PARTY_NOTICES.
import torch
import torch.nn as nn
from model import PointDiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = PointDiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            attention_type=args.attention_type,
            feature_embedding_type=args.feature_embedding_type,
            dinov3_use_intermediate_layers=args.dinov3_use_intermediate_layers,
            dinov3_num_intermediate_layers=args.dinov3_num_intermediate_layers,
            finetune_feature_embedding=args.feature_embedding_lr_scale > 0,
        )
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale


        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.steps = args.num_sampling_steps

        self.args = args

    def sample_t(self, n: int, device=None):

        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels,
                return_model_input_output=False,
                force_zero_t=False,
                args=None,
                valid_mask=None,
                sky_mask=None,
                ):

        # Image-conditioned point map generation: the conditioning image is passed
        # through unchanged (the condition is never dropped during training).
        labels_dropped = labels  # image in this task

        # logit-normal timestep sampling: t = sigmoid(P_mean + P_std * eps)
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]

        if force_zero_t:
            # Force 10% of samples to be exactly 0.0
            # (Useful if t=0 requires special handling in your model)
            if args.force_zero_t_ratio is not None:
                force_zero_t_ratio = args.force_zero_t_ratio
            else:
                force_zero_t_ratio = 0.1

            mask_zero = torch.rand(t.shape, device=t.device) < force_zero_t_ratio
            t[mask_zero] = 0.0

        e = torch.randn_like(x) * self.noise_scale

        if args.noise_fill_invalid:
            assert valid_mask is not None
            mask_expanded = valid_mask.unsqueeze(1).float()
            x_filled = x * mask_expanded + e * (1 - mask_expanded)
            z = t * x_filled + (1 - t) * e
        else:
            z = t * x + (1 - t) * e

        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # velocity (v-) loss
        loss = (v - v_pred) ** 2

        if self.args.exclude_invalid_gt:
            if args.center_shift_point:
                # use valid mask from dataloader
                assert valid_mask is not None

                if args.sky_loss_weight > 0.0:
                    assert sky_mask is not None
                    # Combine valid_mask and sky_mask
                    non_sky_mask = valid_mask & (~sky_mask)  # Valid non-sky regions
                    sky_only_mask = sky_mask  # Sky regions
                    non_sky_mask_expanded = non_sky_mask.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]
                    sky_only_mask_expanded = sky_only_mask.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]

                    # 2. Robust Non-Sky Loss Calculation
                    # Check if there are ANY non-sky pixels in the batch
                    if non_sky_mask_expanded.any():
                        non_sky_loss = loss[non_sky_mask_expanded].mean()
                    else:
                        # If looking straight at sky, object loss is 0
                        non_sky_loss = torch.tensor(0.0, device=loss.device)

                    # 3. Robust Sky Loss Calculation
                    # Check if there are ANY sky pixels in the batch
                    if sky_only_mask_expanded.any():
                        sky_loss = loss[sky_only_mask_expanded].mean()
                    else:
                        # If indoors or looking down, sky loss is 0
                        sky_loss = torch.tensor(0.0, device=loss.device)

                    # 4. Final Weighted Sum
                    # We maintain gradients for whichever part exists
                    loss = non_sky_loss + args.sky_loss_weight * sky_loss

                else:
                    valid_mask_expanded = valid_mask.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]
                    loss = loss[valid_mask_expanded].mean()
            else:
                valid_mask = (x[:, 2:] > 0).repeat(1, 3, 1, 1)
                loss = loss[valid_mask].mean()
        else:
            loss = loss.mean(dim=(1, 2, 3)).mean()

        if return_model_input_output:
            model_in_out = {
                't': t,
                'input': z,
                'output': x_pred
            }
            return loss, model_in_out

        return loss


    @torch.no_grad()
    def generate(self, labels, return_intermediate_steps=False):
        device = labels.device
        bsz = labels.size(0)
        # Use the condition image size rather than a fixed self.img_size.
        _, _, H, W = labels.shape
        # --generate_noise_scale 0.0 (the default) starts the ODE from zeros, which
        # makes inference deterministic.
        z = self.args.generate_noise_scale * torch.randn(bsz, 3, H, W, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device)
        timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if return_intermediate_steps:
            intermediate_results = {}
            # hardcoded for 50 steps for now
            intermediate_steps = [0, 15, 22, 30, 38, 45]

            if self.steps == 50:
                pass
            elif self.steps == 100:
                intermediate_steps = [x * 2 for x in intermediate_steps]
            elif self.steps == 200:
                intermediate_steps = [x * 4 for x in intermediate_steps]
            elif self.steps == 500:
                intermediate_steps = [x * 10 for x in intermediate_steps]
            elif self.steps == 1000:
                intermediate_steps = [x * 20 for x in intermediate_steps]
            else:
                stride = self.steps // 6
                intermediate_steps = [i * stride for i in range(6)]
            if self.steps < 10:
                intermediate_steps = list(range(self.steps))

        # ode
        # Save initial noise as step -1 (before any denoising)
        if return_intermediate_steps:
            intermediate_results[-1] = z.clone()  # Use -1 to distinguish from step 0 denoising

        # Cache the (frozen) image-encoder features once, instead of recomputing them
        # at every sampling step.
        h_patches = H // self.net.patch_size
        w_patches = W // self.net.patch_size
        cached_y_emb_cond = self.net.extract_y_embedding(labels, h_patches, w_patches)

        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = self._euler_step(z, t, t_next, labels, cached_y_emb_cond)

            if return_intermediate_steps and i in intermediate_steps:
                intermediate_results[i] = z

        # last step euler; guarded because self.steps == 0 leaves timesteps with a
        # single element, so timesteps[-2] would be out of range
        if self.steps > 0:
            z = self._euler_step(z, timesteps[-2], timesteps[-1], labels, cached_y_emb_cond)

        if return_intermediate_steps:
            return z, intermediate_results

        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, cached_y_emb_cond=None):
        x_cond = self.net(z, t.flatten(), labels, cached_y_emb=cached_y_emb_cond)
        return (x_cond - z) / (1.0 - t).clamp_min(self.args.sample_t_eps)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, cached_y_emb_cond=None):
        v_pred = self._forward_sample(z, t, labels, cached_y_emb_cond)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
