# coding=utf-8

# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import get_2d_sincos_pos_embed, RMSNorm, rotate_half
from util.paths import repo_path

try:
    from flash_attn_interface import flash_attn_func
    FA3_AVAILABLE = True
except ImportError:
    FA3_AVAILABLE = False


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DynamicRoPE:
    """Applies precomputed 2-D rotary position embeddings to q/k."""
    def __init__(self, cos, sin):
        self.cos = cos
        self.sin = sin

    def __call__(self, t):
        return t * self.cos + rotate_half(t) * self.sin


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        # Allow any size divisible by patch_size (support rectangular inputs)
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, \
            f"Input size ({H}x{W}) must be divisible by patch size ({self.patch_size[0]}x{self.patch_size[1]})"
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def scaled_dot_product_attention_flash3(query, key, value, dropout_p=0.0) -> torch.Tensor:
    """
    Computes scaled dot product attention using the flash_attn library (FA3).
    
    Args:
        query: Shape (Batch, NumHeads, SeqLen_Q, HeadDim)
        key:   Shape (Batch, NumHeads, SeqLen_K, HeadDim)
        value: Shape (Batch, NumHeads, SeqLen_K, HeadDim)
    
    Returns:
        output: Shape (Batch, NumHeads, SeqLen_Q, HeadDim)
    """
    assert FA3_AVAILABLE, "flash_attn_func is not available. Please install flash attention library."

    # 1. Permute dimensions from (B, H, S, D) to (B, S, H, D)
    # Flash Attention requires the 'Heads' dimension to be 3rd, not 2nd.
    # Mannually cast to bfloat16 for better performance and compatibility.
    q = query.transpose(1, 2).to(torch.bfloat16)
    k = key.transpose(1, 2).to(torch.bfloat16)
    v = value.transpose(1, 2).to(torch.bfloat16)

    # 2. Run Flash Attention
    # Note: softmax_scale defaults to 1/sqrt(d) if not provided.
    assert dropout_p == 0.0, "Flash Attention 3 currently does not support dropout."
    # https://github.com/Dao-AILab/flash-attention/issues/1377
    output = flash_attn_func(
        q, k, v,
        # dropout_p=dropout_p,
    )[0]

    # 3. Permute back to (B, H, S, D)
    return output.transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.,
                 attention_type='torch'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.attention_type = attention_type

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        if self.attention_type == 'torch':
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_drop.p if self.training else 0.,)
        elif self.attention_type == 'flash3':
            x = scaled_dot_product_attention_flash3(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            raise NotImplementedError(f'Attention type {self.attention_type} not implemented')

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of PointDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class PointDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0,
                 attention_type='torch'):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop,
                              attention_type=attention_type)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, feat_rope):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PointDiT(nn.Module):
    """
    Pixel-space diffusion transformer that denoises a point map, conditioned on
    frozen DINOv3 features of the input image.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        attention_type='torch',
        feature_embedding_type='dinov3_vitb16',
        dinov3_use_intermediate_layers=True,
        dinov3_num_intermediate_layers=4,
        finetune_feature_embedding=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.attention_type = attention_type
        self.feature_embedding_type = feature_embedding_type
        self.dinov3_use_intermediate_layers = dinov3_use_intermediate_layers
        self.dinov3_num_intermediate_layers = dinov3_num_intermediate_layers
        self.finetune_feature_embedding = finetune_feature_embedding

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)

        # frozen pre-trained image encoder producing the conditioning tokens
        if self.feature_embedding_type.startswith('dinov3'):
            vit_type = self.feature_embedding_type.split('_')[-1]
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
            # Configurable so the model can be built on any machine. Defaults assume the
            # DINOv3 checkout in third_party/ and the gated weights in pretrained/dinov3/,
            # both resolved inside the repository rather than against the cwd.
            dinov3_repo = os.environ.get(
                'DINOV3_REPO', repo_path('third_party/dinov3'))
            dinov3_weights_dir = os.environ.get(
                'DINOV3_WEIGHTS_DIR', repo_path('pretrained/dinov3'))
            local_weights_path = os.path.join(
                dinov3_weights_dir, f'dinov3_{vit_type}_pretrain_lvd1689m-{sha}.pth')
            if not os.path.isdir(dinov3_repo):
                raise FileNotFoundError(
                    f'DINOv3 repository not found at "{dinov3_repo}". Clone it with\n'
                    f'    git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3\n'
                    f'or set DINOV3_REPO to an existing checkout.')
            # NOTE: need to load the weights in the main function due to the random init of the __init__
            if os.path.isfile(local_weights_path):
                self.y_embedder = torch.hub.load(
                    dinov3_repo, f'dinov3_{vit_type}', source='local', weights=local_weights_path)
            else:
                # Every PointDiT checkpoint already stores the DINOv3 encoder weights under
                # net.y_embedder.*, so evaluation/inference only needs the architecture here.
                # The gated LVD-1689M weights are required only to train from scratch, and
                # main.py warns about that case (it can see whether a checkpoint is coming).
                self.y_embedder = torch.hub.load(
                    dinov3_repo, f'dinov3_{vit_type}', source='local', pretrained=False)
            # freeze DINOv3 parameters
            if not self.finetune_feature_embedding:
                for param in self.y_embedder.parameters():
                    param.requires_grad = False

            self.y_embedder.num_patches = self.y_embedder.patch_embed.num_patches

            # compute equally-spaced layer indices for intermediate feature extraction
            if self.dinov3_use_intermediate_layers:
                dinov3_depth = len(self.y_embedder.blocks)
                n = self.dinov3_num_intermediate_layers
                # n equally-spaced layers
                self.dinov3_intermediate_layer_indices = [
                    int((i + 1) * dinov3_depth / n) - 1 for i in range(n)
                ]
                self.y_embedder.num_features = self.y_embedder.num_features * n

        else:
            raise NotImplementedError(f'Feature embedding type {self.feature_embedding_type} not implemented')

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # position embedding for the conditioning image tokens (frozen sin-cos, added in
        # forward); only needed when the encoder's feature dim differs from the model dim.
        y_feat_dim = self.y_embedder.num_features
        if y_feat_dim != hidden_size:
            self.pos_embed_y = nn.Parameter(torch.zeros(1, num_patches, y_feat_dim), requires_grad=False)
        else:
            self.pos_embed_y = None

        # Image and point tokens are concatenated along the channel dim, then projected
        # back to the model dim (this is PointDiT's conditioning mechanism).
        self.concat_proj = nn.Linear(y_feat_dim + hidden_size, hidden_size, bias=True)

        # transformer
        self.blocks = nn.ModuleList([
            PointDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                          attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                          proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                          attention_type=attention_type,
                          )
            for i in range(depth)
        ])

        # output head: linear D -> patch_size^2 * 3, unpatchified back to H x W x 3
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if hasattr(self, 'pos_embed_y') and self.pos_embed_y is not None:
            # Initialize (and freeze) pos_embed_y by sin-cos embedding:
            # TODO: assuming same number of patches as pointmap
            pos_embed_y = get_2d_sincos_pos_embed(self.pos_embed_y.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            self.pos_embed_y.data.copy_(torch.from_numpy(pos_embed_y).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def interpolate_pos_encoding(self, h, w):
        """
        Interpolate position embeddings to handle variable input sizes.
        h, w: number of patches in height and width
        Returns: pos_embed [1, h*w, C]
        """
        N = h * w
        if N == self.pos_embed.shape[1]:
            # Check if it's the same square size
            orig_size = int(self.pos_embed.shape[1] ** 0.5)
            if h == orig_size and w == orig_size:
                return self.pos_embed

        # Check cache
        cache_key = (h, w)
        if not hasattr(self, '_pos_embed_cache'):
            self._pos_embed_cache = {}
        if cache_key in self._pos_embed_cache:
            return self._pos_embed_cache[cache_key]

        # Get original grid size (square)
        orig_size = int(self.pos_embed.shape[1] ** 0.5)

        # Reshape to 2D grid
        pos_embed = self.pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)

        # Interpolate to new size
        pos_embed = F.interpolate(pos_embed, size=(h, w), mode='bicubic', align_corners=False)

        # Reshape back
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)

        if len(self._pos_embed_cache) > 64:
            self._pos_embed_cache.clear()
        self._pos_embed_cache[cache_key] = pos_embed
        return pos_embed

    def interpolate_pos_encoding_y(self, h, w):
        """
        Interpolate y position embeddings (for image features) to handle variable input sizes.
        h, w: number of patches in height and width
        Returns: pos_embed_y [1, h*w, C]
        """
        if self.pos_embed_y is None:
            return self.interpolate_pos_encoding(h, w)

        N = h * w
        if N == self.pos_embed_y.shape[1]:
            orig_size = int(self.pos_embed_y.shape[1] ** 0.5)
            if h == orig_size and w == orig_size:
                return self.pos_embed_y

        # Check cache
        cache_key = (h, w)
        if not hasattr(self, '_pos_embed_y_cache'):
            self._pos_embed_y_cache = {}
        if cache_key in self._pos_embed_y_cache:
            return self._pos_embed_y_cache[cache_key]

        orig_size = int(self.pos_embed_y.shape[1] ** 0.5)
        pos_embed = self.pos_embed_y.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(h, w), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)

        if len(self._pos_embed_y_cache) > 64:
            self._pos_embed_y_cache.clear()
        self._pos_embed_y_cache[cache_key] = pos_embed
        return pos_embed

    def get_rope_for_size(self, h, w, device):
        """
        Generate rotary position embeddings for given grid size (cached).
        h, w: number of patches in height and width
        Returns: (cos, sin) each of shape [h*w, D]
        """
        cache_key = (h, w, device)
        if not hasattr(self, '_rope_cache'):
            self._rope_cache = {}
        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]

        half_head_dim = self.hidden_size // self.num_heads // 2
        theta = 10000
        dim = half_head_dim
        freqs = 1. / (theta ** (torch.arange(0, dim, 2, device=device)[:(dim // 2)].float() / dim))

        # Create position indices for h and w
        t_h = torch.arange(h, device=device).float()
        t_w = torch.arange(w, device=device).float()

        # Compute frequencies for each dimension
        freqs_h = torch.einsum('i,j->ij', t_h, freqs)  # [h, D/4]
        freqs_w = torch.einsum('i,j->ij', t_w, freqs)  # [w, D/4]

        # Repeat for complex representation (interleave)
        freqs_h = freqs_h.repeat_interleave(2, dim=-1)  # [h, D/2]
        freqs_w = freqs_w.repeat_interleave(2, dim=-1)  # [w, D/2]

        # Create 2D grid: [h, w, D]
        freqs_grid = torch.cat([
            freqs_h[:, None, :].expand(-1, w, -1),
            freqs_w[None, :, :].expand(h, -1, -1)
        ], dim=-1)

        # Flatten to [h*w, D]
        freqs_flat = freqs_grid.reshape(h * w, -1)

        result = (freqs_flat.cos(), freqs_flat.sin())
        if len(self._rope_cache) > 64:
            self._rope_cache.clear()
        self._rope_cache[cache_key] = result
        return result

    def unpatchify(self, x, p, h=None, w=None):
        """
        x: (N, T, patch_size**2 * C)
        h, w: number of patches (if None, assumes square)
        imgs: (N, C, H*p, W*p)
        """
        c = self.out_channels
        if h is None or w is None:
            h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], f"h*w ({h}*{w}={h*w}) != num_patches ({x.shape[1]})"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, y, cached_y_emb=None):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,) or (N, C, H, W)
        cached_y_emb: Optional pre-computed y embedding [N, L, C] for sampling efficiency
        """
        B, C, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # Interpolate position embeddings for current size
        pos_embed = self.interpolate_pos_encoding(h_patches, w_patches)

        # Generate RoPE for current size
        rope_cos, rope_sin = self.get_rope_for_size(h_patches, w_patches, x.device)

        # class and time embeddings
        t_emb = self.t_embedder(t)

        # Use cached y_emb if provided (for sampling efficiency), otherwise compute fresh
        if cached_y_emb is not None:
            y_emb = cached_y_emb
        else:
            if self.feature_embedding_type.startswith('dinov3'):
                # DINOv3 image normalization
                y = (y + 1) / 2.  # to [0, 1]
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(y.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(y.device)
                y = (y - mean) / std

                # downsample input when necessary
                if self.patch_size != 16:
                    downsample_factor = self.patch_size // 16
                    y = F.interpolate(y, scale_factor=1.0 / downsample_factor, mode='bilinear', align_corners=False)

                # DINOv3 feature embedding
                # patch size 16
                if self.dinov3_use_intermediate_layers:
                    # Extract features from n equally-spaced layers and concatenate
                    intermediate_features = self.y_embedder.get_intermediate_layers(
                        y, n=self.dinov3_intermediate_layer_indices, norm=True
                    )
                    # intermediate_features is a tuple of n tensors, each [B, num_patches, embed_dim]
                    y_emb = torch.cat(intermediate_features, dim=-1)  # [B, num_patches, n * embed_dim]
                else:
                    # Use only final layer features
                    y_emb = self.y_embedder.forward_features(y)['x_norm_patchtokens']

            else:
                raise NotImplementedError(f'Feature embedding type {self.feature_embedding_type} not implemented')

        c = t_emb
        # add pos embedding (interpolated for current size) - skip if using cached (already included)
        if cached_y_emb is None:
            pos_embed_y = self.interpolate_pos_encoding_y(h_patches, w_patches)
            y_emb = y_emb + pos_embed_y

        # forward PointDiT
        x = self.x_embedder(x)
        x = x + pos_embed

        # Create dynamic RoPE for current size
        feat_rope = DynamicRoPE(rope_cos, rope_sin)

        # concat image and point tokens along the channel dim, then project to the model dim
        x = torch.cat([y_emb, x], dim=-1)  # [N, L, y_feat_dim + C]
        x = self.concat_proj(x)  # [N, L, C]

        for block in self.blocks:
            x = block(x, c, feat_rope)

        x = self.final_layer(x, c)

        return self.unpatchify(x, self.patch_size, h_patches, w_patches)

    def extract_y_embedding(self, y, h_patches, w_patches):
        """
        Extract DINOv3 feature embeddings from conditioning image.
        Used for caching during sampling to avoid redundant feature extraction.

        Args:
            y: Conditioning image tensor [B, 3, H, W] in range [-1, 1]
            h_patches: Number of patches in height (for position embedding)
            w_patches: Number of patches in width (for position embedding)

        Returns:
            y_emb: Feature embeddings [B, num_patches, feature_dim]
        """
        if self.feature_embedding_type.startswith('dinov3'):
            # DINOv3 image normalization
            y = (y + 1) / 2.  # to [0, 1]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(y.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(y.device)
            y = (y - mean) / std

            if self.patch_size != 16:
                downsample_factor = self.patch_size // 16
                y = F.interpolate(y, scale_factor=1.0 / downsample_factor, mode='bilinear', align_corners=False)

            if self.dinov3_use_intermediate_layers:
                intermediate_features = self.y_embedder.get_intermediate_layers(
                    y, n=self.dinov3_intermediate_layer_indices, norm=True
                )
                y_emb = torch.cat(intermediate_features, dim=-1)
            else:
                y_emb = self.y_embedder.forward_features(y)['x_norm_patchtokens']

        else:
            raise NotImplementedError(f'Feature embedding type {self.feature_embedding_type} not implemented')

        # For img2point task, add position embedding
        pos_embed_y = self.interpolate_pos_encoding_y(h_patches, w_patches)
        y_emb = y_emb + pos_embed_y

        return y_emb


def PointDiT_B_16(**kwargs):
    return PointDiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, patch_size=16, **kwargs)


def PointDiT_L_16(**kwargs):
    return PointDiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, patch_size=16, **kwargs)


def PointDiT_H_16(**kwargs):
    return PointDiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, patch_size=16, **kwargs)


# The /16 suffix is the patch size; main.py parses it back out of --model.
PointDiT_models = {
    'PointDiT-B/16': PointDiT_B_16,
    'PointDiT-L/16': PointDiT_L_16,
    'PointDiT-H/16': PointDiT_H_16,
}

# Legacy aliases. This repository began as a fork of JiT (see THIRD_PARTY_NOTICES),
# whose architecture names it used until the public release. Accepted so older
# --model strings keep working; prefer the PointDiT-* names above.
PointDiT_models['JiT-B/16'] = PointDiT_B_16
PointDiT_models['JiT-L/16'] = PointDiT_L_16
PointDiT_models['JiT-H/16'] = PointDiT_H_16
