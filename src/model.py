import torch
from torch import nn
import math
from einops import rearrange
from src.mamba_simple import Mamba


def split_band(x, move_num, spec_num):
    """
    Splits the spectral dimension into overlapping segments.
    x: Tensor of shape (b, c, h, w)
    Returns tensor of shape (b, n, c, h, w) where n is number of segments.
    """
    b, c, h, w = x.shape
    slices = []
    for i in range(0, c, move_num):
        if i + spec_num > c:
            slice = x[:, c - spec_num : c, :, :]
        else:
            slice = x[:, i : i + spec_num, :, :]
        slices.append(slice)
    slices = torch.stack(slices, dim=1)
    return slices


class Residual_SSMN(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize_SSMN(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class mamba_block1(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Residual_SSMN(
                    LayerNormalize_SSMN(
                        dim,
                        Mamba(
                            d_model=dim,
                            d_state=64,
                            d_conv=4,
                            expand=2,
                            use_fast_path=False,
                        ),
                    )
                )
            )

    def forward(self, x):
        for attn in self.layers:
            x = attn(x)
        return x


class mamba_block2(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Residual_SSMN(
                    LayerNormalize_SSMN(
                        dim,
                        Mamba(
                            d_model=dim,
                            d_state=64,
                            d_conv=4,
                            expand=2,
                            use_fast_path=False,
                        ),
                    )
                )
            )

    def forward(self, x):
        for attn in self.layers:
            x = attn(x)
        return x


class RandomMasking(nn.Module):
    def __init__(self, mask_prob=0.5, mask_size=0.2, mode="full_spectrum"):
        """
        mask_prob: probability to apply masking per sample
        mask_size: fraction of feature (spatial area or spectral bands) to mask
        mode: 'full_spectrum' or 'random_channels' or 'spatial_patch'
        """
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_size = mask_size
        self.mode = mode

    def forward(self, x):
        if not self.training or torch.rand(1) > self.mask_prob:
            return x
        B, C, H, W = x.shape
        mask = torch.ones_like(x)
        if self.mode == "full_spectrum":
            for b in range(B):
                # choose a random rectangular patch to mask
                patch_h = max(1, int(H * self.mask_size))
                patch_w = max(1, int(W * self.mask_size))
                i = torch.randint(0, H - patch_h + 1, (1,)).item()
                j = torch.randint(0, W - patch_w + 1, (1,)).item()
                mask[b, :, i : i + patch_h, j : j + patch_w] = 0.0

        elif self.mode == "random_channels":
            num_mask = int(C * self.mask_size)
            mask_idx = torch.randperm(C)[:num_mask]
            mask[:, mask_idx, :, :] = 0.0
        return x * mask


class LSSDecoderBlock(nn.Module):

    def __init__(self, channels, hss_depth=1):
        super().__init__()
        self.hss = mamba_block1(channels, hss_depth)
        self.local3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.local5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        g = rearrange(x, "b c h w -> b (h w) c")  # [B, C, L]
        g = self.hss(g)  # [B, C, L]
        g = rearrange(g, "b (h w) c -> b c h w", h=H, w=W)
        l3 = self.local3(x)
        l5 = self.local5(x)
        cat = torch.cat([g, l3, l5], dim=1)
        out = self.act(self.fuse(cat)) + x
        return out


class AnomalyDetectionModel(nn.Module):
    """
    Encoder-Decoder architecture for anomaly detection.
    It uses a dual-branch approach:
      - The spatial branch processes the patch via 2D convolutions and mamba_block1.
      - The spectral branch splits the spectral channels, processes them via mamba_block2.
    The outputs are fused via adaptive gating and then decoded to reconstruct the input patch.
    """

    def __init__(
        self,
        in_channels,
        mode,
        dim=64,
        depth=1,
        spec_num=12,
        spec_rate=0.5,
        spa_token=16,
    ):
        super().__init__()
        self.mode = mode
        self.spec_num = spec_num
        self.move_num = int(math.ceil(spec_num * spec_rate))
        self.spa_token = spa_token

        # Preprocessing: project input channels to a feature dimension.
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # Spatial branch: process through 2D convolutions and mamba block.
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.spatial_conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.spatial_mamba = mamba_block1(dim, depth)

        # Spectral branch: process the spectral features.

        spe_dim = dim
        self.dim = dim
        self.nn1 = nn.Sequential(
            nn.Linear(dim, spe_dim),
            nn.LayerNorm(spe_dim),
            nn.GELU(),
        )
        dropout = 0.1
        self.dropout = nn.Dropout(dropout)
        num_patch = math.floor(
            (dim - (self.spec_num - self.move_num)) / self.move_num
        ) + math.ceil(
            (
                ((dim - (self.spec_num - self.move_num)) % self.move_num)
                + (self.spec_num - self.move_num)
            )
            / self.move_num
        )

        self.spe_token1 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.spe_token2 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.SPEM = mamba_block2(self.spec_num, depth)

        self.nn2 = nn.Sequential(
            nn.Linear(self.spec_num * num_patch, dim),  # 原始光谱
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.spectral_mamba = mamba_block2(spec_num, depth)

        self.spec_fc = nn.Sequential(
            nn.Linear(spec_num * 1, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        decoder_ch = dim
        self.lss1 = LSSDecoderBlock(decoder_ch, hss_depth=2)
        self.lss2 = LSSDecoderBlock(decoder_ch, hss_depth=2)

        self.out_conv = nn.Conv2d(decoder_ch, in_channels, 1)

        self.use_random_mask = True
        self.random_mask = RandomMasking()

        self.fusion_fc = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),  # 128 -> 64
            nn.GELU(),
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2 * dim, 1, kernel_size=1),  # 128 -> 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x: input patch tensor of shape (b, C, H, W), as produced by block embedding.
        Returns:
           - reconstructed patch output: (b, C, H, W)
           - fusion feature (optional for loss visualization)
        """
        b, C, H, W = x.shape
        if self.use_random_mask:
            x = self.random_mask(x)
        feat = self.preprocess(x).permute(0, 2, 3, 1)

        # Spatial branch:
        spa_feat = feat.permute(0, 3, 1, 2)
        spatial_feat = self.spatial_conv(spa_feat)
        spatial_feat = self.spatial_conv2(spatial_feat)
        spatial_feat = self.spatial_mamba(
            rearrange(spatial_feat, "b c h w -> b (h w) c")
        )
        spatial_feat = rearrange(spatial_feat, "b (h w) c -> b c h w", h=H, w=W)

        # Spectral branch:
        x_s = feat.unsqueeze(1)
        x_s = self.spe_token1(x_s) + self.spe_token2(x_s) + x_s
        x_s = self.dropout(x_s).squeeze(1).permute(0, 3, 1, 2)
        Patch_pool = torch.nn.AvgPool2d((H, W)).cuda()
        x_s = Patch_pool(x_s)
        spec_slices = split_band(x_s, self.move_num, self.spec_num)
        bb, nn, cc, hh, ww = spec_slices.shape
        spec_slices = rearrange(spec_slices, "b n c h w -> (b h w) n c")

        spec_feat = self.spectral_mamba(spec_slices)
        spec_feat = rearrange(spec_feat, "(b h w) n c -> b (n c) h w", h=hh, w=ww).mean(
            -1
        )
        spec_feat = (
            self.nn2(spec_feat.squeeze(2)).unsqueeze(1).unsqueeze(1).permute(0, 3, 1, 2)
        )
        spec_feat = spec_feat.view(b, self.dim, 1, 1).expand(b, self.dim, H, W)

        if self.mode == "spatial":
            fused = self.fusion_fc(spatial_feat)  # no spectral
        elif self.mode == "spectral":
            fused = self.fusion_fc(spec_feat)  # no spatial
        else:  # 'Gated Fusion'
            cat = torch.cat([spatial_feat, spec_feat], dim=1)
            gate = self.gate_conv(cat)
            fused_raw = gate * spatial_feat + (1.0 - gate) * spec_feat
            fused = self.fusion_fc(fused_raw)
        # Decoder
        z = self.lss1(fused)
        z = self.lss2(z)
        out = self.out_conv(z)
        return out, fused
