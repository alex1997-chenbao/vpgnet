# mmdetection3d/models/fusion_modules/freq_prompt_local_mha_cross_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from typing import List, Dict, Tuple, Optional

from mmdet3d.structures import points_cam2img
from mmdet3d.models.layers.fusion_layers.vote_fusion import (
    apply_3d_transformation,
    coord_2d_transform,
)


class PromptGenerator(nn.Module):
    """
    简化版：只生成一个频率提示（FFT 只计算一次）
    """
    def __init__(self, scale_factor=8, embed_dim=64, freq_nums=0.25, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.freq_nums = freq_nums
        self.patch_size = patch_size
        self.target_dim = embed_dim // scale_factor  # e.g., 64 // 8 = 8

        self.freq_proj = nn.Conv2d(3, self.target_dim, kernel_size=patch_size, stride=patch_size)

        self.lightweight_mlp = nn.Sequential(
            nn.Linear(self.target_dim, self.target_dim),
            nn.GELU()
        )

        self.shared_mlp = nn.Linear(self.target_dim, embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def fft_highpass(self, x, rate):
        mask = torch.zeros_like(x, device=x.device)
        _, _, h, w = x.shape
        line = int((h * w * rate) ** 0.5 // 2)
        cy, cx = h // 2, w // 2
        cy_s, cy_e = max(0, cy - line), min(h, cy + line)
        cx_s, cx_e = max(0, cx - line), min(w, cx + line)
        mask[:, :, cy_s:cy_e, cx_s:cx_e] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        fft = fft * (1 - mask)
        ifft = torch.fft.ifft2(torch.fft.ifftshift(fft), norm="forward").real
        return torch.abs(ifft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self.fft_highpass(x, self.freq_nums)
        base_prompt = self.freq_proj(freq)
        B, C_t, Hp, Wp = base_prompt.shape
        flat = base_prompt.flatten(2).permute(0, 2, 1)

        p = self.lightweight_mlp(flat)
        p = self.shared_mlp(p)
        p = p.permute(0, 2, 1).view(B, -1, Hp, Wp)
        p = F.interpolate(p, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        return p


@MODELS.register_module()
class FreqPromptLocalMHACrossFusion(BaseModule):
    def __init__(self,
                 pts_in_dim: int = 128,
                 img_in_dim: int = 256,
                 out_dim: int = 64,
                 k_size: int = 5,
                 num_mha_layers: int = 4,
                 num_heads: int = 8,
                 patch_size: int = 16,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        assert k_size % 2 == 1
        assert out_dim % num_heads == 0

        self.k_size = k_size
        self.P = k_size * k_size
        self.num_mha_layers = num_mha_layers
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads

        self.prompt_gen = PromptGenerator(
            embed_dim=out_dim,
            patch_size=patch_size
        )

        self.img_proj = nn.Conv2d(img_in_dim, out_dim, kernel_size=1)

        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=out_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_mha_layers)
        ])

        self.q_proj = nn.Linear(pts_in_dim, out_dim)
        self.k_proj = nn.Linear(out_dim, out_dim)
        self.v_proj = nn.Linear(out_dim, out_dim)
        self.attn_out_proj = nn.Linear(out_dim, out_dim)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim)
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        half = k_size // 2
        offsets = torch.arange(-half, half + 1, dtype=torch.float32)
        oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
        offsets = torch.stack([ox, oy], dim=-1).reshape(1, 1, -1, 2)
        self.register_buffer('offsets_base', offsets)

    def forward(self,
                pts_feat: torch.Tensor,
                imgs: torch.Tensor,
                sam_feat: torch.Tensor,
                seeds_3d: torch.Tensor,
                img_metas: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
            fused_feat: (B, 64, N) —— 点级融合特征，用于 joint_tower
            img_enhanced_feat: (B, 64, H, W) —— 图像级增强特征，可用于 img_tower
        """
        B, C_pts, N = pts_feat.shape
        device = pts_feat.device

        # 1. SAM 特征上采样 + 投影
        sam_feat_upsampled = F.interpolate(sam_feat, size=imgs.shape[2:], mode='bilinear', align_corners=True)
        img_feat = self.img_proj(sam_feat_upsampled)  # (B, 64, H, W)

        # 2. 生成频率提示（图像级）
        freq_prompt = self.prompt_gen(imgs)  # (B, 64, H, W)

        # 3. 生成图像级增强特征（用于 img_tower）
        img_enhanced_feat = img_feat + freq_prompt  # (B, 64, H, W)

        # 4. 3D → 2D 投影
        pixels_2d_list = []
        for i in range(B):
            seed_3d = seeds_3d[i]
            meta = img_metas[i]
            xyz_depth = apply_3d_transformation(seed_3d, 'DEPTH', meta, reverse=True)
            depth2img = xyz_depth.new_tensor(meta['depth2img'])
            uvz = points_cam2img(xyz_depth, depth2img, True)
            uv = (uvz[..., :2] - 1).round()
            pixels_2d = coord_2d_transform(meta, uv, True)
            pixels_2d_list.append(pixels_2d)
        pixels_2d = torch.stack(pixels_2d_list, dim=0)

        H, W = img_feat.shape[2:]
        x_norm = 2.0 * pixels_2d[..., 0] / (W - 1) - 1.0
        y_norm = 2.0 * pixels_2d[..., 1] / (H - 1) - 1.0
        grid_base = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)

        offsets = self.offsets_base.to(device)
        offsets[..., 0] *= 2.0 / (W - 1)
        offsets[..., 1] *= 2.0 / (H - 1)
        grid = grid_base + offsets

        # 5. 采样局部特征（用于点级融合）
        local_feat = F.grid_sample(img_feat, grid, mode='bilinear', padding_mode='border', align_corners=True)
        local_feat = local_feat.permute(0, 2, 3, 1).contiguous()  # (B, N, P, 64)

        # 6. 局部频率提示注入（只一次）
        if self.num_mha_layers > 0:
            freq_sampled = F.grid_sample(freq_prompt, grid_base.expand(-1, -1, self.P, -1),
                                         mode='bilinear', padding_mode='border', align_corners=True)
            freq_sampled = freq_sampled.permute(0, 2, 3, 1).contiguous()
            local_feat = local_feat + freq_sampled

        # 7. 局部多头自注意力
        for i in range(self.num_mha_layers):
            feat_flat = local_feat.view(B * N, self.P, -1)
            attn_out, _ = self.mha_layers[i](feat_flat, feat_flat, feat_flat)
            local_feat = local_feat.view(B * N, self.P, -1) + attn_out
            local_feat = local_feat.view(B, N, self.P, -1)

        # 8. 点级交叉注意力
        Q = self.q_proj(pts_feat.permute(0, 2, 1))
        K = self.k_proj(local_feat)
        V = self.v_proj(local_feat)

        Q_flat = Q.view(B * N, 1, self.out_dim)
        K_flat = K.view(B * N, self.P, self.out_dim)
        V_flat = V.view(B * N, self.P, self.out_dim)

        Q_flat = Q_flat.view(B * N, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_flat = K_flat.view(B * N, self.P, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_flat = V_flat.view(B * N, self.P, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = (Q_flat @ K_flat.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = attn_weights.softmax(dim=-1)

        out_flat = attn_weights @ V_flat
        out_flat = out_flat.squeeze(2).transpose(1, 2).contiguous()
        out_flat = out_flat.view(B * N, self.out_dim)
        out_flat = self.attn_out_proj(out_flat)

        out = out_flat.view(B, N, self.out_dim)

        Q_residual = Q
        out = self.norm1(Q_residual + out)
        out = out + self.ffn(out)
        out = self.norm2(out)

        fused_feat = out.permute(0, 2, 1)  # (B, 64, N)

        # 返回点级融合特征 + 图像级增强特征
        return fused_feat, local_feat.mean(dim=2)