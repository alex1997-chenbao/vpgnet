import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from typing import Tuple

# --- PatchEmbed2 ---
class PatchEmbed2(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=64):
        super().__init__()
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)  # (B, embed_dim, H_out, W_out)

# --- FrequencyPromptNeck (修复版) ---
@MODELS.register_module()
class FrequencyPromptNeck(BaseModule):
    def __init__(self,
                 sam_embed_dim=256,      # 输入特征通道数（当前是256）
                 output_dim=64,
                 num_layers=4,
                 num_heads=8,
                 mlp_ratio=4.0,
                 patch_size=16,
                 freq_rate=0.25,
                 init_cfg=None,
                 **kwargs):  # 捕获 config 中多余参数
        super().__init__(init_cfg=init_cfg)
        if kwargs:
            print(f"[FrequencyPromptNeck] Ignored unused args: {list(kwargs.keys())}")

        hidden_dim = output_dim  # 频率支路隐藏维度

        # 主干降维：256 → output_dim
        self.sam_downsampler = nn.Linear(sam_embed_dim, output_dim)
        # 语义 embedding：256 → hidden_dim
        self.embedding_gen = nn.Linear(sam_embed_dim, hidden_dim)

        # 频率特征提取 Conv
        self.prompt_gen = PatchEmbed2(patch_size=patch_size,
                                      in_chans=3,
                                      embed_dim=hidden_dim)

        # 融合 MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            self._make_block(dim=output_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(output_dim)
        self.freq_rate = freq_rate

    def _make_block(self, dim, num_heads, mlp_ratio):
        mlp_hidden = int(dim * mlp_ratio)
        return nn.ModuleDict({
            'norm1': nn.LayerNorm(dim),
            'attn': nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True),
            'norm2': nn.LayerNorm(dim),
            'mlp': nn.Sequential(
                nn.Linear(dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, dim),
            )
        })

    def fft_high_pass(self, x: torch.Tensor, rate: float) -> torch.Tensor:
        """高通滤波，返回幅度图"""
        w, h = x.shape[-2:]
        mask = torch.zeros_like(x, device=x.device)
        line = int((w * h * rate) ** 0.5 // 2 * 1.2)
        cy, cx = h // 2, w // 2
        mask[:, :, cy-line:cy+line, cx-line:cx+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        fft = fft * (1 - mask)
        fft_hires = torch.fft.ifftshift(fft)
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        return torch.abs(inv)

    def forward(self, img: torch.Tensor, sam_feat: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        img:      (B, 3, H_img, W_img)   # 原始图像，尺寸不固定
        sam_feat: (B, 256, H_f, W_f)    # 当前是 (B, 256, 72, 72)
        """
        B, C_sam, H_f, W_f = sam_feat.shape
        assert C_sam == 256, f"Expected 256 channels, got {C_sam}"

        # 1. 处理 sam_feat (CHW → NCHW → flatten to sequence)
        sam_flat = sam_feat.flatten(2).permute(0, 2, 1)  # (B, H_f*W_f, 256)

        # 主干降维
        backbone_flat = self.sam_downsampler(sam_flat)      # (B, N, output_dim)

        # 语义 embedding
        embedding = self.embedding_gen(sam_flat)           # (B, N, hidden_dim)

        # 2. 频率支路
        high_freq_img = self.fft_high_pass(img, self.freq_rate)     # (B, 3, H_img, W_img)
        freq_map = self.prompt_gen(high_freq_img)                   # (B, hidden_dim, H_out, W_out)

        # 关键：将频率特征上采样/插值到 sam_feat 的空间尺寸 (H_f, W_f)
        freq_map = F.interpolate(
            freq_map,
            size=(H_f, W_f),
            mode='bilinear',
            align_corners=False
        )  # (B, hidden_dim, H_f, W_f)

        freq_flat = freq_map.flatten(2).permute(0, 2, 1)  # (B, N, hidden_dim)

        # 3. 融合生成 prompt
        prompt_flat = self.fusion_mlp(embedding + freq_flat)  # (B, N, hidden_dim → hidden_dim)

        # 注意：prompt_flat 是 hidden_dim，但 backbone_flat 是 output_dim
        # 如果 hidden_dim != output_dim，需要投影
        if prompt_flat.shape[-1] != backbone_flat.shape[-1]:
            # 添加一个投影层（可学习）
            if not hasattr(self, 'prompt_proj'):
                self.prompt_proj = nn.Linear(prompt_flat.shape[-1], backbone_flat.shape[-1]).to(prompt_flat.device)
            prompt_flat = self.prompt_proj(prompt_flat)

        # 4. 多层残差注入 + Transformer
        x = backbone_flat
        for block in self.transformer_blocks:
            x = x + prompt_flat                                    # 注入频率提示
            attn_out, _ = block['attn'](block['norm1'](x), block['norm1'](x), block['norm1'](x))
            x = x + attn_out
            x = x + block['mlp'](block['norm2'](x))

        x = self.final_norm(x)  # (B, N, output_dim)

        # 恢复为特征图
        out = x.permute(0, 2, 1).view(B, -1, H_f, W_f)  # (B, output_dim, H_f, W_f)

        return out  # MMDet3D neck 标准输出格式