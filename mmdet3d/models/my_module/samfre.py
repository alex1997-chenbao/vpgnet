# mmdet3d/models/fusion_modules/frequency_prompt_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from typing import Optional, Tuple

@MODELS.register_module()
class FrequencyPromptAdapter(BaseModule):
    """
    像素级频率特征 Adapter (Haar 小波版)
    """
    def __init__(self,
                 scale_factor: int = 4, 
                 embed_dim: int = 256,
                 depth: int = 4,
                 init_cfg: Optional[dict] = dict(type='Kaiming', layer=['Conv2d']),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.depth = depth
        bottleneck_dim = embed_dim // scale_factor
        self.in_channels = 12

        # 1. 输入压缩：12通道（4个子带*3通道）→ bottleneck
        self.mlp_tune_in = nn.Sequential(
            nn.Conv2d(self.in_channels, bottleneck_dim, kernel_size=1),
            nn.GELU()
        )

        # 2. depth 层轻量提炼
        self.lightweight_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=1),
                nn.GELU()
            ) for _ in range(depth)
        ])

        # 3. 升维回 SAM 通道数
        self.mlp_up = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1)

        # **关键修改 1：使用 InstanceNorm2d 替换 LayerNorm**
        # InstanceNorm2d 只需要通道数 (C) 即可初始化，无需 H 和 W。
        # 它在 H 和 W 上计算统计量，对每个实例（Batch内的每个样本）独立归一化。
        self.norm_pi = nn.InstanceNorm2d(embed_dim, affine=True) 
        
    def forward(self,
                x_img: torch.Tensor,      # (B, 3, H, W) 原始图像
                f_sam: torch.Tensor) -> torch.Tensor:  # (B, C, H', W') SAM 特征
        """
        Returns:
            f_enhanced: (B, C, H, W) 频率和 SAM 融合后插值到图像尺寸的特征
        """
        B, C, H, W = x_img.shape
        original_size = (H, W)
        # # 1. 提取 Haar 小波频率特征
        # F_i, original_size = self._haar_wavelet(x_img) 
        
        # # 确定中低分辨率 (H/2, W/2) 作为融合目标尺寸
        # low_res_size = F_i.shape[2:] 
        # target_high_res = original_size

        # # 2. 生成频率 Prompt 图 Pⁱ
        # x = self.mlp_tune_in(F_i)
        # for layer in self.lightweight_mlps:
        #     x = layer(x)
        # P_i = self.mlp_up(x)  # P_i: (B, embed_dim, H_p, W_p)

        # # 3. **关键修改 2：将 SAM 特征插值到频率特征尺寸**
        # f_sam_low = F.interpolate(
        #     f_sam, size=low_res_size, mode='bilinear', align_corners=False
        # )
        
        # # --- 打印统计量 (保留不变) ---
        # # print(f"P_i 形状: {P_i.shape}")
        # # print(f"f_sam_low 形状: {f_sam_low.shape}")
        # # print("-" * 40)
        
        # # P_i_mean = torch.mean(P_i)
        # # P_i_variance = torch.var(P_i, unbiased=True) 
        # # print("--- P_i 统计量 (归一化前) ---")
        # # print(f"  全局均值: {P_i_mean.item():.6f}")
        # # print(f"  全局方差: {P_i_variance.item():.6f}")
        
        # # f_sam_low_mean = torch.mean(f_sam_low)
        # # f_sam_low_variance = torch.var(f_sam_low, unbiased=True)
        # # print("\n--- f_sam_low 统计量 ---")
        # # print(f"  全局均值: {f_sam_low_mean.item():.6f}")
        # # print(f"  全局方差: {f_sam_low_variance.item():.6f}")
        # # print("-" * 40)
        
        # # # 4. **关键修改 3：对 P_i 进行 Instance Norm 归一化后加法融合**
        # P_i_norm = self.norm_pi(P_i) 
        # # 4.1 打印统计量 (归一化后)
        # P_i_norm_mean = torch.mean(P_i_norm)
        # P_i_norm_variance = torch.var(P_i_norm, unbiased=True)
        # print("--- P_i_norm 统计量 (归一化后) ---")
        # print(f"  全局均值: {P_i_norm_mean.item():.6f}")
        # print(f"  全局方差: {P_i_norm_variance.item():.6f}")
        # print("-" * 40)
        
        # f_fused_low = f_sam_low + P_i_norm 

        # 5. **关键修改 4：将融合后的结果插值回原始图像尺寸**
        f_enhanced = F.interpolate(
            f_sam, size=original_size, mode='bilinear', align_corners=False
        )

        return f_enhanced

    def _haar_wavelet(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        纯 PyTorch 一级 Haar 小波分解（等价于 db1）
        """
        B, C, H, W = x.shape
        original_size = (H, W)

        # 自动 padding 到 2 的倍数（reflect 边界最自然）
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
            _, _, H, W = x.shape 

        # Haar 小波分解：使用所有四个子带 (LL, LH, HL, HH)
        LL = x[:, :, 0::2, 0::2] + x[:, :, 1::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]
        LH = x[:, :, 0::2, 0::2] - x[:, :, 1::2, 0::2] + x[:, :, 0::2, 1::2] - x[:, :, 1::2, 1::2]
        HL = x[:, :, 0::2, 0::2] + x[:, :, 1::2, 0::2] - x[:, :, 0::2, 1::2] - x[:, :, 1::2, 1::2]
        HH = x[:, :, 0::2, 0::2] - x[:, :, 1::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]

        F_i = torch.cat([LL, LH, HL, HH], dim=1) 

        return F_i, original_size