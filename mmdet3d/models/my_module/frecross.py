# # mmdet3d/models/fusion_modules/full_local_cross_attention.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmengine.model import BaseModule
# from mmdet3d.registry import MODELS
# from typing import List, Dict, Tuple

# # 导入 MMDetection3D 官方依赖项
# from mmdet3d.structures import points_cam2img
# from mmdet3d.models.layers.fusion_layers.vote_fusion import apply_3d_transformation, coord_2d_transform

# @MODELS.register_module()
# class FullLocalCrossAttention(BaseModule):
#     def __init__(self,
#                  pts_in_dim: int = 128,  # 3D 种子点特征维度
#                  img_in_dim: int = 256,  # 2D SAM/Freq 融合特征维度
#                  out_dim: int = 256,
#                  k_size: int = 5,
#                  init_cfg=None):
#         super().__init__(init_cfg=init_cfg)

#         assert k_size % 2 == 1, f"k_size 必须是奇数，当前为 {k_size}"
#         self.k_size = k_size
#         self.P = k_size * k_size
#         # 注意力缩放因子 (scale)
#         self.scale = out_dim ** -0.5

#         # 投影头 (用于局部交叉注意力)
#         self.query_proj = nn.Conv1d(pts_in_dim, out_dim, kernel_size=1)
#         self.key_proj = nn.Conv2d(img_in_dim, out_dim, kernel_size=1)
#         self.value_proj = nn.Conv2d(img_in_dim, out_dim, kernel_size=1)

#         # FFN (前馈网络)
#         self.ffn = nn.Sequential(
#             nn.Conv1d(out_dim, out_dim * 4, kernel_size=1),
#             nn.GELU(),
#             nn.Conv1d(out_dim * 4, out_dim, kernel_size=1)
#         )

#         self.norm1 = nn.LayerNorm(out_dim)
#         self.norm2 = nn.LayerNorm(out_dim)

#         # 预生成 k x k 局部偏移（用于 grid_sample）
#         half = k_size // 2
#         offsets = torch.arange(-half, half + 1, dtype=torch.float32)
#         oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
#         # (1, 1, P, 2)
#         offsets = torch.stack([ox, oy], dim=-1).reshape(1, 1, -1, 2)
#         self.register_buffer('offsets_base', offsets)
#         self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / self.scale)))
#     def forward(self,
#                 pts_feat: torch.Tensor,      # (B, C_pts, N) 3D 种子点特征
#                 img_feat: torch.Tensor,      # (B, C_img, H, W) SAM/Freq 融合后的 2D 特征
#                 seeds_3d: torch.Tensor,      # (B, N, 3) 3D 种子点坐标
#                 metas: List[Dict],
#                 mask_labels: torch.Tensor,    # (B, H, W) 新增输入
#                 imgs: List[torch.Tensor],    # (B, 3, H, W) 新增输入
#                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # 增加返回 seed_masks
        
#         B, C_pts, N = pts_feat.shape
#         _, C_img, H, W = img_feat.shape
#         device = img_feat.device
        
#         Q = self.query_proj(pts_feat).permute(0, 2, 1)
#         img_metas = [item.metainfo for item in metas]

#         # --- 2. 3D -> 2D 投影与对齐 ---
#         pixels_2d_list = []
#         for i in range(B):
#             seed_3d_depth = seeds_3d[i] 
#             img_meta = img_metas[i]
            
#             xyz_depth = apply_3d_transformation(seed_3d_depth, 'DEPTH', img_meta, reverse=True)
#             depth2img = xyz_depth.new_tensor(img_meta['depth2img'])
#             uvz_origin = points_cam2img(xyz_depth, depth2img, True)
#             uv_origin = (uvz_origin[..., :2] - 1).round()
            
#             # 物理翻转补偿
#             if img_meta.get('flip', False) or img_meta.get('pcd_horizontal_flip', False):
#                 img_w = img_meta['img_shape'][1] 
#                 uv_origin[..., 0] = (img_w - 1) - uv_origin[..., 0]
            
#             # Pipeline 补偿 (Resize/Pad)
#             pixels_2d_sample = coord_2d_transform(img_meta, uv_origin, True)
#             pixels_2d_list.append(pixels_2d_sample)

#         pixels_2d = torch.stack(pixels_2d_list, dim=0) # (B, N, 2)

#         # --- 3. 采样逻辑 (此处必须先计算采样结果) ---
#         x_norm = (2.0 * pixels_2d[..., 0].float() / (W - 1) - 1.0).clamp(-1, 1)
#         y_norm = (2.0 * pixels_2d[..., 1].float() / (H - 1) - 1.0).clamp(-1, 1)
#         grid_base = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)

#         # sampled_mask = F.grid_sample(
#         #     mask_labels.unsqueeze(1).float(), grid_base, 
#         #     mode='nearest', padding_mode='zeros', align_corners=True
#         # ) 
#         # seed_masks = sampled_mask.view(B, N) # 模型采样到的 ID

#         # --- 4. [新增] 模型内实时可视化验证 (修正后的逻辑) ---
#         # 建议增加频率控制，否则每轮写磁盘会极慢

#         # if 1:  # 调试开关
#         #     debug_dir = 'debug_projection_results'
#         #     import os, cv2, numpy as np
#         #     os.makedirs(debug_dir, exist_ok=True)

#         #     def get_vis_color(mid):
#         #         if mid <= 0: return (0, 0, 0)
#         #         # 增加随机种子偏移，并限制颜色在较亮范围，确保颜色“鲜艳”
#         #         np.random.seed(int(mid) + 100) 
#         #         return [int(c) for c in np.random.randint(100, 255, 3)]

#         #     for i in range(B):
#         #         file_idx = os.path.splitext(os.path.basename(img_metas[i]['img_path']))[0]
                
#         #         # 1. 还原底图 (增加对比度增强)
#         #         vis_img = imgs[i].detach().cpu().numpy().transpose(1, 2, 0)
#         #         # 简单的线性拉伸让底图暗一点，突出 Mask 和点
#         #         vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-6) * 180 
#         #         vis_img = vis_img.astype(np.uint8).copy()

#         #         # 2. 绘制 Mask 层 (减小透明度，让 Mask 颜色更深)
#         #         mask_2d = mask_labels[i].detach().cpu().numpy().astype(np.uint16)
#         #         mask_overlay = np.zeros_like(vis_img)
#         #         for mid in np.unique(mask_2d):
#         #             if mid == 0: continue
#         #             mask_overlay[mask_2d == mid] = get_vis_color(mid)
                
#         #         # alpha=0.4 (原图权重), beta=0.8 (Mask权重)，让 Mask 覆盖感更强
#         #         vis_img = cv2.addWeighted(vis_img, 0.4, mask_overlay, 0.8, 0)

#         #         # 3. 绘制采样点 (加大半径，加深边缘)
#         #         uv_draw = pixels_2d[i].detach().cpu().numpy()
#         #         s_ids = seed_masks[i].detach().cpu().numpy()
                
#         #         for pt_idx in range(len(uv_draw)):
#         #             u, v = int(round(uv_draw[pt_idx][0])), int(round(uv_draw[pt_idx][1]))
#         #             if 0 <= u < vis_img.shape[1] and 0 <= v < vis_img.shape[0]:
#         #                 p_color = get_vis_color(s_ids[pt_idx])
                        
#         #                 # 绘制实心圆点：半径增大到 4
#         #                 cv2.circle(vis_img, (u, v), 4, p_color, -1)
#         #                 # 绘制白色外边缘：增加厚度到 2，形成强烈对比
#         #                 cv2.circle(vis_img, (u, v), 4, (255, 255, 255), 2)
#         #                 # 绘制最外层黑色细圈：让点在亮色背景下也清晰
#         #                 cv2.circle(vis_img, (u, v), 5, (0, 0, 0), 1)

#         #         flip_tag = "Flip" if img_metas[i].get('flip') else "NoFlip"
#         #         cv2.imwrite(os.path.join(debug_dir, f"in_model_{file_idx}_{flip_tag}.jpg"), vis_img)
#         # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#         # 4. 生成 k×k 局部采样网格
#         offsets = self.offsets_base.clone().to(device)
#         offsets[..., 0] *= 2.0 / (W - 1)
#         offsets[..., 1] *= 2.0 / (H - 1)
#         grid = grid_base + offsets  # (B, N, P, 2)
        
#         # 5. K/V 投影 and 采样原始 2D 特征 (V_raw)
#         K_map = self.key_proj(img_feat)
#         V_map = self.value_proj(img_feat)
        
#         # 采样原始 2D 特征：仅采样中心点 (P=1) 处的特征 (用于返回)
#         grid_center = grid_base.expand(-1, -1, 1, -1) 
#         V_raw = F.grid_sample(img_feat, grid_center, mode='bilinear', padding_mode='border', align_corners=True)
#         img_feat_sampled = V_raw.squeeze(2) # (B, C_img, N)
        
#         # 6. grid_sample 采样 kxk 局部特征 (K/V)
#         K_local = F.grid_sample(K_map, grid, mode='bilinear', padding_mode='border', align_corners=True)
#         V_local = F.grid_sample(V_map, grid, mode='bilinear', padding_mode='border', align_corners=True)
#         K_local = K_local.permute(0, 2, 3, 1)  # (B, N, P, C_out)
#         V_local = V_local.permute(0, 2, 3, 1) # (B, N, P, C_out)

#         # 7. 局部交叉注意力
#         attn = (Q.unsqueeze(2) @ K_local.transpose(-2, -1)).squeeze(2) * self.logit_scale.exp()
#         attn_weights = attn.softmax(dim=-1)

#         # 加权求和 V_local
#         out = (attn_weights.unsqueeze(-1) * V_local).sum(dim=2)  # (B, N, C_out)

#         # 8. 残差 + FFN
#         out = self.norm1(Q + out) # 残差连接
#         out = out + self.ffn(out.permute(0, 2, 1)).permute(0, 2, 1) # FFN
#         out = self.norm2(out)

#         # 返回融合特征、采样的 2D 特征、以及新采样的 seed_masks
#         fused_feat = out.permute(0, 2, 1) # (B, out_dim, N)
#         return fused_feat,None


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from typing import List, Dict, Tuple

from mmdet3d.structures import points_cam2img
from mmdet3d.models.layers.fusion_layers.vote_fusion import (
    apply_3d_transformation,
    coord_2d_transform,
)


@MODELS.register_module()
class FullLocalCrossAttention(BaseModule):
    def __init__(self,
                 pts_in_dim: int = 128,
                 img_in_dim: int = 256,
                 out_dim: int = 256,
                 k_size: int = 5,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert k_size % 2 == 1, f'k_size must be odd, got {k_size}'
        self.k_size = k_size
        self.P = k_size * k_size
        self.out_dim = out_dim

        # Standard attention scale: 1 / sqrt(d)
        self.scale = out_dim ** -0.5
        # Learnable temperature multiplier, initialized to 1.0
        self.logit_scale = nn.Parameter(torch.zeros(1))

        # 3D query projection
        self.query_proj = nn.Conv1d(pts_in_dim, out_dim, kernel_size=1)

        # Merge K/V projection into one Linear
        self.kv_proj = nn.Linear(img_in_dim, out_dim * 2)

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv1d(out_dim, out_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(out_dim * 4, out_dim, kernel_size=1)
        )

        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # Precompute k x k offsets for local grid sampling
        half = k_size // 2
        offsets = torch.arange(-half, half + 1, dtype=torch.float32)
        oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
        offsets = torch.stack([ox, oy], dim=-1).reshape(1, 1, -1, 2)  # (1, 1, P, 2)
        self.register_buffer('offsets_base', offsets, persistent=False)

    def forward(self,
                pts_feat: torch.Tensor,      # (B, C_pts, N)
                img_feat: torch.Tensor,      # (B, C_img, H, W)
                seeds_3d: torch.Tensor,      # (B, N, 3)
                metas: List[Dict],
                mask_labels: torch.Tensor = None,
                imgs: List[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, _, N = pts_feat.shape
        _, _, H, W = img_feat.shape

        # Q: (B, N, out_dim)
        Q = self.query_proj(pts_feat).transpose(1, 2).contiguous()

        # Compatible with both Det3DDataSample and plain dict meta
        img_metas = [
            item.metainfo if hasattr(item, 'metainfo') else item
            for item in metas
        ]

        # 1. Project 3D seeds to image plane
        pixels_2d_list = []
        for i in range(B):
            seed_3d_depth = seeds_3d[i]
            img_meta = img_metas[i]

            xyz_depth = apply_3d_transformation(
                seed_3d_depth, 'DEPTH', img_meta, reverse=True)

            depth2img = xyz_depth.new_tensor(img_meta['depth2img'])
            uvz_origin = points_cam2img(xyz_depth, depth2img, with_depth=True)

            # Keep sub-pixel precision for bilinear sampling
            uv_origin = uvz_origin[..., :2] - 1.0

            # Physical flip compensation
            if img_meta.get('flip', False) or img_meta.get('pcd_horizontal_flip', False):
                img_w = img_meta['img_shape'][1]
                uv_origin[..., 0] = (img_w - 1) - uv_origin[..., 0]

            # Pipeline transform compensation (resize / pad / crop)
            pixels_2d_sample = coord_2d_transform(img_meta, uv_origin, True)
            pixels_2d_list.append(pixels_2d_sample)

        pixels_2d = torch.stack(pixels_2d_list, dim=0)  # (B, N, 2)

        img_hw = img_feat.new_tensor([
            item.get('img_shape', item.get('batch_input_shape', (H, W)))[:2]
            for item in img_metas
        ],
                                     dtype=torch.float32)
        img_h = img_hw[:, 0].view(B, 1)
        img_w = img_hw[:, 1].view(B, 1)

        # Normalize by the resized image shape instead of the feature map shape.
        x_denom = (img_w - 1).clamp_min(1.0)
        y_denom = (img_h - 1).clamp_min(1.0)
        x_norm = 2.0 * pixels_2d[..., 0].float() / x_denom - 1.0
        y_norm = 2.0 * pixels_2d[..., 1].float() / y_denom - 1.0

        grid_base = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # (B, N, 1, 2)

        # 3. Build k x k local sampling grid
        scale_xy = torch.stack(
            [2.0 / x_denom.clamp_min(1.0), 2.0 / y_denom.clamp_min(1.0)],
            dim=-1).unsqueeze(2)

        offsets = self.offsets_base * scale_xy
        grid = (grid_base + offsets).clamp(-1.0, 1.0)  # (B, N, P, 2)

        # 4. Sample local raw 2D features once
        # Output: (B, C_img, N, P)
        img_local = F.grid_sample(
            img_feat,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        # Center feature from local window, no second grid_sample needed
        img_feat_sampled = img_local[..., self.P // 2]  # (B, C_img, N)

        # Prepare for Linear projection: (B, N, P, C_img)
        img_local = img_local.permute(0, 2, 3, 1).contiguous()

        # 5. Project local 2D features to K/V
        kv = self.kv_proj(img_local)  # (B, N, P, 2*out_dim)
        K_local, V_local = kv.chunk(2, dim=-1)  # each: (B, N, P, out_dim)

        # 6. Local cross-attention
        attn_scale = self.scale * self.logit_scale.exp()
        attn = torch.matmul(
            Q.unsqueeze(2),                    # (B, N, 1, D)
            K_local.transpose(-2, -1)         # (B, N, D, P)
        ).squeeze(2) * attn_scale             # (B, N, P)

        attn_weights = attn.softmax(dim=-1)

        out = (attn_weights.unsqueeze(-1) * V_local).sum(dim=2)  # (B, N, D)

        # 7. Residual + FFN
        out = self.norm1(Q + out)
        out = out + self.ffn(out.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        out = self.norm2(out)

        fused_feat = out.transpose(1, 2).contiguous()  # (B, out_dim, N)

        return fused_feat, img_feat_sampled
