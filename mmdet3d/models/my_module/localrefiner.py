# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmengine.model import BaseModule
# from mmdet3d.registry import MODELS
# from typing import Optional

# def index_points(points, idx):
#     B = points.shape[0]
#     view_shape = list(idx.shape)
#     view_shape[1:] = [1] * (len(view_shape) - 1)
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1
#     batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
#     return points[batch_indices, idx, :]

# @MODELS.register_module()
# class LocalGeometricRefiner(BaseModule):
#     def __init__(self,
#                  k: int = 16,
#                  embed_dim: int = 256,
#                  tau: float = 5.0,
#                  init_cfg: Optional[dict] = dict(type='Kaiming', layer=['Linear', 'Conv1d'])):
#         super().__init__(init_cfg=init_cfg)
#         self.k = k
#         self.embed_dim = embed_dim
#         self.tau = tau

#         # --- 基础特征投影 ---
#         self.fc_in = nn.Sequential(
#             nn.Conv1d(embed_dim, embed_dim, 1), 
#             nn.BatchNorm1d(embed_dim), 
#             nn.ReLU(inplace=True), 
#             nn.Conv1d(embed_dim, embed_dim, 1)
#         )

#         # --- 局部几何 (KNN 阶段) ---
#         self.fc_delta = nn.Sequential(
#             nn.Linear(3, 64), nn.ReLU(inplace=True), 
#             nn.Linear(64, embed_dim)
#         )
#         self.fc_delta_1 = nn.Sequential(
#             nn.Linear(embed_dim * 2, embed_dim), 
#             nn.ReLU(inplace=True), 
#             nn.Linear(embed_dim, embed_dim)
#         )

#         # --- 实例几何 (方案 A: 唯一位置与相对坐标) ---
#         # 用于处理相对坐标 relative_xyz (B, N, 3)
#         self.fc_relative_xyz = nn.Sequential(
#             nn.Linear(3, 64), 
#             nn.ReLU(inplace=True),
#             nn.Linear(64, embed_dim)
#         )
#         # 用于处理实例中心坐标 mask_center (B, 3, N)
#         self.fc_mask_center = nn.Sequential(
#             nn.Conv1d(3, 64, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(64, embed_dim, 1)
#         )

#         # --- 绝对位置 (原始位置信息保留) ---
#         self.fc_delta_abs = nn.Sequential(
#             nn.Linear(3, 64), nn.ReLU(inplace=True), 
#             nn.Linear(64, embed_dim)
#         )

#         # --- 融合与输出 ---
#         # 拼接: 1.阶段一特征, 2.实例平均特征, 3.相对几何特征, 4.中心点特征, 5.绝对位置特征
#         self.fuse_mlp = nn.Sequential(
#             nn.Conv1d(4 * embed_dim, embed_dim, 1), 
#             nn.BatchNorm1d(embed_dim), 
#             nn.ReLU(inplace=True), 
#             nn.Conv1d(embed_dim, embed_dim, 1), 
#             nn.BatchNorm1d(embed_dim), 
#             nn.ReLU(inplace=True)
#         )
#         self.out_mlp = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim), 
#             nn.ReLU(inplace=True), 
#             nn.Linear(embed_dim, embed_dim)
#         )

#     def forward(self,
#                 seeds_3d: torch.Tensor,   # (B, N, 3)
#                 fused_feat: torch.Tensor, # (B, C, N)
#                 seed_masks: torch.Tensor  # (B, N) 包含实例 ID
#                 ) -> torch.Tensor:
       
#         B, C, N = fused_feat.shape
        
#         # 1. KNN 寻找局部最近邻
#         dist_mat = torch.norm(seeds_3d.unsqueeze(2) - seeds_3d.unsqueeze(1), dim=-1)
#         # 构造一个掩码：如果 seed_masks 不相等，则距离 +inf
#         mask_diff = (seed_masks.unsqueeze(2) != seed_masks.unsqueeze(1))
#         dist_mat = dist_mat + mask_diff * 0.5

#         _, knn_idx = torch.topk(dist_mat, k=self.k, dim=-1, largest=False)
#         knn_xyz = index_points(seeds_3d, knn_idx)
#         knn_dist = dist_mat.gather(-1, knn_idx)
#         invalid_knn_mask = (knn_dist >= 2.0).unsqueeze(-1) # (B, N, K, 1)
#         # --- 阶段一：局部特征增强 ---
#         pre_stage1 = fused_feat
#         kpt_feature_proj = self.fc_in(fused_feat)
#         pos_enc = seeds_3d.unsqueeze(2) - knn_xyz # 局部相对位移
#         pos_enc_feat = self.fc_delta(pos_enc)
#         pos_enc_feat = pos_enc_feat.masked_fill(invalid_knn_mask, 0.0) # 剔除噪声几何
#         knn_feature = index_points(fused_feat.transpose(1, 2), knn_idx)
#         knn_feature_enhanced = self.fc_delta_1(torch.cat([knn_feature, pos_enc_feat], dim=-1))
       
#         query = kpt_feature_proj.transpose(1, 2).unsqueeze(2).expand(-1, -1, self.k, -1)
#         sim = F.cosine_similarity(query, knn_feature_enhanced, dim=-1)
#         weights = F.softmax(sim / self.tau, dim=-1)
        
#         kpt_feature_stage1 = torch.matmul(weights.unsqueeze(2), knn_feature_enhanced).squeeze(2)
#         kpt_feature_stage1 = F.relu(kpt_feature_stage1 + pre_stage1.transpose(1, 2)) # (B, N, C)

#         # --- 阶段二：方案 A - 实例感知与唯一位置建模 ---
#         max_id = int(seed_masks.max().item())
#         one_hot_masks = F.one_hot(seed_masks.long(), num_classes=max_id + 1).float() # (B, N, ID)
        
#         # A.1 计算实例几何中心 (Centroid)
#         instance_count = one_hot_masks.sum(dim=1, keepdim=True) + 1e-6
#         mask_centers_pool = torch.bmm(seeds_3d.transpose(1, 2), one_hot_masks) / instance_count # (B, 3, ID)
#         mask_centers_pool = mask_centers_pool.detach()
#         # A.2 将中心点广播回点级 (唯一位置信息)
#         kpt_mask_center = torch.bmm(mask_centers_pool, one_hot_masks.transpose(1, 2)) # (B, 3, N)
#         center_feat = self.fc_mask_center(kpt_mask_center) # (B, C, N)
        
#         # A.3 计算点到中心的相对位移 (唯一几何形状)
#         relative_xyz = seeds_3d - kpt_mask_center.transpose(1, 2) # (B, N, 3)
#         relative_geo_feat = self.fc_relative_xyz(relative_xyz).transpose(1, 2) # (B, C, N)

#         # A.4 实例级语义平均 (原有逻辑保留)
#         instance_feat_sum = torch.bmm(kpt_feature_stage1.transpose(1, 2), one_hot_masks)
#         instance_mean_feat = instance_feat_sum / instance_count
#         kpt_global_sem_rep = torch.bmm(instance_mean_feat, one_hot_masks.transpose(1, 2)) # (B, C, N)

#         # A.5 绝对位置编码 (保留原始位置先验)
#         # pos_enc_abs = self.fc_delta_abs(seeds_3d).transpose(1, 2) # (B, C, N)

#         # --- 阶段三：多维特征融合 ---
#         # 拼接: 1.局部语义, 2.实例语义, 3.相对几何, 4.唯一位置中心, 5.绝对位置
#         cat_feat = torch.cat([
#             kpt_feature_stage1.transpose(1, 2),
#             kpt_global_sem_rep,
#             relative_geo_feat,
#             center_feat
#         ], dim=1)
       
#         fused_out = self.fuse_mlp(cat_feat)
#         fused_out = F.relu(fused_out.transpose(1, 2) + kpt_feature_stage1)
       
#         # --- 阶段四：输出投影 ---
#         out = self.out_mlp(fused_out)
#         return F.relu(fused_out + out).transpose(1, 2).contiguous()



import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from typing import Optional


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate pairwise squared distances without the extra sqrt in cdist."""
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src**2, dim=-1, keepdim=True)
    dist += torch.sum(dst**2, dim=-1).unsqueeze(1)
    return dist.clamp_min_(0.0)

@MODELS.register_module()
class LocalGeometricRefiner(BaseModule):
    def __init__(self,
                 embed_dim: int = 256,
                 k: int = 16,
                 tau: float = 5.0,
                 radius: float = 0.2,
                 init_cfg: Optional[dict] = dict(type='Kaiming', layer=['Linear', 'Conv1d'])):
        super().__init__(init_cfg=init_cfg)

        self.k = k
        self.embed_dim = embed_dim
        self.tau = tau
        self.radius = radius

        self.query_proj = nn.Conv1d(embed_dim, embed_dim, 1)

        self.geo_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.key_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.value_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.fuse_mlp = nn.Sequential(
            nn.Conv1d(embed_dim * 4 + 3, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self,
                seeds_3d: torch.Tensor,      # (B, M, 3)
                fused_feat: torch.Tensor     # (B, C, M)
                ) -> torch.Tensor:

        B, C, M = fused_feat.shape
        k = min(self.k, M)

        seed_feat = fused_feat.transpose(1, 2).contiguous()  # (B, M, C)
        query_feat = self.query_proj(fused_feat).transpose(1, 2).contiguous()

        # Radius-constrained KNN among GALF-fused seed points.
        dist_mat = square_distance(seeds_3d, seeds_3d)
        knn_dist_sq, knn_idx = torch.topk(dist_mat, k=k, dim=-1, largest=False)
        knn_xyz = index_points(seeds_3d, knn_idx)            # (B, M, K, 3)
        knn_feat = index_points(seed_feat, knn_idx)          # (B, M, K, C)

        rel_xyz = knn_xyz - seeds_3d.unsqueeze(2)
        rel_dist = torch.sqrt(knn_dist_sq.clamp_min(1e-12)).unsqueeze(-1)
        geo_feat = self.geo_encoder(torch.cat([rel_xyz, rel_dist], dim=-1))

        pair_feat = torch.cat([knn_feat, geo_feat], dim=-1)
        key_feat = self.key_proj(pair_feat)
        value_feat = self.value_proj(pair_feat)

        sim = F.cosine_similarity(query_feat.unsqueeze(2), key_feat, dim=-1)
        valid_mask = rel_dist.squeeze(-1) < self.radius
        sim = (sim / self.tau).masked_fill(~valid_mask, torch.finfo(sim.dtype).min)

        weights = F.softmax(sim, dim=-1) * valid_mask.to(sim.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        local_context = (weights.unsqueeze(-1) * value_feat).sum(dim=2)

        masked_geo = geo_feat.masked_fill(
            ~valid_mask.unsqueeze(-1), torch.finfo(geo_feat.dtype).min)
        geo_context = masked_geo.max(dim=2).values
        geo_context = torch.where(valid_mask.any(dim=2, keepdim=True),
                                  geo_context,
                                  torch.zeros_like(geo_context))

        global_context = seed_feat.mean(dim=1, keepdim=True).expand(-1, M, -1)
        context = torch.cat(
            [seed_feat, local_context, geo_context, seeds_3d, global_context],
            dim=-1)

        refined_feat = self.fuse_mlp(context.transpose(1, 2).contiguous())
        return refined_feat.contiguous()


@MODELS.register_module()
class GraphOnlyLocalAggregator(BaseModule):
    def __init__(self,
                 support_in_dim: int = 256,
                 embed_dim: int = 256,
                 k: int = 16,
                 dist_threshold: float = 0.2,
                 init_cfg: Optional[dict] = dict(type='Kaiming', layer=['Linear', 'Conv1d'])):
        super().__init__(init_cfg=init_cfg)

        self.k = k
        self.dist_threshold = dist_threshold

        self.query_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.support_proj = nn.Conv1d(support_in_dim, embed_dim, 1)

        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self,
                query_xyz: torch.Tensor,
                query_feat: torch.Tensor,
                support_xyz: torch.Tensor,
                support_feat: torch.Tensor) -> torch.Tensor:

        k = min(self.k, support_xyz.shape[1])
        dist_threshold_sq = self.dist_threshold * self.dist_threshold

        query_feat = self.query_proj(query_feat).transpose(1, 2).contiguous()
        support_feat = self.support_proj(support_feat).transpose(1, 2).contiguous()

        dist_mat = square_distance(query_xyz, support_xyz)
        knn_dist_sq, knn_idx = torch.topk(dist_mat, k=k, dim=-1, largest=False)
        knn_xyz = index_points(support_xyz, knn_idx)
        knn_feat = index_points(support_feat, knn_idx)

        rel_xyz = query_xyz.unsqueeze(2) - knn_xyz
        query_expand = query_feat.unsqueeze(2).expand(-1, -1, k, -1)
        edge_input = torch.cat(
            [query_expand, knn_feat - query_expand, rel_xyz], dim=-1)
        edge_feat = self.edge_mlp(edge_input)

        valid_mask = (knn_dist_sq < dist_threshold_sq).unsqueeze(-1)
        edge_feat = edge_feat.masked_fill(~valid_mask, 0.0)
        local_feat = edge_feat.max(dim=2).values

        fused = torch.cat([query_feat, local_feat], dim=-1).transpose(1, 2).contiguous()
        fused = self.fuse_mlp(fused).transpose(1, 2).contiguous()
        fused = F.relu(fused + query_feat)
        out = F.relu(fused + self.out_mlp(fused))

        return out.transpose(1, 2).contiguous()
