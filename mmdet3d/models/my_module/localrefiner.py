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
