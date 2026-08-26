import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .unet import UNet3D


class POCA_NET_UNET(nn.Module):
    r"""
    Reconstructs voxel-wise scattering densities from PoCA point clouds.

    The model consists of two stages. First, the spatial and scattering
    features of individual PoCA points are encoded using a shared point
    MLP. The resulting point features are then aggregated into a regular
    voxel grid using a learned per-point attention score combined with
    Gaussian spatial weighting.

    Two voxelization schemes are supported:

    - ``"gaus_full"``: each PoCA point contributes to all voxels according
      to a Gaussian spatial kernel.
    - ``"gaus3"``: each PoCA point contributes only to a local
      ``(2 * offset_radius + 1)^3`` voxel stencil.

    The aggregated 16-dimensional learned voxel representation is
    concatenated with a logarithmic evidence channel and passed to a
    3D U-Net. The U-Net produces a single voxel-wise prediction of the
    scattering density.

    Args:
        voxel_centers (Tensor):
            Tensor containing the 3D coordinates of the voxel centers,
            with shape ``(X * Y * Z, 3)``.

        voxel_shape (tuple):
            Number of voxels along the three spatial dimensions,
            specified as ``(X, Y, Z)``.

        radius (float, optional):
            Characteristic radius of the Gaussian spatial weighting.
            The Gaussian standard deviation is set to ``radius / 2``.
            Defaults to ``0.1``.

        feat_mean (Tensor, optional):
            Mean values used to normalize the eight input PoCA features.
            If ``None``, a zero vector is used.

        feat_std (Tensor, optional):
            Standard deviations used to normalize the eight input PoCA
            features. If ``None``, a unit vector is used.

        offset_radius (int, optional):
            Radius of the local voxel stencil used by the ``"gaus3"``
            voxelizer. For example, ``offset_radius=1`` corresponds to
            a ``3 x 3 x 3`` stencil. Defaults to ``1``.

        voxelizer_type (str, optional):
            Voxelization scheme to use. Must be either ``"gaus_full"``
            or ``"gaus3``. Defaults to ``"gaus_full"``.

    Input:
        poca_tensor (Tensor):
            PoCA point features with shape ``(B, N, F)``.

        point_mask (Tensor, optional):
            Boolean mask from the DataLoader collate function with
            shape ``(B, N)`` indicating valid PoCA points. If ``None``, 
            all points are treated as valid.

    Output:
        Tensor:
            Voxel-wise predicted scattering densities with shape
            ``(B, X, Y, Z)``.
    """

    def __init__(
        self,
        voxel_centers: Tensor,
        voxel_shape: tuple,
        radius: float = 0.1,
        feat_mean: Tensor = None,
        feat_std: Tensor = None,
        offset_radius: int = 1,
        voxelizer_type: str = 'gaus_full'
    ):
        super().__init__()

        X, Y, Z = voxel_shape
        assert voxel_centers.shape[0] == X * Y * Z, (
            f"voxel_centers has {voxel_centers.shape[0]} rows, "
            f"expected X*Y*Z = {X * Y * Z}"
        )

        self.register_buffer("voxel_centers", voxel_centers)
        self.voxel_shape = voxel_shape
        self.n_voxels = voxel_centers.shape[0]
        self.radius = radius
        self.stencil_r = offset_radius
        self.voxelizer_type = voxelizer_type

        if self.voxelizer_type == "gaus3":
            centers_grid = voxel_centers.view(
                X, Y, Z, 3
            )
            step_x = (
                centers_grid[1, 0, 0]
                - centers_grid[0, 0, 0]
            )
            step_y = (
                centers_grid[0, 1, 0]
                - centers_grid[0, 0, 0]
            )
            step_z = (
                centers_grid[0, 0, 1]
                - centers_grid[0, 0, 0]
            )
            pitch = torch.stack([
                step_x.norm(),
                step_y.norm(),
                step_z.norm(),
            ])
            self.register_buffer(
                "voxel_pitch",
                pitch
            )
            self.register_buffer(
                "origin",
                centers_grid[0, 0, 0].clone()
            )

            r = self.stencil_r

            offsets = torch.stack(
                torch.meshgrid(
                    torch.arange(-r, r + 1),
                    torch.arange(-r, r + 1),
                    torch.arange(-r, r + 1),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 3)

            self.register_buffer(
                "stencil_offsets",
                offsets
            )

        # Feature statistics

        if feat_mean is None:
            feat_mean = torch.zeros(8)
        if feat_std is None:
            feat_std = torch.ones(8)

        self.register_buffer("feat_mean", feat_mean)
        self.register_buffer("feat_std", feat_std)

        # Point encoder

        self.point_mlp = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        # Learned point scalar attention score

        self.attn_score = nn.Linear(16, 1)

      
        # UNet

        self.unet = UNet3D(in_channels=17, base=32)

    def pad_to_unet(self, x):
        B, C, X, Y, Z = x.shape
        pad_x = (4 - X % 4) % 4
        pad_y = (4 - Y % 4) % 4
        pad_z = (4 - Z % 4) % 4
        return F.pad(x, (0, pad_z, 0, pad_y, 0, pad_x))

    def forward(self, poca_tensor, point_mask=None):

        B, N, _ = poca_tensor.shape
        X, Y, Z = self.voxel_shape
        V = self.n_voxels
        device = poca_tensor.device

        if point_mask is None:
            point_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # Extracting the 8 PoCA features

        poca_tensor = poca_tensor[:, :, :8]
        K = poca_tensor.shape[1]

        
        # Input feature standardization
        poca_norm = (poca_tensor - self.feat_mean) / (self.feat_std + 1e-8)
        poca_flat = poca_norm.reshape(B * K, 8)

        #feature encoding
        muon_feats = self.point_mlp(poca_flat).view(B, K, 16)

        # Global per-point attention
        raw_scores = self.attn_score(muon_feats).squeeze(-1)  # (B, K)
        
        if self.voxelizer_type == "gaus_full":

            # each PoCA point contributes to all voxels according 
            # to a Gaussian spatial kernel.

            xyz_sel = poca_tensor[:, :, :3]  # spatial features

            diff = (
                self.voxel_centers[None, :, None, :]
                - xyz_sel[:, None, :, :]
            )

            dists = diff.norm(dim=-1)

            sigma = self.radius / 2

            weights = torch.exp(
                -dists**2 /
                (2 * sigma**2)
            )

            weights = (
                weights
                * point_mask[:, None, :].float()
            )

            # Global attention over points
            raw_scores_V = (
                raw_scores[:, None, :]
                .expand(B, V, K)
            )

            alpha = torch.softmax(
                raw_scores_V,
                dim=-1
            )

            alpha = alpha * weights

            alpha = alpha / (
                alpha.sum(
                    dim=-1,
                    keepdim=True
                ) + 1e-8
            )

            # Learned voxel features
            agg_feat = torch.einsum(
                "bvk,bkc->bvc",
                alpha,
                muon_feats
            )

            # Evidence
            evidence = weights.sum(
                dim=-1,
                keepdim=True
            )

        elif self.voxelizer_type == "gaus3":

            # 3x3x3 local stencil aggregation

            xyz_sel = poca_tensor[:, :, :3]

            # Find containing voxel
            rel = (
                xyz_sel - self.origin
            ) / self.voxel_pitch

            base_idx = rel.floor().long()

            # Candidate voxels
            S = self.stencil_offsets.shape[0]

            cand_idx = (
                base_idx[:, :, None, :]
                + self.stencil_offsets[None, None, :, :]
            )

            shape_t = torch.tensor(
                [X, Y, Z],
                device=device,
                dtype=torch.long
            )

            valid = (
                (cand_idx >= 0)
                & (cand_idx < shape_t)
            ).all(dim=-1)

            # Clamp only for safe indexing
            cand_idx_clamped = cand_idx.clone()

            cand_idx_clamped[..., 0] = cand_idx_clamped[..., 0].clamp(
                0, X - 1
            )

            cand_idx_clamped[..., 1] = cand_idx_clamped[..., 1].clamp(
                0, Y - 1
            )

            cand_idx_clamped[..., 2] = cand_idx_clamped[..., 2].clamp(
                0, Z - 1
            )

            flat_vidx = (
                cand_idx_clamped[..., 0] * (Y * Z)
                + cand_idx_clamped[..., 1] * Z
                + cand_idx_clamped[..., 2]
            )

            # Gaussian weights
            vcenters = self.voxel_centers[
                flat_vidx
            ]

            dists = (
                vcenters
                - xyz_sel[:, :, None, :]
            ).norm(dim=-1)

            sigma = self.radius / 2

            weights = torch.exp(
                -dists**2 /
                (2 * sigma**2)
            )

            weights = (
                weights
                * valid.float()
                * point_mask[:, :, None].float()
            )

            # Combine attention + spatial weight
            alpha = (
                torch.softmax(
                    raw_scores,
                    dim=1
                )
            )

            edge_alpha = (
                alpha[:, :, None]
                * weights
            )

            # Flatten edges
            b_idx = torch.arange(
                B,
                device=device
            )[:, None, None].expand(
                B, K, S
            )

            flat_edge_vidx = (
                b_idx * V + flat_vidx
            ).reshape(-1)

            edge_alpha_flat = (
                edge_alpha.reshape(-1)
            )

            # Normalize aggregation weights
            denom = torch.zeros(
                B * V,
                device=device,
                dtype=edge_alpha_flat.dtype
            )

            denom.index_add_(
                0,
                flat_edge_vidx,
                edge_alpha_flat
            )

            edge_alpha_norm = (
                edge_alpha_flat
                / (
                    denom[flat_edge_vidx]
                    + 1e-8
                )
            )

            # Aggregate learned features
            feat_edges = (
                muon_feats[:, :, None, :]
                * edge_alpha_norm.view(
                    B, K, S, 1
                )
            ).reshape(-1, 16)

            agg_feat = torch.zeros(
                B * V,
                16,
                device=device,
                dtype=feat_edges.dtype
            )

            agg_feat.index_add_(
                0,
                flat_edge_vidx,
                feat_edges
            )

            agg_feat = agg_feat.view(
                B, V, 16
            )

            # Evidence
            evidence = torch.zeros(
                B * V,
                device=device,
                dtype=weights.dtype
            )

            evidence.index_add_(
                0,
                flat_edge_vidx,
                weights.reshape(-1)
            )

            evidence = evidence.view(
                B, V, 1
            )

        else:
            raise RuntimeError(
                f"Unknown voxelizer_type: {self.voxelizer_type}"
            )
        
        evidence = torch.log1p(evidence).view(B, V, 1)

        # Learned voxel representation -> (B,C,X,Y,Z)
    
        voxel_input = torch.cat([agg_feat, evidence], dim=-1)   # (B, V, 17)
        voxel_input = voxel_input.view(B, X, Y, Z, 17)
        voxel_input = voxel_input.permute(0, 4, 1, 2, 3).contiguous()

        # Passing to UNet

        voxel_input = self.pad_to_unet(voxel_input)
        out = self.unet(voxel_input)
        out = out.squeeze(1)
        out = out[:, :X, :Y, :Z]

        return out