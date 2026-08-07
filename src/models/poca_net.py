import torch
import torch.nn as nn
from torch import Tensor


class POCA_NET(nn.Module):
    """
    Neural network for local feature aggregation of 3D muon PoCA points
    into voxel-wise predictions, using attention-based pooling.

    Each voxel aggregates features from nearby PoCA points within a radius,
    using a learned attention score. The final output is a scalar prediction
    per voxel, reshaped into a 3D grid. Note that the output is not constrained to be positive, allowing the model to learn log(X0) 
    since X0 can span orders of magnitude across materials.

    Args:
        voxel_centers (Tensor): (V, 3) coordinates of voxel centers.
        voxel_shape (tuple): (X Y Z) shape of the output 3D grid.
        radius (float): Radius (in meters) around each voxel cemter to aggregate points from.
    """
    
    def __init__(
        self,
        voxel_centers: Tensor,
        voxel_shape: tuple,
        radius: float = 0.2
    ):
        super().__init__()
        self.register_buffer('voxel_centers', voxel_centers)
        self.voxel_shape = voxel_shape
        self.n_voxels = voxel_centers.shape[0]
        self.radius = radius
    
 
        # normalizing voxel centers when concatenating
        vc = voxel_centers
        self.register_buffer('voxel_centers_norm', 
            (vc - vc.mean(dim=0)) / (vc.std(dim=0) + 1e-8))

        # Processes each POCA point's features
        self.point_mlp = nn.Sequential(
            nn.Linear(8, 16),
            #nn.GroupNorm(4, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(p=0.05),  # Reduced dropout
            nn.Linear(16, 16),   # Smaller hidden size
            # nn.GroupNorm(4, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )

        # Processes aggregated voxel features 
        self.voxel_mlp = nn.Sequential(
            nn.Linear(16 + 3, 256),
            #nn.GroupNorm(4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            #nn.GroupNorm(4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(128, 1),
            #nn.Softplus()  # predict log(X0)
        )

        # Attention mechanism to score each point
        self.attn_score = nn.Linear(16, 1)
        
        self._initialize_weights() 
        
    def _initialize_weights(self) -> None: 
        """
        Xavier initialization for linear layers.
        """
        for m in self.modules(): 
            if isinstance(m, nn.Linear): 
                torch.nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None: torch.nn.init.constant_(m.bias, 0) 
        

    def forward(self, poca_tensor, point_mask=None):
        B, N, _ = poca_tensor.shape
        V = self.n_voxels
        C = 16  # point_mlp output dim
        K = 10000
        device = poca_tensor.device

        if point_mask is None:
            point_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        poca_tensor = poca_tensor[:, :, :8]

        # --- Spatial filter: keep only muons inside voxelized volume ---
        xmin = self.voxel_centers[:, 0].min() - self.radius
        xmax = self.voxel_centers[:, 0].max() + self.radius
        ymin = self.voxel_centers[:, 1].min() - self.radius
        ymax = self.voxel_centers[:, 1].max() + self.radius
        zmin = self.voxel_centers[:, 2].min() - self.radius
        zmax = self.voxel_centers[:, 2].max() + self.radius

        xyz = poca_tensor[:, :, :3]  # (B, N, 3)
        inbox = (
            (xyz[:, :, 0] >= xmin) & (xyz[:, :, 0] <= xmax) &
            (xyz[:, :, 1] >= ymin) & (xyz[:, :, 1] <= ymax) &
            (xyz[:, :, 2] >= zmin) & (xyz[:, :, 2] <= zmax) &
            point_mask
        )  # (B, N)

        # --- Subsample to top-K by |theta| per sample ---
        theta = poca_tensor[:, :, 6]  # (B, N)
        theta_masked = theta.abs() * inbox.float()  # zero out out-of-volume points

        K_actual = min(K, inbox.sum(dim=1).min().item())  # guard if fewer than K in-volume
        topk_idx = torch.topk(theta_masked, K_actual, dim=1).indices  # (B, K)

        # Gather selected points
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(B, K_actual, 8)
        poca_sel = torch.gather(poca_tensor, 1, topk_idx_exp)  # (B, K, 8)

        # --- Point MLP ---
        poca_flat = poca_sel.reshape(B * K_actual, 8)
        muon_feats = self.point_mlp(poca_flat)                          # (B*K, C)
        raw_scores = self.attn_score(muon_feats).squeeze(-1)            # (B*K,)

        muon_feats = muon_feats.view(B, K_actual, C)                    # (B, K, C)
        raw_scores = raw_scores.view(B, K_actual)                       # (B, K)

        # --- Batched attention via broadcasting (no loop) ---
        xyz_sel = poca_sel[:, :, :3]                                    # (B, K, 3)

        diff = self.voxel_centers[None, :, None, :] - xyz_sel[:, None, :, :]  # (B, V, K, 3)
        dists = diff.norm(dim=-1)                                              # (B, V, K)

        sigma = self.radius / 2
        weights = torch.exp(-dists**2 / (2 * sigma**2))                       # (B, V, K)
        raw_scores_V = raw_scores[:, None, :].expand(B, V, K_actual)          # (B, V, K)
        raw_scores_V = raw_scores_V + torch.log(weights + 1e-8)
        alpha = torch.softmax(raw_scores_V, dim=-1)                           # (B, V, K)

        # alpha: (B, V, K),  muon_feats: (B, K, C)
        agg_feat = torch.einsum('bvk,bkc->bvc', alpha, muon_feats)     # (B, V, C)

        vc_norm = self.voxel_centers_norm[None].expand(B, -1, -1)      # (B, V, 3)
        voxel_input = torch.cat([agg_feat, vc_norm], dim=-1)           # (B, V, C+3)
        voxel_out = self.voxel_mlp(voxel_input)                        # (B, V, 1)

        return voxel_out.squeeze(-1).view(B, *self.voxel_shape)        # (B, X, Y, Z)
            
