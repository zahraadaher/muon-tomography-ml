import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

    
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
    
class UNet3D(nn.Module):

    r"""
    3D U-Net for voxel-wise feature processing.

    The network consists of two encoder stages, a bottleneck, and two
    decoder stages with skip connections. The final 1x1x1 convolution
    produces one output channel per voxel.

    Parameters
    ----------
    in_channels : int, optional
        Number of input voxel features. Default is 4.
    base : int, optional
        Number of feature channels in the first encoder block.
        The subsequent blocks use 2*base and 4*base channels.
        Default is 32.

    Input
    -----
    x : Tensor
        Voxelized input with shape (B, in_channels, X, Y, Z).

    Returns
    -------
    Tensor
        Single-channel voxel-wise prediction with shape
        (B, 1, X, Y, Z).
    """

    def __init__(self, in_channels=4, base=32):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(base*2, base*4)

        self.up2 = nn.ConvTranspose3d(base*4, base*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)

        self.up1 = nn.ConvTranspose3d(base*2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        self.out = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)
    


class UNetVoxelX0InferNet(nn.Module):
    r"""
    Voxel-based neural network for reconstructing voxel-wise X0.

    PoCA points are first converted into a voxel representation using
    either hard voxel assignment or Gaussian spatial weighting. The
    resulting four-channel voxel representation is transformed with
    log1p and processed by a 3D U-Net. The network predicts log(1/X0)
    for each voxel.

    Parameters
    ----------
    voxel_centers : Tensor
        Coordinates of the voxel centers with shape (V, 3).
    voxel_shape : tuple
        Spatial voxel dimensions given as (X, Y, Z).
    voxelizer_type : str
        Voxelization method. ``"hard"`` uses hard voxel assignment,
        while ``"gaus3"`` uses Gaussian-weighted aggregation over
        a local stencil.
    radius : float, optional
        Spatial scale used by the Gaussian voxelizer. Default is 0.1.
    offset_radius : int, optional
        Radius of the voxel stencil used by the Gaussian voxelizer.
        A value of 1 corresponds to a 3x3x3 stencil. Default is 1.

    Input
    -----
    poca_tensor : Tensor
        PoCA features with shape (B, N, F).
    point_mask : Tensor, optional
        Boolean mask with shape (B, N) indicating valid PoCA points.
        If omitted, all points are treated as valid.

    Returns
    -------
    Tensor
        Predicted log(1/X0) volume with shape (B, X, Y, Z).
    """

    def __init__(self, 
                voxel_centers:Tensor,
                voxel_shape:tuple,
                voxelizer_type:str,
                radius: float = 0.1,
                offset_radius = 1):
        super().__init__()

        self.voxel_shape = voxel_shape
        self.voxelizer_type = voxelizer_type
        self.radius = radius
        self.offset_radius = offset_radius

        if self.voxelizer_type == 'hard':
            self.voxelizer = OnlineVoxelizer(
            voxel_centers,
            voxel_shape
        )
        elif self.voxelizer_type == 'gaus3':
            self.voxelizer = OnlineVoxelizerGaussian(
                voxel_centers,
                voxel_shape,
                radius=self.radius,
                stencil_radius=self.offset_radius 
                )

        self.unet = UNet3D(
            in_channels=4,
            base=32
    )
    
    def pad_to_unet(self, x):
        B, C, X, Y, Z = x.shape

        pad_x = (4 - X % 4) % 4
        pad_y = (4 - Y % 4) % 4
        pad_z = (4 - Z % 4) % 4

        return F.pad(x, (0, pad_z, 0, pad_y, 0, pad_x))
        
    def forward(self, poca_tensor, point_mask=None):
        B, N, _ = poca_tensor.shape
        X, Y, Z = self.voxel_shape
        device = poca_tensor.device

        if point_mask is None:
            point_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        voxel_grid = self.voxelizer(poca_tensor, point_mask)
        voxel_grid = torch.log1p(voxel_grid)
        voxel_grid = self.pad_to_unet(voxel_grid)
        out = self.unet(voxel_grid)
        out = out.squeeze(1)
        out = out[:, :X, :Y, :Z]
        return out


class OnlineVoxelizer(nn.Module):

    def __init__(self, voxel_centers: Tensor, voxel_shape: tuple):
        super().__init__()

        self.register_buffer("voxel_centers", voxel_centers)

        self.voxel_shape = voxel_shape
        X,Y,Z = voxel_shape

        xs = voxel_centers[:,0].view(X,Y,Z)
        ys = voxel_centers[:,1].view(X,Y,Z)
        zs = voxel_centers[:,2].view(X,Y,Z)

        dx = xs[1,0,0] - xs[0,0,0]
        dy = ys[0,1,0] - ys[0,0,0]
        dz = zs[0,0,1] - zs[0,0,0]

        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.xmin = xs.min() - dx/2
        self.ymin = ys.min() - dy/2
        self.zmin = zs.min() - dz/2

    def forward(self, poca_tensor, point_mask):

        B,N,_ = poca_tensor.shape
        X,Y,Z = self.voxel_shape
        device = poca_tensor.device

        x = poca_tensor[:,:,0]
        y = poca_tensor[:,:,1]
        z = poca_tensor[:,:,2]

        theta = poca_tensor[:,:,6]
        theta_unc = poca_tensor[:,:,7]

        # voxel indices
        ix = ((x - self.xmin)/self.dx).long()
        iy = ((y - self.ymin)/self.dy).long()
        iz = ((z - self.zmin)/self.dz).long()

        valid = (
            point_mask &
            (ix >= 0) & (ix < X) &
            (iy >= 0) & (iy < Y) &
            (iz >= 0) & (iz < Z)
        )

        batch_ids = (
            torch.arange(B, device=device)
            .unsqueeze(1)
            .expand(B,N)
        )

        ix = ix[valid]
        iy = iy[valid]
        iz = iz[valid]

        theta = theta[valid]
        theta_unc = theta_unc[valid]

        batch_ids = batch_ids[valid]

        # channels:
        # 0 = count
        # 1 = mean theta
        # 2 = rms theta
        # 3 = mean theta_unc

        C = 4

        grid = torch.zeros(
            B, C, X, Y, Z,
            device=device
        )

        # counts
        grid[:,0].index_put_(
            (batch_ids, ix, iy, iz),
            torch.ones_like(theta),
            accumulate=True
        )

        # theta sum
        grid[:,1].index_put_(
            (batch_ids, ix, iy, iz),
            theta,
            accumulate=True
        )

        # theta^2
        grid[:,2].index_put_(
            (batch_ids, ix, iy, iz),
            theta**2,
            accumulate=True
        )

        # theta uncertainty
        grid[:,3].index_put_(
            (batch_ids, ix, iy, iz),
            theta_unc,
            accumulate=True
        )

        count = grid[:,0].clamp(min=1)

        grid[:,1] /= count
        grid[:,2] = torch.sqrt(grid[:,2] / count)
        grid[:,3] /= count

        return grid
    

class OnlineVoxelizerGaussian(nn.Module):

    def __init__(
        self,
        voxel_centers: Tensor,
        voxel_shape: tuple,
        radius:float=0.1,
        stencil_radius:int=1,
    ):
        super().__init__()

        self.register_buffer("voxel_centers", voxel_centers)

        self.voxel_shape = voxel_shape
        X, Y, Z = voxel_shape

        self.radius = radius
        self.stencil_r = stencil_radius

        centers_grid = voxel_centers.view(X, Y, Z, 3)

        step_x = centers_grid[1, 0, 0] - centers_grid[0, 0, 0]
        step_y = centers_grid[0, 1, 0] - centers_grid[0, 0, 0]
        step_z = centers_grid[0, 0, 1] - centers_grid[0, 0, 0]

        self.register_buffer(
            "voxel_pitch",
            torch.stack([
                step_x.norm(),
                step_y.norm(),
                step_z.norm()
            ])
        )

        self.register_buffer(
            "origin",
            centers_grid[0, 0, 0].clone()
        )

        # Fixed 3x3x3 stencil

        r = stencil_radius

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

    def forward(self, poca_tensor, point_mask):

        B, N, _ = poca_tensor.shape
        X, Y, Z = self.voxel_shape
        V = X * Y * Z
        device = poca_tensor.device

        # PoCA features

        xyz = poca_tensor[:, :, :3]
        theta = poca_tensor[:, :, 6]
        theta_unc = poca_tensor[:, :, 7]

        # Find containing voxel

        rel = (
            xyz - self.origin
        ) / self.voxel_pitch

        base_idx = rel.floor().long()

        valid_base = (
            (base_idx >= 0) &
            (base_idx < torch.tensor(
                [X, Y, Z],
                device=device
            ))
        ).all(dim=-1)

        valid = point_mask & valid_base

        # Construct 3x3x3 candidate voxels

        S = self.stencil_offsets.shape[0]

        cand_idx = (
            base_idx[:, :, None, :]
            + self.stencil_offsets[None, None, :, :]
        )
        # (B, N, 27, 3)

        shape_t = torch.tensor(
            [X, Y, Z],
            device=device,
            dtype=torch.long
        )

        valid_voxels = (
            (cand_idx >= 0) &
            (cand_idx < shape_t)
        ).all(dim=-1)

        cand_idx_clamped = torch.maximum(
            cand_idx,
            torch.zeros(3, device=device, dtype=torch.long)
        )

        cand_idx_clamped = torch.minimum(
            cand_idx_clamped,
            shape_t - 1
        )

        flat_vidx = (
            cand_idx_clamped[..., 0] * (Y * Z)
            + cand_idx_clamped[..., 1] * Z
            + cand_idx_clamped[..., 2]
        )

        # Gaussian spatial weights

        vcenters = self.voxel_centers[flat_vidx]

        dists = (
            vcenters
            - xyz[:, :, None, :]
        ).norm(dim=-1)

        sigma = self.radius / 2

        weights = torch.exp(
            -dists**2 / (2 * sigma**2)
        )

        weights = (
            weights
            * valid_voxels.float()
            * valid[:, :, None].float()
        )

        # Batch indices

        batch_idx = torch.arange(
            B,
            device=device
        )[:, None, None].expand(B, N, S)

        flat_edge_vidx = (
            batch_idx * V + flat_vidx
        ).reshape(-1)

        weights_flat = weights.reshape(-1)

        # Create output

        grid = torch.zeros(
            B * V,
            4,
            device=device,
            dtype=poca_tensor.dtype
        )

        # Gaussian-weighted accumulation

        # count
        grid[:, 0].index_add_(
            0,
            flat_edge_vidx,
            weights_flat
        )

        # theta
        grid[:, 1].index_add_(
            0,
            flat_edge_vidx,
            (
                weights * theta[:, :, None]
            ).reshape(-1)
        )

        # theta^2
        grid[:, 2].index_add_(
            0,
            flat_edge_vidx,
            (
                weights * theta[:, :, None]**2
            ).reshape(-1)
        )

        # theta uncertainty
        grid[:, 3].index_add_(
            0,
            flat_edge_vidx,
            (
                weights * theta_unc[:, :, None]
            ).reshape(-1)
        )

        grid = grid.view(
            B, X, Y, Z, 4
        ).permute(
            0, 4, 1, 2, 3
        ).contiguous()

        # Weighted statistics

        count = grid[:, 0].clamp(min=1e-8)

        grid[:, 1] /= count

        grid[:, 2] = torch.sqrt(
            grid[:, 2] / count
        )

        grid[:, 3] /= count

        return grid
