import os ,sys
sys.path.append( os.path.dirname(os.path.realpath(__file__+"/..")))

import torch

#from .probabilistic_unet import ProbUNetVoxelX0Infer
from .unet import UNetVoxelX0InferNet
from .poca_net import POCA_NET_UNET
from src.params import ModelParams


def get_model(
    model_config: ModelParams,
    voxel_centers: torch.Tensor,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor
):
    model_type = model_config.type

    if model_type == "poca_net_unet":
    
            return POCA_NET_UNET(
                voxel_centers=voxel_centers,
                voxel_shape=model_config.voxel_shape,
                radius=model_config.radius,
                feat_mean=feat_mean,
                feat_std=feat_std,
                offset_radius=model_config.offset_radius,
                voxelizer_type=model_config.voxelizer_type
            )
    
    elif model_type == "unet":

        return UNetVoxelX0InferNet(
            voxel_centers=voxel_centers,
            voxel_shape=model_config.voxel_shape,
            radius=model_config.radius,
            offset_radius=model_config.offset_radius,
            voxelizer_type=model_config.voxelizer_type
        )

    # elif model_config.type == "prob_unet":

    #    return ProbUNetVoxelX0Infer(
    #         in_channels=model_config.in_channels,
    #         out_channels=model_config.out_channels,
    #         base_features=model_config.base_features,
    #         depth=model_config.depth,
    #         use_resblock=model_config.use_resblock,
    #     )


    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )