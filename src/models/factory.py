from .poca_net import POCA_NET


def get_model(
    model_config,
    voxel_centers,
):

    model_type = model_config.type


    if model_type == "poca_net":

        return POCA_NET(
            voxel_centers=voxel_centers,
            voxel_shape=model_config.voxel_shape,
            radius=model_config.radius,
        )


    elif model_type == "unet":

        return UNetVoxelX0InferNet(
            voxel_centers=voxel_centers,
            voxel_shape=model_config.voxel_shape,
            radius=model_config.radius,
        )


    elif model_type == "mlp_unet":

        return MLPUNetVoxelX0InferNet(
            voxel_centers=voxel_centers,
            voxel_shape=model_config.voxel_shape,
            radius=model_config.radius,
        )


    elif model_type == "classifier":

        return MuonSetClassifier(
            n_classes=model_config.n_classes,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dropout=model_config.dropout,
        )


    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )