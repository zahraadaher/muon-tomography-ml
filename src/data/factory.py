from .tomopt import TomOptDataModule


def get_dataset(config):

    if config.type == "tomopt":
        return TomOptDataModule(
            hdf5_path=config.path,
            split_config=config.split
        )

    raise ValueError(
        f"Unknown dataset {config['type']}"
    )