from typing import Optional

from .tomopt import TomOptDataModule
from ..params import DataParams


def get_dataset(config: DataParams, n_points: Optional[int], normalize: Optional[bool], in_memory: Optional[bool]):

    if config.type == "tomopt":
        return TomOptDataModule(
            hdf5_path=config.path,
            split_config=config.split,
            n_points=n_points,
            normalize=normalize,
            in_memory=in_memory
        )

    raise ValueError(
        f"Unknown dataset {config['type']}"
    )