import json
from dataclasses import dataclass, field, asdict
from typing import Dict


@dataclass
class ModelParams:
    type: str = "poca_net"
    voxel_shape: tuple = (10, 10, 4)
    radius: float = 0.1
    offset_radius: float = 1
    voxelizer_type: str = 'hard'
    device: str = "cuda"


@dataclass
class SplitParams:
    type: str = "balanced_material_position"
    random_seed: int = 32
    n_materials: int = 5
    n_positions: int = 49
    n_val_pos: int = 4
    n_test_pos: int = 4


@dataclass
class DataParams:
    type: str = "tomopt"
    path: str = "datasets/dataset.h5"

    normalize_features: bool = False
    in_memory: bool = True

    batch_size: int = 32
    num_workers: int = 4

    split: SplitParams = field(default_factory=SplitParams)


@dataclass
class OptimizerParams:
    type: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-3


@dataclass
class SchedulerParams:
    type: str = "reduce_on_plateau"
    factor: float = 0.3
    patience: int = 3


@dataclass
class LossParams:
    type: str = "huber"
    delta: float = 1.0


@dataclass
class EarlyStoppingParams:
    enabled: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    monitor: str = "val_loss"


@dataclass
class CheckpointParams:
    save_best_only: bool = True
    monitor: str = "val_loss"


@dataclass
class TrainParams:
    epochs: int = 100

    batch_size: int = 32

    optimizer: OptimizerParams = field(
        default_factory=OptimizerParams
    )

    scheduler: SchedulerParams = field(
        default_factory=SchedulerParams
    )

    loss: LossParams = field(
        default_factory=LossParams
    )

    early_stopping: EarlyStoppingParams = field(
        default_factory=EarlyStoppingParams
    )

    checkpoint: CheckpointParams = field(
        default_factory=CheckpointParams
    )


@dataclass
class ExperimentParams:

    experiment: Dict = field(default_factory=dict)

    data: DataParams = field(
        default_factory=DataParams
    )

    model: ModelParams = field(
        default_factory=ModelParams
    )

    train: TrainParams = field(
        default_factory=TrainParams
    )


    @classmethod
    def from_json(cls, path):

        with open(path, "r") as f:
            cfg = json.load(f)


        return cls(

            experiment=cfg.get(
                "experiment",
                {}
            ),

            data=DataParams(
                **{
                    **cfg["data"],
                    "split": SplitParams(
                        **cfg["data"]["split"]
                    )
                }
            ),

            model=ModelParams(
                **cfg["model"]
            ),

            train=TrainParams(

                optimizer=OptimizerParams(
                    **cfg["train"]["optimizer"]
                ),

                scheduler=SchedulerParams(
                    **cfg["train"]["scheduler"]
                ),

                loss=LossParams(
                    **cfg["train"]["loss"]
                ),

                early_stopping=EarlyStoppingParams(
                    **cfg["train"]["early_stopping"]
                ),

                checkpoint=CheckpointParams(
                    **cfg["train"]["checkpoint"]
                ),

                epochs=cfg["train"]["epochs"]
            )
        )


    def to_json(self, path):

        with open(path, "w") as f:
            json.dump(
                asdict(self),
                f,
                indent=4
            )