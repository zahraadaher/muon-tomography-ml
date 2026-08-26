import os
import argparse
import json
import torch
from functools import partial

from src.params import ExperimentParams

from src.loss.factory import get_loss
from src.train.trainer import Trainer
from src.data.collate import collate_poca_batch
from src.utils.build_utils import build_data_module
from src.utils.build_utils import set_data_splits
from src.utils.build_utils import build_model


def main():
    parser = argparse.ArgumentParser(description="Train a model for Muon Tomography")
    parser.add_argument("--config", required=True, type=str, help="Path to config JSON")
    parser.add_argument("--output", type=str, help="Output directory", default="outputs/")

    args = parser.parse_args()

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join(
        args.output,
        config_name
    )
    print(f"Outputs will be saved to: {out_dir}")

    # loading experiment parameters
    print(f"Loading parameters from {args.config}...")
    params = ExperimentParams.from_json(args.config)
    
    device = torch.device(params.model.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # saving splits for reproducibility
    split_path = "/home/ucl/cp3/zdaher/POCA_NET/muon-tomography-ml/datasets/splits/default_split.json"

    #----------------------------------------------------------------------------------------------
    #---------------------Dataset -----------------------------------------------------------------

    split_path = "/home/ucl/cp3/zdaher/POCA_NET/muon-tomography-ml/datasets/splits/default_split.json"

    # build data module
    data_module = build_data_module(config_json=args.config, n_points=20000)

    if os.path.exists(split_path):
        # extract dataset split indices, also handles feature stats extraction, 
        # computed using the train dataset
        print(f"Loading split from {split_path}")
        set_data_splits(data_module=data_module, split_json= split_path)

    else:
        # if no splits are found, uses the generated splits when the data module was created
        print("Creating new split")
        os.makedirs(
            os.path.dirname(split_path),
            exist_ok=True
        )
        with open(split_path, "w") as f:
            json.dump(
                {
                    "train": data_module.train_indices,
                    "val": data_module.val_indices,
                    "test": data_module.test_indices
                },
                f,
                indent=4
            )

    # creating train/val/test datasets
    train_dataset, val_dataset, test_dataset = data_module.create_datasets()

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))


    #----------------------------------------------------------------------------------------------
    #---------------------Model ------------------------------------------------------------------- 

    model = build_model(config_json=args.config, data_module=data_module)

    #----------------------------------------------------------------------------------------------
    #---------------------Loss ------------------------------------------------------------------- 

    loss = get_loss(
        params.train.loss
    )

    #----------------------------------------------------------------------------------------------
    #---------------------Trainig -----------------------------------------------------------------

    # custom collate function (handles voxel dimension ordering for tomopt data, 
    # as well as different event number in a batch, since we are dealing with sparse input)
    metadata = data_module.get_metadata()
    collate_fn = partial(
        collate_poca_batch,
        voxel_shape=metadata["voxel_shape"]
    )

    # training module
    trainer = Trainer(
        model=model,
        loss_fn=loss,
        train_config=params.train,
        output_dir=out_dir,
        device=device,
        collate_fn=collate_fn
    )

    train_losses, val_losses = trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=params.train.epochs,
        batch_size=params.train.batch_size
    )

    #----------------------------------------------------------------------------------------------
    #---------------------Saving -----------------------------------------------------------------
    
    model_path = os.path.join(
        out_dir,
        "best_model.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )


    logs_path = os.path.join(
        out_dir,
        "training_logs.json"
    )

    with open(logs_path, "w") as f:

        json.dump(
            {
                "train_losses": train_losses,
                "val_losses": val_losses
            },
            f,
            indent=4
        )


    print(
        f"Training finished. Results saved to {out_dir}"
    )


if __name__ == "__main__":
    main()