import os
import argparse
import json
import torch
import partial

from src.params import ExperimentParams

from src.data.factory import get_dataset
from src.utils.geometry_tomopt import get_voxel_centers
from src.models.factory import get_model
from src.loss.factory import get_loss
from src.train.trainer import Trainer
from src.data.collate import collate_poca_batch


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

    # 1. Load Parameters
    print(f"Loading parameters from {args.config}...")
    params = ExperimentParams.from_json(args.config)
    
    device = torch.device(params.model.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # setting up data module
    normalize = params.data.normalize_features
    data_module = get_dataset(config=params.data)
    data_module.prepare(normalize=normalize)

    train_dataset, val_dataset, test_dataset = (
        data_module.create_datasets(
            normalize=normalize
        )
    )

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    ### model ### 

    metadata = data_module.get_metadata()

    voxel_centers = get_voxel_centers(
        voxel_shape=metadata["voxel_shape"],
        voxel_size=metadata["voxel_size"],
        origin=metadata["voxel_origin"]
    )

    model = get_model(
        params.model,
        voxel_centers
    )

    ### loss ###

    loss = get_loss(
        params.train.loss
    )

    ### colate function ###
    collate_fn = partial(
        collate_poca_batch,
        voxel_shape=metadata["voxel_shape"]
    )

    ### training ###

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

    ### save results ###

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