import os
import torch

from torch.utils.data import DataLoader

from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .early_stopping import EarlyStopping


class Trainer:

    def __init__(
        self,
        model,
        loss_fn,
        train_config,
        output_dir,
        device,
        collate_fn=None,
    ):

        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = get_optimizer(
            model,
            train_config.optimizer
        )

        scheduler_config = getattr(
            train_config,
            "scheduler",
            None
        )

        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_config
        )

        cfg = train_config.early_stopping
        if cfg.enabled:
            self.early_stopping = EarlyStopping(
                patience=cfg.patience,
                min_delta=cfg.min_delta
            )
        else:
            self.early_stopping = None

        self.save_best_only = (
            train_config.checkpoint.save_best_only
        )  

        self.output_dir = output_dir
        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        self.collate_fn = collate_fn
         
        self.device = device


    def train_epoch(self, dataloader):

        self.model.train()

        total_loss = 0

        for batch in dataloader:

            features = batch["features"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            if "mask" in batch:
                # handles mask from collated point cloud data
                prediction = self.model(
                    features,
                    batch["mask"].to(self.device)
                )
            else:
                prediction = self.model(features)

            loss = self.loss_fn(
                prediction,
                target
            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()


        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader):

        self.model.eval()

        total_loss = 0

        for batch in dataloader:

            features = batch["features"].to(self.device)
            target = batch["target"].to(self.device)

            if "mask" in batch:
                # handles mask from collated point cloud data
                prediction = self.model(
                    features,
                    batch["mask"].to(self.device)
                )
            else:
                prediction = self.model(features)

            loss = self.loss_fn(
                prediction,
                target
            )

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def save_checkpoint(self, filename="best_model.pth", epoch=None, val_loss=None):

        path = os.path.join(
            self.output_dir,
            filename
        )

        torch.save(
            {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            },
            path
        )

    def fit(
        self,
        train_dataset,
        val_dataset,
        epochs,
        batch_size
    ):
        train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=self.collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        train_history = []
        val_history = []

        best_val = float("inf")

        for epoch in range(epochs):

            train_loss = self.train_epoch(
                train_loader
            )

            val_loss = self.validate(
                val_loader
            )

            train_history.append(
                train_loss
            )

            val_history.append(
                val_loss
            )

            if self.scheduler is not None:
        
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)

                else:
                    self.scheduler.step()

            if self.save_best_only:

                if val_loss < best_val:
                    best_val = val_loss
                    self.save_checkpoint(epoch=epoch, val_loss=val_loss)

            else:

                self.save_checkpoint(
                    filename=f"model_epoch_{epoch}.pth",
                    epoch=epoch,
                    val_loss=val_loss
                )


            print(
                f"Epoch {epoch}: "
                f"train={train_loss:.4f}, "
                f"val={val_loss:.4f}"
            )

            if self.early_stopping is not None:
                stop = self.early_stopping.step(val_loss)
                if stop:
                    print(
                        f"Early stopping at epoch {epoch+1}"
                    )
                    break

        return train_history, val_history