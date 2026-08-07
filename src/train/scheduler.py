import torch


def get_scheduler(optimizer, config):

    if config is None:
        return None


    sched_type = config.type.lower()


    if sched_type == "reduce_on_plateau":

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.factor,
            patience=config.patience
        )


    elif sched_type == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )


    else:
        raise ValueError(
            f"Unknown scheduler {sched_type}"
        )