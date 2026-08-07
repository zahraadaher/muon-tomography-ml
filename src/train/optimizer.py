import torch


def get_optimizer(model, config):

    opt_type = config.type.lower()

    if opt_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

    elif opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

    else:
        raise ValueError(
            f"Unknown optimizer {opt_type}"
        )