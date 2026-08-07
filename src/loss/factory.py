import torch


def get_loss(loss_config):

    loss_type = loss_config.type.lower()


    if loss_type == "huber":

        return torch.nn.HuberLoss(
            delta= loss_config.delta
            )
        

    elif loss_type == "cross_entropy":

         return torch.nn.CrossEntropyLoss()


    else:

        return ValueError(
            f"Unknown loss type: {loss_type}"
        )
