from torch.nn.utils.rnn import pad_sequence
import torch


def collate_poca_batch(batch, voxel_shape=None):

    C, H, W = voxel_shape
     
    features = [
        item["features"]
        for item in batch
    ]

    targets = [ 
        item["target"].view(C, H, W).permute(1, 2, 0)
        for item in batch
    ]

    material_ids = [
        item["material_id"]
        for item in batch
    ]

    padded_features = pad_sequence(
        features,
        batch_first=True,
        padding_value=0
    )

    lengths = torch.tensor(
        [x.shape[0] for x in features]
    )

    max_len = padded_features.shape[1]

    mask = (
        torch.arange(max_len)[None,:]
        <
        lengths[:,None]
    )

    return {
        "features": padded_features,
        "target": torch.stack(targets),
        "mask": mask,
        "material_id": torch.tensor(material_ids)
    }