from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset


__all__=["MuonTomographyDataset"]

class MuonTomographyDataset(Dataset):
    """
    Dataset class used by MuonDataManager.
    Can operate in entirely cached (in_memory=True) or lazy mode (in_memory=False).
    """
    def __init__(self, 
                 data_indices: List[str],
                 reader,
                 cached_data: Dict[str, np.ndarray],
                 in_memory: bool = True,
                 feat_transform=None,
                 target_transform=None
                 ):
        self.data_indices = data_indices
        self.cached_data = cached_data
        self.reader = reader
        self.in_memory = in_memory
        self.transform = feat_transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data_indices)
        
    def __getitem__(self, idx):
        scan_name = self.data_indices[idx]
        
        if self.in_memory:
            sample = self.cached_data[scan_name]
        else:
            sample = self.reader.get_sample(scan_name)

        def to_tensor(x):
            if x is None:
                return None

            return torch.as_tensor(
                x,
                dtype=torch.float32
            )
            
        features = to_tensor(sample["features"])

        target = to_tensor(sample["target"])
        
        if self.transform:
            features = self.transform(features)

        if self.target_transform:
            target = self.target_transform(target)

        return {
                "features": features,
                "target": target,
                "prediction": to_tensor(sample["prediction"]),
                "material_id": torch.tensor(
                    sample["material_id"],
                    dtype=torch.long
                )
            }
        