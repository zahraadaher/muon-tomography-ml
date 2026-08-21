from typing import Any, List, Dict, Union, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from .readers.hdf5_reader import HDF5Reader 
from .transforms import NormalizeFeatures, LogInverseTarget


__all__=["MuonTomographyDataset"]

class MuonTomographyDataset(Dataset):
    """
    Main dataset class for Muon Tomography.

    Args:
    -----
    - data_indices: list of indices for a dataset
    - reader: optional dataset reader (e.g. hdf5 reader, npz reader). It is required if there is no cached_data is None.
    - cached_data: optionally available cached data. If None, reader is required.
    - feat_transform: optional transform for standardizing input features.
    - target_transform: optional tranform for target

    """
    def __init__(self, 
                 data_indices: List[str],
                 reader: Optional[Union[HDF5Reader, Any]] = None,
                 cached_data: Optional[Dict[str, np.ndarray]] = None,
                 feat_transform: Optional[NormalizeFeatures] = None,
                 target_transform: Optional[LogInverseTarget] = None
                 ):
        self.data_indices = data_indices
        self.cached_data = cached_data
        self.reader = reader
        self.transform = feat_transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data_indices)
        
    def __getitem__(self, idx):
        scan_name = self.data_indices[idx]
        
        if self.cached_data is not None:
            sample = self.cached_data[scan_name]
        else:
            if self.reader is None: raise Exception("Provide reader for MuonTomographyDataset, there are no cached data to parse.")
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

        if (
            sample["prediction"] is not None
            and self.target_transform 
            and sample["prediction"].ndim > 0   # otherwase if prediction is 0, it does not exist   
            ):
                prediction = to_tensor(sample["prediction"])
                prediction = self.target_transform(prediction)

        return {
                "features": features,
                "target": target,
                "prediction": prediction,
                "material_id": sample["material_id"],
                "mask": to_tensor(sample["mask"])
            }
        