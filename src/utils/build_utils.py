import os ,sys
sys.path.append( os.path.dirname(os.path.realpath(__file__+"/..")))

from typing import Dict, List, Union
import json
import h5py
import torch

from src.params import ExperimentParams
from src.models.poca_net_unet import POCA_NET_UNET
from src.models.unet import UNetVoxelX0InferNet
from src.data.base import BaseDataModule
from src.data.tomopt import TomOptDataModule
from src.data.factory import get_dataset
from src.models.factory import get_model
from src.utils.feat_statistics import get_or_compute_feature_stats

__all__ = ['build_data_module', 'build_model', 'set_data_splits', 'get_scans_by_material', 'get_voxel_centers']

def build_data_module(
    config_json: str,
    n_points: int = 20000
    )-> BaseDataModule:
    r"""
    Needs to be called at start of training/evaluation to load dataset.

    Args:
    -----
    config_json: path to json config file, storing data and model configs.
    n_points: number of muon events to be sampled.

    Return:
    ------
    Instance of data module, loaded with full dataset.
    """
    
    params = ExperimentParams.from_json(config_json)
    normalize = params.data.normalize_features
    data_module = get_dataset(config=params.data, n_points=n_points, normalize = normalize)
    data_module.prepare()

    return data_module

def build_model(
    config_json: str,
    data_module: BaseDataModule
            
)-> Union[POCA_NET_UNET, UNetVoxelX0InferNet]:
    r"""
    Model type (POCA_NET, UNET etc) is extrcted from the json config.
    The data_module, prebuilt with `build_data_module()` is needed for the metadata
    storing voxelization info to be used by thr model, as well as the feature standardization stats
    on the trainig dataset.

    Args:
    ----- 
    - config_json : path to json config file
    - data_module : data_module pre-built instance

    Return:
    -------
    configured model instance
    """
    
    params = ExperimentParams.from_json(config_json)
    metadata = data_module.get_metadata()

    voxel_centers = get_voxel_centers(
    voxel_shape=metadata["voxel_shape"],
    voxel_size=metadata["voxel_size"],
    origin=metadata["voxel_origin"]
    )

    model = get_model(
        params.model,
        voxel_centers,
        feat_mean=data_module.feature_mean[:8],
        feat_std=data_module.feature_std[:8]
        )

    return model

def set_data_splits(
    data_module: BaseDataModule,
    split_json : str,
    )->None:
    r"""
    Sets the splitting indices of the train/val/test datasets extracted from a json file.
    Splitting indices are assigned randomly when the data module is built, but they can be set for a specific
    splitted dataset, to ensure reproducinility at evaluation.
    Then feature standardization stats are extracted from the same directory.
    If the json containing the stats does not exist at the path, it is created. 

    Args:
    -----
    data_module: instance of a pre-built data module. 
    split_json: path to json file stroign split indices
    """
    if os.path.exists(split_json):

        print(f"Loading split from {split_json}")

        with open(split_json, "r") as f:
            split = json.load(f)

        data_module.train_indices = split["train"]
        data_module.val_indices = split["val"]
        data_module.test_indices = split["test"]

    # feature standardization stats to be passed to model, extracted from training dataset
    # if not existing at split path, will be calculated
    data_module.feature_mean, data_module.feature_std = get_or_compute_feature_stats(
                        reader=data_module.reader,
                        indices=data_module.train_indices,
                        split_path=split_json,
                        metadata=data_module.get_metadata()
                        )

def get_scans_by_material(
    data_path: str,
    data_module: TomOptDataModule,
    dataset_type: str = 'test'
)-> Dict[str, List[str]]:
    r"""
    Groups samples by material name in a given dataset. To be used exclusevly for `TomOptDataModule` class due to dataset structure.
    
    Args:
    -----
    - data_path: path to the h5py dataset.
    - data_module: instance of the pre-built TomOptDataModule class
    - dataset_type: either of 'test', 'val'. 'train'

    Return:
    -------
    dict: {material_name: [scan_name_1, ...]}
    """
    # Group scans by material name
    scans_by_material = {}   
    if dataset_type == 'test':
        indices = data_module.test_indices
    elif dataset_type == 'val':
        indices  = data_module.val_indices 
    elif dataset_type == 'train':
        indices = data_module.train_indices 
    else: raise Exception("specify dataset type: test, val or train.")
    
    with h5py.File(data_path, 'r') as h5f:
        dg = h5f['data']
        for scan_name in indices:
            mat = dg[scan_name].attrs.get('material_name', b'')
            if isinstance(mat, bytes):
                mat = mat.decode('utf-8')
            mat = mat.lower()
            scans_by_material.setdefault(mat, []).append(scan_name)

    print("\nScans per material:")
    for mat, scans in sorted(scans_by_material.items()):
        print(f"  {mat:<15}: {len(scans)} scans")

    return scans_by_material

def get_voxel_centers(
    voxel_shape: List[int],
    voxel_size: float,
    origin: List[float]
    )-> torch.Tensor:
    """
    Defines the voxel center positions in spatial coodinates, given the grid dimensions, 
    voxel side length, and the spatial position of the corner of the grid corresponding to 
    the voxel with index (0, 0, 0).

    Args:
    ----
    - voxel_shape: number of voxels in XYZ of the voxelized VOI
    - voxel_size: side length [meters] of a voxel
    - origin: coordinates [meters] of the origin of the grid

    Return:
    -------
    Tensor of the xyz coordinates of the voxelized grid [meters]
    """
    nx, ny, nz = voxel_shape
    ox, oy, oz = origin

    x = torch.arange(nx)
    y = torch.arange(ny)
    z = torch.arange(nz)

    xv, yv, zv = torch.meshgrid(
        x, y, z,
        indexing="ij"
    )

    centers = torch.stack(
        [xv, yv, zv],
        dim=-1
    ).float()

    centers = centers * voxel_size + voxel_size/2

    centers[..., 0] += ox
    centers[..., 1] += oy
    centers[..., 2] += oz

    return centers.reshape(-1, 3)

        