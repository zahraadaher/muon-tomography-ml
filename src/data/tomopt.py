from locale import normalize
import logging

from .base import BaseDataModule
from .readers.hdf5_reader import HDF5Reader
from .dataset import MuonTomographyDataset
from .splits.balanced_material_position import BalancedMaterialPositionSplit
from .transforms import NormalizeFeatures, LogTarget
from src.utils.feat_statistics import compute_feature_stats


class TomOptDataModule(BaseDataModule):

    def __init__(
        self,
        hdf5_path,
        split_config,
        in_memory=True
    ):

        self.reader = HDF5Reader(hdf5_path)
        self.split_config = split_config

        self.in_memory = in_memory
        self.cached_data = {}

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

    def prepare(self, normalize=False):

        # Load data cache
        logging.info("Preparing dataset")
        for name in self.reader.list_samples():

            self.cached_data[name] = (
                self.reader.get_sample(name)
            )
        logging.info(
            f"Loaded {len(self.cached_data)} samples"
        )

        # Create train/val/test split
        splitter = BalancedMaterialPositionSplit(
            n_positions=self.split_config.n_positions,
            n_materials=self.split_config.n_materials,
            n_val_pos=self.split_config.n_val_pos,
            n_test_pos=self.split_config.n_test_pos,
            random_seed=self.split_config.random_seed
        )

        (
            self.train_indices,
            self.val_indices,
            self.test_indices
        ) = splitter.split(self.reader)

        # Compute feature statistics only if normalization is requested
        if normalize:
            self.feature_mean, self.feature_std = compute_feature_stats(
                self.reader,
                self.train_indices
            )
        else:
            self.feature_mean = None
            self.feature_std = None

    def create_datasets(self, normalize=False):

        train_transform = None
        val_transform = None
        test_transform = None
        x0_transform = LogTarget()

        if normalize:
            if self.feature_mean is None or self.feature_std is None:
                raise RuntimeError(
                    "Normalization requested but feature statistics were not computed. "
                    "Call prepare(normalize=True) first."
                )
            
            feat_transform = NormalizeFeatures(
                self.feature_mean,
                self.feature_std
            )
            train_transform = feat_transform
            val_transform = feat_transform
            test_transform = feat_transform

        train_dataset = MuonTomographyDataset(
            data_indices=self.train_indices,
            reader=self.reader,
            feat_transform=train_transform,
            target_transform=x0_transform,
            cached_data=self.cached_data,
            in_memory=self.in_memory
        )

        val_dataset = MuonTomographyDataset(
            data_indices=self.val_indices,
            reader=self.reader,
            feat_transform=val_transform,
            target_transform=x0_transform,
            cached_data=self.cached_data,
            in_memory=self.in_memory
        )

        test_dataset = MuonTomographyDataset(
            data_indices=self.test_indices,
            reader=self.reader,
            feat_transform=test_transform,
            target_transform=x0_transform,
            cached_data=self.cached_data,
            in_memory=self.in_memory
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset
        )

    def get_metadata(self):
        # TODO: to be later extracted from the hdf5 instead of being hardcoded

        return {
            "voxel_shape": (10, 10, 4),  # XYX
            "voxel_size": 0.1,  # in meters
            "voxel_origin": (0.0, 0.0, 0.3),  # in meters
            "n_features": 26,  # 6 xyz detector hits + poca xyz + poca theta + poca uncertainties
        }