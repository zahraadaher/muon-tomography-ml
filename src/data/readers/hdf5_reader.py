import h5py
import numpy as np
from typing import Any, Dict

class HDF5Reader:

    def __init__(self, path):
        self.path = path
        self._h5f = None

    def __enter__(self):
        self._h5f = h5py.File(self.path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._h5f is not None:
            self._h5f.close()
            self._h5f = None

    def _file(self):
        # allows use both with and without the context manager
        return self._h5f if self._h5f is not None else h5py.File(self.path, "r")

    def get_sample(self, scan_name: str)-> Dict[str, Any]:
        h5f = self._file()
        grp = h5f["data"][scan_name]
        sample = {
            "features": grp["features"][:],
            "target": grp["gt"][:] if "gt" in grp else None,
            "prediction": grp["preds"][:] if "preds" in grp else None,
            "material_id": int(grp.attrs.get("material_id", -1)),
            "mask": 0
        }
        if self._h5f is None:
            h5f.close()
        return sample

    def get_metadata(self, scan_name):
        h5f = self._file()
        grp = h5f["data"][scan_name]
        meta = {
            "material_id": int(grp.attrs.get("material_id", -1)),
            "position": int(grp.attrs.get("position", -1)),
        }
        if self._h5f is None:
            h5f.close()
        return meta

    def list_samples(self):
        h5f = self._file()
        names = list(h5f["data"].keys())
        if self._h5f is None:
            h5f.close()
        return names