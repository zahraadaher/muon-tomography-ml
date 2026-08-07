import h5py


class HDF5Reader:

    def __init__(self, path):
        self.path = path

    def get_sample(self, scan_name):

        with h5py.File(self.path, "r") as h5f:

            grp = h5f["data"][scan_name]

            return {
                "features": grp["features"][:],

                "target": (
                    grp["gt"][:]
                    if "gt" in grp
                    else None
                ),

                "prediction": (
                    grp["preds"][:]
                    if "preds" in grp
                    else None
                ),

                "material_id": int(
                    grp.attrs.get(
                        "material_id",
                        -1
                    )
                )
            }

    def get_metadata(self, scan_name):

        with h5py.File(self.path, "r") as h5f:

            grp = h5f["data"][scan_name]

            return {
                "material_id": int(
                    grp.attrs.get(
                        "material_id",
                        -1
                    )
                ),

                "position": int(
                    grp.attrs.get(
                        "position",
                        -1
                    )
                )
            }

    def list_samples(self):

        with h5py.File(self.path, "r") as h5f:
            return list(h5f["data"].keys())