import pickle
import numpy as np
import h5py
from plyfile import PlyData, PlyElement
# import open3d as o3d  # For .ply files
import os


class DatasetReader:
    """
    A class to read various dataset formats including:
    - pickle (.pkl)
    - raw (.raw)
    - HDF5 (.h5)
    - PLY (.ply)
    - NumPy (.npy, .npz)
    """

    def __init__(self, file_path):
        """
        Initialize the reader with the dataset file path.
        """
        self.file_path = file_path
        self.file_ext = os.path.splitext(file_path)[-1].lower()

    def read(self):
        """
        Read the dataset based on its file extension.
        """
        if self.file_ext == ".pkl" or self.file_ext == ".pickle":
            return self._read_pickle()
        elif self.file_ext == ".raw":
            return self._read_raw()
        elif self.file_ext == ".h5":
            return self._read_h5()
        elif self.file_ext == ".ply":
            return self._read_ply()
        elif self.file_ext == ".npy":
            return self._read_npy()
        elif self.file_ext == ".npz":
            return self._read_npz()
        else:
            raise ValueError(f"Unsupported file format: {self.file_ext}")

    def _read_pickle(self):
        """Read a pickle file."""
        with open(self.file_path, "rb") as f:
            data = pickle.load(f)
        return data

    def _read_raw(self, shape, dtype):
        """
        Read a raw binary file.
        - shape: Tuple indicating the dimensions of the data.
        - dtype: Data type of the raw data (e.g., np.uint8, np.float32).
        """
        with open(self.file_path, "rb") as f:
            data = np.fromfile(f, dtype=dtype).reshape(shape)
        return data

    def _read_h5(self):
        """Read an HDF5 (.h5) file."""
        with h5py.File(self.file_path, "r") as f:
            data = {key: np.array(f[key]) for key in f.keys()}
        return data

    def _read_ply(self):
        """Read a PLY file."""
        mesh = PlyData.read(self.file_path)
        return np.asarray(mesh.points)

    def _read_npy(self):
        """Read a NumPy (.npy) file."""
        return np.load(self.file_path)

    def _read_npz(self):
        """Read a compressed NumPy (.npz) file."""
        with np.load(self.file_path) as data:
            return {key: data[key] for key in data.files}

