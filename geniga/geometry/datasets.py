import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

class SinglePatchIGADataset(Dataset):
    def __init__(self, d:int, h5_files: list[str], return_matrix: bool = False):
        """
        Initializes the GeometryDataset by loading solutions and geo_coeffs from multiple HDF5 files.

        Args:
            d (int): _description_
            h5_files (list[str]): _description_
            return_matrix (bool, optional): _description_. Defaults to False.
            If True, __getitem__ will additionally return the CSR sparse matrix components
            (indptr, indices, values) together with the right-hand side (rhs) vector instead
            of constructing a full sparse matrix object.
        """
        # Load all solutions/geo_coeffs from multiple files and concatenate
        self.d = d
        self.return_matrix = return_matrix
        solutions_list = []
        geo_coeffs_list = []
        mats_data_list = []
        mats_indices_list = []
        mats_indptr_list = []
        rhs_list = []
        for f in h5_files:
            with h5py.File(f, "r") as hf:
                solutions_list.append(hf["solutions"][:])
                geo_coeffs_list.append(hf["geo_coeffs"][:])
                mats_data_list.append(hf["mats_data"][:])
                mats_indices_list.append(hf["mats_indices"][:])
                mats_indptr_list.append(hf["mats_indptr"][:])
                rhs_list.append(hf["rhs"][:])
        self.solutions = np.concatenate(solutions_list, axis=0)
        self.geo_coeffs = np.concatenate(geo_coeffs_list, axis=0)
        self.mats_data = np.concatenate(mats_data_list, axis=0)
        self.mats_indices = np.concatenate(mats_indices_list, axis=0)
        self.mats_indptr = np.concatenate(mats_indptr_list, axis=0)
        self.rhs = np.concatenate(rhs_list, axis=0)
        self.shape = self.solutions.shape[1:]  # Dynamically get the shape (H, W)

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, idx):
        # Access data directly from preloaded arrays
        solution = self.solutions[idx]
        geo_coeffs = self.geo_coeffs[idx]
        solution = torch.tensor(solution, dtype=torch.float32).view(1, *self.shape)  # Reshape dynamically
        geo_coeffs = torch.tensor(geo_coeffs, dtype=torch.float32).view(*self.shape, self.d).permute([1,0])  # Reshape dynamically

        if self.return_matrix:
            indptr = torch.tensor(self.mats_indptr[idx], dtype=torch.int32)
            indices = torch.tensor(self.mats_indices[idx], dtype=torch.int32)
            values = torch.tensor(self.mats_data[idx], dtype=torch.float32)
            rhs = torch.tensor(self.rhs[idx], dtype=torch.float32)
            return geo_coeffs, solution, indptr, indices, values, rhs
        else:
            return geo_coeffs, solution