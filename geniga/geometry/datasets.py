import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from bisect import bisect_right

class SinglePatchIGADataset(Dataset):
    def __init__(self, d:int, h5_files: list[str], return_bc_data: bool = False, in_memory: bool = True):
        """
        Initializes the dataset for IGA samples from one or more HDF5 files.

        Args:
            d (int): Spatial dimension (e.g., 2)
            h5_files (list[str]): Paths to HDF5 files to read from
            return_bc_data (bool): If True, __getitem__ also returns boundary condition data and rhs
            in_memory (bool): If True, load all arrays into RAM on init; if False, read lazily per-sample
        """
        self.d = d
        self.return_bc_data = return_bc_data
        self.in_memory = in_memory
        self.h5_files = list(h5_files)

        self.has_wave_numbers = False
        self.has_bc_data = False
        self.shape = None  # Will be set for in-memory mode; for lazy mode we keep flattened representation

        if self.in_memory:
            # Eagerly load all arrays into memory
            solutions_list = []
            geo_coeffs_list = []
            rhs_list = []
            bc_indices_list = []
            bc_values_list = []
            wave_numbers_list = []

            for f in self.h5_files:
                with h5py.File(f, "r") as hf:
                    solutions_list.append(hf["solutions"][:])
                    geo_coeffs_list.append(hf["geo_coeffs"][:])
                    
                    # Load RHS if present
                    if "rhs" in hf:
                        rhs_list.append(hf["rhs"][:])
                    
                    # Load BC data if present
                    if "bc_indices" in hf and "bc_values" in hf:
                        bc_indices_list.append(hf["bc_indices"][:])
                        bc_values_list.append(hf["bc_values"][:])
                        self.has_bc_data = True

                    if "wave_numbers" in hf:
                        wave_numbers_list.append(hf["wave_numbers"][:])
                        self.has_wave_numbers = True

            self.solutions = np.concatenate(solutions_list, axis=0)
            self.geo_coeffs = np.concatenate(geo_coeffs_list, axis=0)
            
            # RHS data
            if rhs_list:
                self.rhs = np.concatenate(rhs_list, axis=0)
            else:
                self.rhs = None
            
            # BC data
            if self.has_bc_data:
                self.bc_indices = np.concatenate(bc_indices_list, axis=0)
                self.bc_values = np.concatenate(bc_values_list, axis=0)
            else:
                self.bc_indices = None
                self.bc_values = None

            if self.has_wave_numbers:
                self.wave_numbers = np.concatenate(wave_numbers_list, axis=0)
            else:
                self.wave_numbers = None

            # solutions are stored flattened; keep shape=(prodN,) for compatibility
            self.shape = self.solutions.shape[1:]
        else:
            # Lazy mode: build an index over files, and defer reads to __getitem__
            self._file_lengths = []  # number of samples per file
            self._cumulative = []    # cumulative counts for idx mapping
            total = 0
            flat_points = None
            for f in self.h5_files:
                with h5py.File(f, "r") as hf:
                    n = hf["solutions"].shape[0]
                    self._file_lengths.append(n)
                    total += n
                    self._cumulative.append(total)
                    if "wave_numbers" in hf:
                        self.has_wave_numbers = True
                    if "bc_indices" in hf and "bc_values" in hf:
                        self.has_bc_data = True
                    # Determine flattened spatial size once
                    if flat_points is None:
                        # geo_coeffs stored as (n_samples, d * prodN)
                        flat_points = hf["geo_coeffs"].shape[1] // self.d
            self._total_len = total
            self._flat_points = flat_points
            # per-process cache for open file handles (created lazily in workers)
            self._handle_cache = {}

    def __len__(self):
        if self.in_memory:
            return len(self.solutions)
        return self._total_len

    def __getitem__(self, idx):
        if self.in_memory:
            # Access data directly from preloaded arrays
            solution = self.solutions[idx]
            geo_coeffs = self.geo_coeffs[idx]
            solution = torch.tensor(solution, dtype=torch.float32).view(1, *self.shape)
            geo_coeffs = torch.tensor(geo_coeffs, dtype=torch.float32).view(*self.shape, self.d).permute([1,0])

            if self.return_bc_data:
                result = [geo_coeffs, solution]
                
                if self.has_bc_data:
                    bc_indices = torch.tensor(self.bc_indices[idx], dtype=torch.int32)
                    bc_values = torch.tensor(self.bc_values[idx], dtype=torch.float32)
                    result.extend([bc_indices, bc_values])
                
                if self.rhs is not None:
                    rhs = torch.tensor(self.rhs[idx], dtype=torch.float32)
                    result.append(rhs)
                
                if self.has_wave_numbers:
                    wave_number = torch.tensor(self.wave_numbers[idx], dtype=torch.float32)
                    result.append(wave_number)
                
                return tuple(result)
            else:
                if self.has_wave_numbers:
                    wave_number = torch.tensor(self.wave_numbers[idx], dtype=torch.float32)
                    return geo_coeffs, solution, wave_number
                else:
                    return geo_coeffs, solution
        else:
            # Lazy mode: map global idx to file and local index
            file_pos = bisect_right(self._cumulative, idx)
            file_start = 0 if file_pos == 0 else self._cumulative[file_pos - 1]
            idx_in_file = idx - file_start
            fpath = self.h5_files[file_pos]

            # Open or reuse cached handle (safe per-process)
            hf = self._handle_cache.get(fpath)
            if hf is None or not hf.id.valid:
                hf = h5py.File(fpath, "r")
                self._handle_cache[fpath] = hf

            # Read one row per dataset
            sol_np = hf["solutions"][idx_in_file]
            geo_np = hf["geo_coeffs"][idx_in_file]

            # Convert to tensors and match in-memory output shapes: (1, prodN) and (d, prodN)
            solution = torch.from_numpy(sol_np).to(torch.float32).view(1, -1)
            geo_coeffs = torch.from_numpy(geo_np).to(torch.float32).view(self._flat_points, self.d).permute(1, 0)

            if self.return_bc_data:
                result = [geo_coeffs, solution]
                
                if self.has_bc_data:
                    bc_indices = torch.from_numpy(hf["bc_indices"][idx_in_file]).to(torch.int32)
                    bc_values = torch.from_numpy(hf["bc_values"][idx_in_file]).to(torch.float32)
                    result.extend([bc_indices, bc_values])
                
                if "rhs" in hf:
                    rhs = torch.from_numpy(hf["rhs"][idx_in_file]).to(torch.float32)
                    result.append(rhs)
                
                if self.has_wave_numbers:
                    wave_number = torch.from_numpy(hf["wave_numbers"][idx_in_file]).to(torch.float32)
                    result.append(wave_number)
                
                return tuple(result)
            else:
                if self.has_wave_numbers:
                    wave_number = torch.from_numpy(hf["wave_numbers"][idx_in_file]).to(torch.float32)
                    return geo_coeffs, solution, wave_number
                else:
                    return geo_coeffs, solution

    def __del__(self):
        # Ensure any cached file handles are closed
        if hasattr(self, "_handle_cache") and isinstance(self._handle_cache, dict):
            for hf in self._handle_cache.values():
                try:
                    if hf is not None and hf.id.valid:
                        hf.close()
                except Exception:
                    pass