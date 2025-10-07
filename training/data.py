import json
import os

import pandas as pd
import pyvista as pv
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# Define device
if torch.cuda.is_available():
    try:
        assert torch.cuda.get_device_name(1)
        device = torch.device("cuda:1")
    except:
        device = torch.device("cuda")
else:
    device = torch.device("cpu")


class VTI_Dataset(Dataset):
    """
    Load and processes VTI data (2D data).
    Applying Min-Max normalization to output channels (T, Ux, Uy).
    """

    def __init__(self, file_paths: list[str], json_paths: list[str]):
        self.data = []
        self.min_vals = None
        self.max_vals = None
        self.range_vals = None
        self.load_data_into_memory(file_paths, json_paths)

    def load_data_into_memory(self, file_paths: list[str], json_paths: list[str]):
        """Load, process, and normalize all data samples into memory."""
        global_mins = [float("inf")] * 3
        global_maxs = [float("-inf")] * 3
        grids = []

        print(f"Reading {len(file_paths)} files to determine normalization stats...")
        for file_path in tqdm(file_paths):
            grid = pv.read(file_path)
            grids.append(grid)
            T_data = grid.point_data.get("T")
            U_data = grid.point_data.get("U")

            if T_data is not None:
                global_mins[0] = 300  # Ambient temp
                global_maxs[0] = max(global_maxs[0], T_data.max())

            if U_data is not None:
                global_mins[1] = min(global_mins[1], U_data[:, 0].min())
                global_maxs[1] = max(global_maxs[1], U_data[:, 0].max())
                global_mins[2] = min(global_mins[2], U_data[:, 1].min())
                global_maxs[2] = max(global_maxs[2], U_data[:, 1].max())

        self.min_vals = torch.tensor(global_mins, dtype=torch.float32)
        self.max_vals = torch.tensor(global_maxs, dtype=torch.float32)
        self.range_vals = self.max_vals - self.min_vals

        print("\n--- Normalization Stats ---")
        print(f"Min values (T, Ux, Uy): {self.min_vals.cpu().numpy()}")
        print(f"Max values (T, Ux, Uy): {self.max_vals.cpu().numpy()}")
        print("---------------------------\n")

        print(f"Loading and normalizing {len(file_paths)} samples into memory...")
        for idx, grid in enumerate(tqdm(grids)):
            dims_reordered = grid.dimensions[::-1]

            with open(json_paths[idx], "r") as f:
                u_bc_val = json.load(f)["u"]

            cond_tensor = torch.from_numpy(
                grid.point_data["conducibility"].reshape(dims_reordered)
            ).float()
            power_tensor = torch.from_numpy(
                grid.point_data["power"].reshape(dims_reordered)
            ).float()
            u_bc_tensor = torch.full_like(cond_tensor, u_bc_val)
            input_tensor = torch.stack([cond_tensor, power_tensor, u_bc_tensor], dim=0)

            T_tensor = torch.from_numpy(
                grid.point_data["T"].reshape(dims_reordered)
            ).float()

            U_data = grid.point_data["U"]  # U_data è (N, 3) dove N = Nx*Ny

            Ux_tensor = torch.from_numpy(U_data[:, 0].reshape(dims_reordered)).float()
            Uy_tensor = torch.from_numpy(U_data[:, 1].reshape(dims_reordered)).float()

            unnormalized_targets = torch.stack([T_tensor, Ux_tensor, Uy_tensor], dim=0)

            min_vals_view = self.min_vals.view(3, 1, 1, 1).cpu()
            range_vals_view = self.range_vals.view(3, 1, 1, 1).cpu()

            normalized_targets = (
                unnormalized_targets - min_vals_view
            ) / range_vals_view

            self.data.append((input_tensor.squeeze(1), normalized_targets.squeeze(1)))

    def denormalize(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        if self.min_vals is None or self.range_vals is None:
            raise AttributeError("Normalization stats not set.")

        min_view = self.min_vals.to(normalized_tensor.device).view(1, 3, 1, 1)
        range_view = self.range_vals.to(normalized_tensor.device).view(1, 3, 1, 1)

        return normalized_tensor * range_view + min_view

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = self.data[idx]
        return inputs.to(device), targets.to(device)


def create_datasets(path, train_ratio, validation_ratio):
    test_ratio = 1.0 - train_ratio - validation_ratio
    vti_files = sorted(
        [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(path)
            for f in fn
            if f.endswith(".vti")
        ]
    )
    json_files = sorted(
        [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(path)
            for f in fn
            if f.endswith(".json")
        ]
    )

    u_values = []
    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)
            u_values.append(data["u"])

    labels = pd.qcut(u_values, q=5, labels=False, duplicates="drop")

    full_dataset = VTI_Dataset(vti_files, json_files)

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(range(len(full_dataset)), labels))

    labels_train_val = labels[train_val_idx]
    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_ratio / (train_ratio + validation_ratio),
        random_state=42,
    )
    train_idx_rel, val_idx_rel = next(
        sss_val.split(range(len(train_val_idx)), labels_train_val)
    )
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    print("\nDataset split into:")
    print(f"- Training: {len(train_dataset)} samples")
    print(f"- Validation: {len(val_dataset)} samples")
    print(f"- Test: {len(test_dataset)} samples\n")

    return train_dataset, val_dataset, test_dataset, full_dataset


def create_dataloader(train_dataset, val_dataset, test_dataset, batch_size, workers):
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
