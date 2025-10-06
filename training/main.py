import os
import random
import sys
import json
import tempfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ==============================================================================
# --- 1. Configuration is and Hyperparameters ---
# ==============================================================================
"""
Centralized configuration for easy management of hyperparameters and paths.
"""
CONFIG = {
    "DATA": "./../data_generation/simulation_results/dataset.zip",
    "BATCH_SIZE": 10,
    "EPOCHS": 300,
    "LR": 0.05,
    "EXPO": 4,
    "DROPOUT": 0.1,
    "TRAIN_RATIO": 0.7,
    "VALIDATION_RATIO": 0.15,
    "MODEL_SAVE_PATH": "thermal_unet_model.pth",
}

CONFIG["TEST_RATIO"] = 1.0 - CONFIG["TRAIN_RATIO"] - CONFIG["VALIDATION_RATIO"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_device("cuda")

# ==============================================================================
# --- 2. Dataset and DataLoaders ---
# ==============================================================================


class VTI_Dataset(Dataset):
    """
    Custom PyTorch Dataset for VTI files, optimized for in-memory loading
    and applying Min-Max normalization to output channels (T, Ux, Uy).
    """

    def __init__(self, file_paths: list[str], json_paths: list[str]):
        self.data = []
        self.min_vals = None
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
                global_mins[0] = min(global_mins[0], T_data.min())
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
        print(f"Min values (T, Ux, Uy): {self.min_vals.numpy()}")
        print(f"Max values (T, Ux, Uy): {self.max_vals.numpy()}")
        print("---------------------------\n")

        print(f"Loading and normalizing {len(file_paths)} samples into memory...")
        for idx, grid in enumerate(tqdm(grids)):
            dims_reordered = grid.dimensions[::-1]

            # Inputs
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

            # Targets and Normalization
            T_tensor = torch.from_numpy(
                grid.point_data["T"].reshape(dims_reordered)
            ).float()
            U_data = grid.point_data["U"]
            Ux_tensor = torch.from_numpy(U_data[:, 0].reshape(dims_reordered)).float()
            Uy_tensor = torch.from_numpy(U_data[:, 1].reshape(dims_reordered)).float()

            unnormalized_targets = torch.stack([T_tensor, Ux_tensor, Uy_tensor], dim=0)
            normalized_targets = (
                unnormalized_targets - self.min_vals.view(3, 1, 1, 1)
            ) / self.range_vals.view(3, 1, 1, 1)

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
        return self.data[idx]


# ==============================================================================
# --- 3. Model Architecture (U-Net) ---
# ==============================================================================


class EncoderUnit(nn.Module):
    def __init__(self, in_channels, out_channels, size, pad, dropout):
        super(EncoderUnit, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=size,
            stride=2,
            padding=pad,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        return self.leakyrelu(self.dropout(self.bn(self.conv(x))))


class DecoderUnit(nn.Module):
    def __init__(self, in_channels, out_channels, size, pad, dropout):
        super(DecoderUnit, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=size,
            stride=1,
            padding=pad,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.dropout(self.bn(self.conv(self.up(x)))))


class LastDecoderUnit(nn.Module):
    def __init__(self, in_channels, out_channels, size, pad, dropout, out_pad):
        super(LastDecoderUnit, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=size,
            stride=2,
            padding=pad,
            output_padding=out_pad,
            bias=True,
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.conv_transpose(x))


class UNet(nn.Module):
    """U-Net architecture for flow field prediction."""

    def __init__(self, channels, dropout):
        super(UNet, self).__init__()

        # Encoder Path
        self.encoders = nn.ModuleList(
            [
                EncoderUnit(3, channels, dropout=dropout, size=4, pad=1),
                EncoderUnit(channels, channels * 2, dropout=dropout, size=4, pad=1),
                EncoderUnit(channels * 2, channels * 2, dropout=dropout, size=4, pad=1),
                EncoderUnit(channels * 2, channels * 4, dropout=dropout, size=4, pad=1),
                EncoderUnit(channels * 4, channels * 8, dropout=dropout, size=2, pad=1),
                EncoderUnit(channels * 8, channels * 8, dropout=dropout, size=1, pad=1),
                EncoderUnit(
                    channels * 8, channels * 8, dropout=dropout, size=1, pad=1
                ),  # Bottleneck
            ]
        )

        # Decoder Path (with skip connections)
        self.decoders = nn.ModuleList(
            [
                DecoderUnit(
                    channels * 8, channels * 8, dropout=dropout, size=(6, 3), pad=(1, 0)
                ),
                DecoderUnit(
                    channels * 16, channels * 8, dropout=dropout, size=(3, 4), pad=0
                ),
                DecoderUnit(
                    channels * 16, channels * 4, dropout=dropout, size=(3, 2), pad=0
                ),
                DecoderUnit(channels * 8, channels * 2, dropout=dropout, size=1, pad=0),
                DecoderUnit(
                    channels * 4, channels * 2, dropout=dropout, size=(2, 1), pad=(1, 0)
                ),
                DecoderUnit(channels * 4, channels, dropout=dropout, size=3, pad=1),
            ]
        )

        # Final block
        self.last_decoder = LastDecoderUnit(
            channels * 2, 3, dropout=dropout, size=4, pad=1, out_pad=0
        )
        self.fc = nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0)
        self.bn_input = nn.BatchNorm2d(3)

    def forward(self, input):
        x = self.bn_input(input)
        encoder_outputs = [x]
        for layer in self.encoders:
            x = layer(x)
            encoder_outputs.append(x)

        encoder_outputs.pop()  # Last output is the decoder's input
        for layer in self.decoders:
            x = layer(x)
            x = torch.cat([x, encoder_outputs.pop()], 1)

        x = self.last_decoder(x)
        x = torch.cat([x, encoder_outputs.pop()], 1)
        x = self.fc(x)

        return x


# ==============================================================================
# --- 4. Evaluation and Visualization Functions ---
# ==============================================================================


def plot_loss_history(train_loss, val_loss):
    """Plots the training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, "b", label="Training Loss")
    plt.plot(val_loss, "g", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Loss History")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.savefig("losshistory.png")


def psnr(target, prediction, max_val=1.0):
    """Calculates the Peak Signal-to-Noise Ratio."""
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def evaluate_model(model, dataloader, device, dataset):
    """Evaluates the model on the test set, calculating L1, MSE, and PSNR."""
    model.eval()
    total_l1, total_mse, total_psnr = 0, 0, 0
    criterion_mse = nn.MSELoss()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            outputs = model(inputs)

            # Denormalize to calculate metrics in the original domain
            targets_denorm = dataset.denormalize(targets)
            outputs_denorm = dataset.denormalize(outputs)

            total_l1 += nn.L1Loss()(outputs_denorm, targets_denorm).item()
            total_mse += criterion_mse(outputs_denorm, targets_denorm).item()

            for i in range(targets.size(0)):
                max_range = dataset.range_vals.max().item()
                total_psnr += psnr(
                    targets_denorm[i], outputs_denorm[i], max_val=max_range
                )

    avg_l1 = total_l1 / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_psnr = total_psnr / len(dataset)

    print("\n--- Test Set Evaluation Metrics ---")
    print(f"Average L1 Loss: {avg_l1:.5f}")
    print(f"Average MSE: {avg_mse:.5f}")
    print(f"Average PSNR: {avg_psnr:.3f} dB")
    print("----------------------------------")


def plot_comparison(dataset, inputs, targets, outputs, sample_idx=0):
    """
    Plots a 4-column comparison: Target, Prediction, Absolute Error, Relative Error.
    """
    fields = ["T", "Ux", "Uy"]

    # Denormalize for visualization
    targets_denorm = dataset.denormalize(targets.unsqueeze(0))[0].cpu().numpy()
    outputs_denorm = dataset.denormalize(outputs.unsqueeze(0))[0].cpu().numpy()

    error_abs = np.abs(targets_denorm - outputs_denorm)
    error_rel = error_abs / (
        np.abs(targets_denorm).max() + 1e-8
    )  # Vel error relative to max value
    error_rel[0] = error_abs[0] / (
        np.abs(targets_denorm[0]) + 1e-8
    )  # Temp error relative to same cell

    fig, axes = plt.subplots(len(fields), 4, figsize=(20, 10))

    for i, field in enumerate(fields):
        vmin = min(targets_denorm[i].min(), outputs_denorm[i].min())
        vmax = max(targets_denorm[i].max(), outputs_denorm[i].max())

        # Target
        im = axes[i, 0].imshow(targets_denorm[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Target: {field}")
        fig.colorbar(im, ax=axes[i, 0])

        # Prediction
        im = axes[i, 1].imshow(outputs_denorm[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"Prediction: {field}")
        fig.colorbar(im, ax=axes[i, 1])

        # Absolute Error
        im = axes[i, 2].imshow(error_abs[i], cmap="inferno")
        axes[i, 2].set_title("Absolute Error")
        fig.colorbar(im, ax=axes[i, 2])

        # Relative Error
        im = axes[i, 3].imshow(
            error_rel[i], cmap="inferno", vmin=0, vmax=0.1
        )  # max 10% error
        axes[i, 3].set_title("Relative Error")
        fig.colorbar(im, ax=axes[i, 3])

    plt.suptitle(f"Comparison for Test Sample {sample_idx}")
    plt.tight_layout()
    plt.savefig("TestSet.png")


# ==============================================================================
# --- 5. Main Execution Block ---
# ==============================================================================


def main(temp_dir):
    """Main function to run the entire pipeline."""

    # --- Data Loading and Preparation ---
    with zipfile.ZipFile(CONFIG["DATA"], "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    print("Unzipping complete.")
    CONFIG["MAIN_DIRECTORY"] = os.path.join(
        temp_dir, os.path.splitext(CONFIG["DATA"])[0]
    )
    vti_files = sorted(
        [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(temp_dir)
            for f in fn
            if f.endswith(".vti")
        ]
    )
    json_files = sorted(
        [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(temp_dir)
            for f in fn
            if f.endswith(".json")
        ]
    )
    full_dataset = VTI_Dataset(vti_files, json_files)

    dataset_size = len(full_dataset)
    train_size = int(CONFIG["TRAIN_RATIO"] * dataset_size)
    val_size = int(CONFIG["VALIDATION_RATIO"] * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    vti_train_dataloader = DataLoader(
        train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=2
    )
    vti_val_dataloader = DataLoader(
        val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2
    )
    vti_test_dataloader = DataLoader(
        test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2
    )

    print(f"\nDataset split into:")
    print(f"- Training: {len(train_dataset)} samples")
    print(f"- Validation: {len(val_dataset)} samples")
    print(f"- Test: {len(test_dataset)} samples\n")

    # --- Model Initialization ---
    channels = int(2 ** CONFIG["EXPO"] + 0.5)
    net = UNet(channels, CONFIG["DROPOUT"]).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        net.parameters(), lr=CONFIG["LR"], betas=(0.5, 0.999), weight_decay=0.0
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    nn_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print(f"Trainable params: {sum([np.prod(p.size()) for p in nn_parameters])}\n")

    # --- Training Loop ---
    train_loss_hist = []
    val_loss_hist = []

    pbar = tqdm(range(CONFIG["EPOCHS"]))
    for epoch in pbar:
        # Training
        net.train()
        train_loss_acc = 0
        for inputs, targets in vti_train_dataloader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            optimizer.zero_grad()
            with amp.autocast(enabled=torch.cuda.is_available()):
                outputs = net(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_acc += loss.item()

        train_loss_hist.append(train_loss_acc / len(vti_train_dataloader))

        # Validation
        net.eval()
        val_loss_acc = 0
        with torch.no_grad():
            for inputs, targets in vti_val_dataloader:
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                with amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = net(inputs)
                    val_loss_acc += loss_fn(outputs, targets).item()

        val_loss_hist.append(val_loss_acc / len(vti_val_dataloader))
        scheduler.step(val_loss_hist[-1])

        pbar.set_description(
            f"Train Loss: {train_loss_hist[-1]:.5f} | Val Loss: {val_loss_hist[-1]:.5f} | LR: {optimizer.param_groups[0]['lr']:.5f}"
        )

    # --- Save the model ---
    torch.save(net.state_dict(), CONFIG["MODEL_SAVE_PATH"])
    print(f"\nModel saved to {CONFIG['MODEL_SAVE_PATH']}")

    # --- Evaluation and Visualization ---
    plot_loss_history(train_loss_hist, val_loss_hist)

    # Load the best model and evaluate on the test set
    net.load_state_dict(torch.load(CONFIG["MODEL_SAVE_PATH"]))
    evaluate_model(net, vti_test_dataloader, device, full_dataset)

    # Visualize some results from the test set
    net.eval()
    with torch.no_grad():
        inputs, targets = next(iter(vti_test_dataloader))
        inputs, targets = inputs.to(device).float(), targets.to(device).float()
        outputs = net(inputs)

        for j in range(min(3, inputs.size(0))):
            plot_comparison(
                full_dataset, inputs[j], targets[j], outputs[j], sample_idx=j
            )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        main(tmp)
