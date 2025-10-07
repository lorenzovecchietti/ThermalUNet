import os

import numpy as np
from matplotlib import pyplot as plt


def plot_loss_history(train_loss, val_loss, config):
    """Plots the training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, "b", label="Training Loss")
    plt.plot(val_loss, "g", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Combined Loss")
    plt.title("Loss History")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.savefig(os.path.join(config["MODEL_SAVE_PATH"], "losshistory.png"))


def plot_comparison(dataset, inputs, targets, outputs,config, u_free, t_amb, sample_idx=0):
    """
    Plots a 4-column comparison: Target, Prediction, Absolute Error, Relative Error.
    """
    fields = ["T", "Ux", "Uy"]

    # Denormalize for visualization
    # Ensure inputs, targets, and outputs are on CPU before converting to numpy
    targets_denorm = dataset.denormalize(targets.unsqueeze(0))[0].cpu().numpy()
    outputs_denorm = dataset.denormalize(outputs.unsqueeze(0))[0].cpu().numpy()

    error_abs = np.abs(targets_denorm - outputs_denorm)
    error_rel = np.zeros_like(error_abs, dtype=float)
    error_rel[0] = error_abs[0] / (np.abs(targets_denorm[0]).max() - t_amb)
    error_rel[1] = error_abs[1] / u_free
    error_rel[2] = error_abs[2] / np.abs(targets_denorm[2]).max()
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
    plt.savefig(os.path.join(config["MODEL_SAVE_PATH"], f"TestSet_{sample_idx}.png"))
