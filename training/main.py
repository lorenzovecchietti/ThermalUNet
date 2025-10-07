import json
import os
import tempfile
import zipfile

import optuna
import torch
import torch.cuda.amp as amp
import torch.optim as optim
from hyperparams_tuning import objective

from training.data import create_dataloader, create_datasets, device
from training.network import (
    CombinedLoss,
    EarlyStopper,
    UNet,
    evaluate_model,
    train_and_validate,
)
from training.plotting import plot_comparison, plot_loss_history

CONFIG = {
    "DATA": "./../data_generation/simulation_results/dataset.zip",
    "BATCH_SIZE": (16, 32),
    "EPOCHS": 1000,
    "LR": (0.0005, 0.01),
    "EXPO": (4, 8),
    "DROPOUT": (0, 0.25),
    "TRAIN_RATIO": 0.8,
    "VALIDATION_RATIO": 0.05,
    "WORKERS": 0,
    "MODEL_SAVE_PATH": "./nn_results/",
    "TRIALS": 15,
}

T_AMB = 300


def main(temp_dir):
    # Unzip & load data
    with zipfile.ZipFile(CONFIG["DATA"], "r") as zip_ref:
        zip_ref.extractall(temp_dir.name)
    print("Unzipping complete.")

    CONFIG["MAIN_DIRECTORY"] = temp_dir.name
    train_dataset, val_dataset, test_dataset, full_dataset = create_datasets(
        CONFIG["MAIN_DIRECTORY"], CONFIG["TRAIN_RATIO"], CONFIG["VALIDATION_RATIO"]
    )

    model_name = "thermal_unet_model.pth"
    params_name = "best_params.json"
    model_save = os.path.join(CONFIG["MODEL_SAVE_PATH"], model_name)
    params_save = os.path.join(CONFIG["MODEL_SAVE_PATH"], params_name)

    # Optimize & Train
    if not (os.path.isfile(model_save) and os.path.isfile(params_save)):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(
                trial, device, CONFIG, train_dataset, val_dataset, test_dataset
            ),
            n_trials=CONFIG["TRIALS"],
        )

        print("Best hyperparameters: ", study.best_params)
        print("Best validation loss: ", study.best_value)

        best_params = study.best_params
        channels = int(2 ** best_params["expo"] + 0.5)
        net = UNet(channels, best_params["dropout"]).to(device)

        loss_fn = CombinedLoss()

        optimizer = optim.Adam(
            net.parameters(), lr=best_params["lr"], betas=(0.5, 0.999), weight_decay=0.0
        )

        vti_train_dataloader, vti_val_dataloader, vti_test_dataloader = (
            create_dataloader(
                train_dataset,
                val_dataset,
                test_dataset,
                best_params["batch_size"],
                CONFIG["WORKERS"],
            )
        )

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=best_params["lr"],
            epochs=CONFIG["EPOCHS"],
            steps_per_epoch=len(vti_train_dataloader),
        )
        scaler = amp.GradScaler(enabled=torch.cuda.is_available())
        early_stopper = EarlyStopper(patience=100)

        train_loss_hist, val_loss_hist, _ = train_and_validate(
            net,
            loss_fn,
            optimizer,
            scheduler,
            scaler,
            early_stopper,
            vti_train_dataloader,
            vti_val_dataloader,
            CONFIG["EPOCHS"],
            device,
            log_pbar=True,
        )

        torch.save(net.state_dict(), model_save)
        with open(params_save, "w") as f:
            json.dump(best_params, f)
        print(f"\nModel saved to {model_save}")

        plot_loss_history(train_loss_hist, val_loss_hist, CONFIG)
    else:
        with open(params_save, "r") as f:
            best_params = json.load(f)
        channels = int(2 ** best_params["expo"] + 0.5)
        net = UNet(channels, best_params["dropout"]).to(device)
        vti_train_dataloader, vti_val_dataloader, vti_test_dataloader = (
            create_dataloader(
                train_dataset,
                val_dataset,
                test_dataset,
                best_params["batch_size"],
                CONFIG["WORKERS"],
            )
        )

    # Evaluate
    net.load_state_dict(torch.load(model_save))
    evaluate_model(net, vti_val_dataloader, device, full_dataset)

    net.eval()
    k = 0
    with torch.no_grad():
        for inputs, targets in vti_val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)

            for j in range(inputs.size(0)):
                plot_comparison(
                    full_dataset,
                    inputs[j],
                    targets[j],
                    outputs[j],
                    CONFIG,
                    u_free=inputs[j][2][0][0].item(),
                    t_amb=T_AMB,
                    sample_idx=j,
                )
                k += 1


if __name__ == "__main__":
    temp_dir = tempfile.TemporaryDirectory()
    try:
        main(temp_dir)
        temp_dir.cleanup()
    except Exception:
        temp_dir.cleanup()
        raise
