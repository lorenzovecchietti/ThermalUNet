import torch
from torch import amp, optim

from training.data import create_dataloader
from training.network import CombinedLoss, EarlyStopper, UNet, train_and_validate


def objective(trial, device, config, train_dataset, val_dataset, test_dataset):
    lr = trial.suggest_float("lr", config["LR"][0], config["LR"][1], log=True)
    dropout = trial.suggest_float("dropout", config["DROPOUT"][0], config["DROPOUT"][1])
    expo = trial.suggest_int("expo", config["EXPO"][0], config["EXPO"][1])
    batch_size = trial.suggest_int(
        "batch_size", config["BATCH_SIZE"][0], config["BATCH_SIZE"][1], step=4
    )
    channels = int(2**expo + 0.5)

    vti_train_dataloader, _, vti_test_dataloader = create_dataloader(
        train_dataset, val_dataset, test_dataset, batch_size, config["WORKERS"]
    )

    # Configurazione di Modello, Optimizer, Scheduler, etc.
    net = UNet(channels, dropout).to(device)
    loss_fn = CombinedLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=config["EPOCHS"],
        steps_per_epoch=len(vti_train_dataloader),
    )
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())
    early_stopper = EarlyStopper(patience=10)

    # Chiama la funzione di addestramento estratta
    _, _, min_val_loss = train_and_validate(
        net,
        loss_fn,
        optimizer,
        scheduler,
        scaler,
        early_stopper,
        vti_train_dataloader,
        vti_test_dataloader,
        config["EPOCHS"],
        device,
        log_pbar=False,  # Nessun logging verbose per Optuna
    )

    return min_val_loss
