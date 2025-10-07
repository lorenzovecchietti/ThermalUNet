import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        act=nn.ReLU,
        leaky_slope=0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        if act is nn.ReLU:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """Conv 3x3 + BN + Act  -> Conv 3x3 + BN + Act"""

    def __init__(self, in_ch, out_ch, act=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, kernel_size=3, stride=1, padding=1, act=act),
            ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1, act=act),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------
# Encoder (downsampling)
# -----------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, act=nn.LeakyReLU):
        super().__init__()
        self.down = ConvBNAct(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1, act=act
        )
        self.refine = DoubleConv(out_ch, out_ch, act=nn.LeakyReLU)
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.down(x)
        x = self.drop(x)
        x = self.refine(x)
        return x


# -----------------------
# Attention Gate
# -----------------------
class AttentionGate(nn.Module):
    """
    Standard attention gate:
      g -> gating (decoder feature)
      x -> skip connection (encoder feature)
    returns: x * attention_mask (same channels as x)
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(
                psi, size=x.shape[2:], mode="bilinear", align_corners=True
            )
        return x * psi


# -----------------------
# Decoder (upsample + attention + refine)
# -----------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        """
        in_ch: channels of the input feature (coming from previous decoder / bottleneck)
        skip_ch: channels of the encoder skip connection to concatenate
        out_ch: desired output channels after the block (typically = skip_ch)
        """
        super().__init__()
        self.up_conv = ConvBNAct(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1, act=nn.ReLU
        )
        self.attn = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=max(1, out_ch // 2))
        self.double_conv = DoubleConv(out_ch + skip_ch, out_ch, act=nn.ReLU)
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = self.up_conv(x)
        x = self.drop(x)
        skip_att = self.attn(g=x, x=skip)
        x = torch.cat([x, skip_att], dim=1)
        x = self.double_conv(x)
        return x


# -----------------------
# UNet with Attention
# -----------------------
class UNet(nn.Module):
    def __init__(self, base_channels=64, dropout=0.0, in_channels=3, out_channels=3):
        """
        base_channels: numero di canali di base (es. 32, 64)
        dropout: dropout between blocchi (Dropout2d)
        in_channels: canali in input (default 3)
        out_channels: canali di output (default 3)
        """
        super().__init__()
        c = base_channels

        self.conv_in = DoubleConv(in_channels, c, act=nn.LeakyReLU)

        self.enc1 = EncoderBlock(c, c * 2, dropout=dropout, act=nn.LeakyReLU)
        self.enc2 = EncoderBlock(c * 2, c * 4, dropout=dropout, act=nn.LeakyReLU)
        self.enc3 = EncoderBlock(c * 4, c * 8, dropout=dropout, act=nn.LeakyReLU)
        self.enc4 = EncoderBlock(
            c * 8, c * 8, dropout=dropout, act=nn.LeakyReLU
        )  # keep top channels = 8c

        self.bottleneck = DoubleConv(c * 8, c * 8, act=nn.LeakyReLU)

        # -----------------------
        # DECODER VELOCITY
        # -----------------------
        self.dec4_vel = DecoderBlock(
            in_ch=c * 8, skip_ch=c * 8, out_ch=c * 8, dropout=dropout
        )
        self.dec3_vel = DecoderBlock(
            in_ch=c * 8, skip_ch=c * 4, out_ch=c * 4, dropout=dropout
        )
        self.dec2_vel = DecoderBlock(
            in_ch=c * 4, skip_ch=c * 2, out_ch=c * 2, dropout=dropout
        )
        self.dec1_vel = DecoderBlock(
            in_ch=c * 2, skip_ch=c * 1, out_ch=c * 1, dropout=dropout
        )
        self.final_conv_vel = nn.Conv2d(c, 2, kernel_size=1, stride=1, padding=0)

        # -----------------------
        # DECODER TEMPERATURE
        # -----------------------
        self.dec4_temp = DecoderBlock(
            in_ch=c * 8, skip_ch=c * 8, out_ch=c * 8, dropout=dropout
        )
        self.dec3_temp = DecoderBlock(
            in_ch=c * 8, skip_ch=c * 4, out_ch=c * 4, dropout=dropout
        )
        self.dec2_temp = DecoderBlock(
            in_ch=c * 4, skip_ch=c * 2, out_ch=c * 2, dropout=dropout
        )
        self.dec1_temp = DecoderBlock(
            in_ch=c * 2, skip_ch=c * 1, out_ch=c * 1, dropout=dropout
        )
        self.final_conv_temp = nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x0 = self.conv_in(x)  # S0 (same spatial as input)
        x1 = self.enc1(x0)  # S1 (down x2)
        x2 = self.enc2(x1)  # S2 (down x4)
        x3 = self.enc3(x2)  # S3 (down x8)
        x4 = self.enc4(x3)  # S4 (down x16)

        # Bottleneck
        b = self.bottleneck(x4)

        # Decoder Vel
        d_vel = self.dec4_vel(b, x3)
        d_vel = self.dec3_vel(d_vel, x2)
        d_vel = self.dec2_vel(d_vel, x1)
        d_vel = self.dec1_vel(d_vel, x0)

        out_vel = self.final_conv_vel(d_vel)

        # Decoder Temperature
        d_temp = self.dec4_temp(b, x3)
        d_temp = self.dec3_temp(d_temp, x2)
        d_temp = self.dec2_temp(d_temp, x1)
        d_temp = self.dec1_temp(d_temp, x0)

        out_temp = self.final_conv_temp(d_temp)
        out = torch.cat([out_temp, out_vel], dim=1)
        return out


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, outputs, targets):
        return self.mse.forward(outputs, targets)


# ==============================================================================
# --- Early Stopping ---
# ==============================================================================


class EarlyStopper:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_and_validate(
    net,
    loss_fn,
    optimizer,
    scheduler,
    scaler,
    early_stopper,
    train_dataloader,
    val_dataloader,
    epochs,
    device,
    log_pbar=False,
):
    train_loss_hist = []
    val_loss_hist = []
    min_val_loss = float("inf")

    iterator = tqdm(range(epochs)) if log_pbar else range(epochs)

    for _ in iterator:
        net.train()
        train_loss_acc = 0
        for inputs, targets in train_dataloader:
            noise = torch.randn_like(inputs)
            inputs = inputs + noise
            optimizer.zero_grad()
            with amp.autocast(
                device_type=str(device), enabled=torch.cuda.is_available()
            ):
                outputs = net(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_acc += loss.item()
            scheduler.step()

        train_loss_hist.append(train_loss_acc / len(train_dataloader))

        # --- Validation ---
        net.eval()
        val_loss_acc = 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                with amp.autocast(
                    device_type=str(device), enabled=torch.cuda.is_available()
                ):
                    outputs = net(inputs)
                    val_loss_acc += loss_fn(outputs, targets).item()

        val_loss = val_loss_acc / len(val_dataloader)
        val_loss_hist.append(val_loss)

        if log_pbar:
            iterator.set_description(
                f"Train Loss: {train_loss_hist[-1]:.5f} | Val Loss: {val_loss_hist[-1]:.5f} | LR: {optimizer.param_groups[0]['lr']:.5f}"
            )

        if val_loss < min_val_loss:
            min_val_loss = val_loss

        if early_stopper.early_stop(val_loss):
            break

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return train_loss_hist, val_loss_hist, min_val_loss


def psnr(target, prediction, max_val=1.0):
    """Calculates the Peak Signal-to-Noise Ratio."""
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def evaluate_model(model, dataloader, device, dataset):
    model.eval()
    total_l1, total_mse, total_psnr = 0, 0, 0
    criterion_mse = nn.MSELoss()

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)

            targets_denorm = dataset.denormalize(targets.cpu())
            outputs_denorm = dataset.denormalize(outputs.cpu())

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
