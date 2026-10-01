import os
import yaml
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import ASVspoofDataset
from src.models.lcnn import LCNN
from src.metrics import compute_eer


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    avg_loss = 0.0
    for spectrogram, label, _ in tqdm(dataloader, desc="Training"):
        spectrogram = spectrogram.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(spectrogram, target=label)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        avg_loss += loss.item()

    return avg_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    avg_loss = 0.0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for spectrogram, label, _ in dataloader:
            spectrogram = spectrogram.to(device)
            label = label.to(device)

            logits = model(spectrogram)
            loss = criterion(logits, label)

            scores = (logits[:, 0] - logits[:, 1]).cpu().numpy()
            avg_loss += loss.item()
            all_scores.extend(scores)
            all_labels.extend(label.cpu().numpy())

    avg_loss = avg_loss / len(dataloader)
    all_scores = torch.tensor(all_scores).numpy()
    all_labels = torch.tensor(all_labels).numpy()

    bonafide_scores = all_scores[all_labels == 0]
    spoof_scores = all_scores[all_labels == 1]

    eer_ratio, threshold = compute_eer(bonafide_scores, spoof_scores)
    val_eer = eer_ratio * 100.0

    return avg_loss, val_eer


def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
    )

    train_dataset = ASVspoofDataset(
        cfg["data"]["train_protocol"],
        cfg["data"]["train_dir"],
        max_frames=cfg["data"]["max_frames"],
    )
    eval_dataset = ASVspoofDataset(
        cfg["data"]["eval_protocol"],
        cfg["data"]["eval_dir"],
        max_frames=cfg["data"]["max_frames"],
        is_eval=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    model = LCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    best_eer = float("inf")
    best_loss = float("inf")
    save_path = cfg["training"]["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        for epoch in range(cfg["training"]["epochs"]):
            print(f"Epoch {epoch + 1}/{cfg['training']['epochs']}")
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device
            )
            val_loss, val_eer = evaluate(model, eval_loader, criterion, device)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Train Loss: {train_loss:.4f} | Eval Loss: {val_loss:.4f} | "
                f"Eval EER: {val_eer:.4f}% | LR: {current_lr:.6f}"
            )

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "eval_eer": val_eer,
                    "lr": current_lr,
                }
            )

            if (val_eer < best_eer) or (val_eer == best_eer and val_loss < best_loss):
                best_eer = val_eer
                best_loss = val_loss
                torch.save(
                    {"epoch": epoch + 1, "model_state_dict": model.state_dict()},
                    save_path,
                )
                print(f"Best EER (saved): {best_eer:.4f}%")
                wandb.save(save_path)

        print(f"Best Eval EER: {best_eer:.4f}%")

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()