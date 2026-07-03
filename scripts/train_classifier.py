from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from steel_defect.config import CLASS_NAMES, FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, WEIGHTS_DIR, ensure_output_dirs
from steel_defect.visualization.plots import save_confusion_matrix


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class MobileNetV2SE(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.features = base.features
        self.se = SEBlock(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "mobilenet_v2_se":
        return MobileNetV2SE(num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model: {name}")


def build_loaders(data_dir: Path, image_size: int, batch_size: int, workers: int):
    train_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    datasets_by_split = {
        "train": datasets.ImageFolder(data_dir / "train", transform=train_tfms),
        "val": datasets.ImageFolder(data_dir / "val", transform=eval_tfms),
        "test": datasets.ImageFolder(data_dir / "test", transform=eval_tfms),
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in datasets_by_split.items()
    }
    return loaders, datasets_by_split


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(1).detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    start = time.perf_counter()
    for images, target in loader:
        images = images.to(device)
        logits = model(images)
        preds.extend(logits.argmax(1).cpu().numpy().tolist())
        labels.extend(target.numpy().tolist())
    elapsed = time.perf_counter() - start
    return np.array(labels), np.array(preds), elapsed / max(len(loader.dataset), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet/MobileNet classifiers on NEU-DET.")
    parser.add_argument("--model", choices=["resnet50", "mobilenet_v2", "mobilenet_v2_se"], default="mobilenet_v2")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "classification",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders, datasets_by_split = build_loaders(args.data_dir, args.image_size, args.batch_size, args.workers)
    if datasets_by_split["train"].classes != CLASS_NAMES:
        print(f"ImageFolder classes: {datasets_by_split['train'].classes}")

    model = build_model(args.model, len(CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    weight_path = WEIGHTS_DIR / f"{args.model}_best.pt"
    torch.save({"model": args.model, "state_dict": best_state, "classes": CLASS_NAMES}, weight_path)

    rows = []
    for split in ["val", "test"]:
        y_true, y_pred, seconds_per_image = predict(model, loaders[split], device)
        report = classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        save_confusion_matrix(
            y_true,
            y_pred,
            CLASS_NAMES,
            FIGURES_DIR / f"{args.model}_{split}_confusion_matrix.png",
            f"{args.model} {split} confusion matrix",
        )
        rows.append(
            {
                "model": args.model,
                "split": split,
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
                "seconds_per_image": seconds_per_image,
            }
        )

    pd.DataFrame(history).to_csv(REPORTS_DIR / f"{args.model}_history.csv", index=False)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / f"{args.model}_metrics.csv", index=False)
    print(pd.DataFrame(rows))
    print(f"Saved best weights to {weight_path}")


if __name__ == "__main__":
    main()

