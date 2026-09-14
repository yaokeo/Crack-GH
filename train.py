from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from crackseg.config import load_config, resolve_project_path, seed_everything
from crackseg.data import build_dataloaders
from crackseg.losses import build_loss
from crackseg.metrics import MetricAverage, binary_segmentation_metrics
from crackseg.models import build_model


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def build_optimizer(model: nn.Module, config: dict) -> AdamW:
    base_lr = float(config["lr"])
    offset_lr_scale = float(config.get("offset_lr_scale", 0.1))
    offset_parameters: list[nn.Parameter] = []
    base_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "offset_head" in name or "scope_gate" in name:
            offset_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    parameter_groups: list[dict[str, Any]] = [
        {"params": base_parameters, "group_name": "base"}
    ]
    if offset_parameters:
        parameter_groups.append(
            {
                "params": offset_parameters,
                "lr": base_lr * offset_lr_scale,
                "group_name": "akconv_offset",
            }
        )
    return AdamW(
        parameter_groups,
        lr=base_lr,
        weight_decay=float(config["weight_decay"]),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    sample_count = 0
    diagnostic_sums: dict[str, float] = {}
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        ):
            output = model(images)
            loss = criterion(output, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        if isinstance(output, dict):
            for key in (
                "akconv_offset_mean",
                "akconv_offset_max",
                "akconv_scope_mean",
            ):
                value = output.get(key)
                if isinstance(value, Tensor):
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + (
                        value.detach().mean().item() * batch_size
                    )
    diagnostics = {
        key: value / max(1, sample_count)
        for key, value in diagnostic_sums.items()
    }
    return total_loss / max(1, sample_count), diagnostics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    meter = MetricAverage()
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        output = model(images)
        loss = criterion(output, masks)
        logits = output["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("Model logits must be a Tensor")
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        values = binary_segmentation_metrics(logits, masks, threshold=threshold)
        values["loss"] = loss.item()
        meter.update(values, images.shape[0])
    return meter.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CrackGH-UNet")
    parser.add_argument("--config", default="configs/base_crack.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 3407)))
    device = select_device(args.device)

    data_config = config["data"]
    dataset_root = (
        resolve_project_path(config, data_config["dataset_root"])
        if data_config.get("dataset_root")
        else None
    )
    train_split = (
        resolve_project_path(config, data_config["train_split"])
        if data_config.get("train_split")
        else None
    )
    val_split = (
        resolve_project_path(config, data_config["val_split"])
        if data_config.get("val_split")
        else None
    )
    train_loader, val_loader = build_dataloaders(
        resolve_project_path(config, data_config["train_images"]),
        resolve_project_path(config, data_config["train_masks"]),
        resolve_project_path(config, data_config["val_images"]),
        resolve_project_path(config, data_config["val_masks"]),
        data_config,
        train_split=train_split,
        val_split=val_split,
        dataset_root=dataset_root,
    )
    model = build_model(config["model"]).to(device)
    criterion = build_loss(config["loss"])
    train_config = config["train"]
    optimizer = build_optimizer(model, train_config)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    epochs = int(train_config["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    has_akconv = any(
        group.get("group_name") == "akconv_offset" for group in optimizer.param_groups
    )
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output_dir = resolve_project_path(config, train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
            },
            ensure_ascii=False,
        )
    )

    for epoch in range(1, epochs + 1):
        warmup_epochs = max(
            1, int(train_config.get("akconv_offset_warmup_epochs", 10))
        )
        radius_scale = min(1.0, epoch / warmup_epochs)
        radius_setter = getattr(unwrap_model(model), "set_akconv_radius_scale", None)
        if has_akconv and radius_setter is not None:
            radius_setter(radius_scale)
        train_loss, train_diagnostics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
        )
        metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            threshold=float(train_config.get("threshold", 0.5)),
        )
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, **metrics}
        if has_akconv:
            record["akconv_radius_scale"] = radius_scale
            record.update(train_diagnostics)
        print(json.dumps(record, ensure_ascii=False))

        checkpoint = {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        }
        torch.save(checkpoint, output_dir / "last.pth")
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(checkpoint, output_dir / "best.pth")


if __name__ == "__main__":
    main()
