"""Evaluation classifier: SmallResNet training and test-accuracy measurement (Section 7.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import DATASET_CONFIGS
from common.data import (
    _unlock_test_split,
    get_split_tensors,
    load_raw,
    load_splits,
    to_model_range,
)
from common.nets import build_model
from common.utils import set_seed


def train_eval_classifier(
    dataset: str,
    save_path: Optional[Union[str, Path]] = None,
    raw_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    split_data: Optional[Dict[str, Any]] = None,
    data_root: Union[str, Path] = "./data",
    split_path: Optional[Union[str, Path]] = None,
    max_epochs: Optional[int] = None,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train or load the `SmallResNet` evaluation classifier on the training split (Section 7.7).

    Follows Section 6.2 Rule 9 & Section 7.7:
    - Trained from scratch on the training split ONLY, with training-only augmentation
      (random crop with padding 4 and horizontal flip on CIFAR-10).
    - Early-stopped on validation accuracy.
    - Evaluated on the test split ONCE (`evaluate_classifier_test_once`) to report test accuracy.
    """
    dataset = dataset.lower()
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = build_model(dataset, "classifier", seed=seed).to(dev)

    if save_path is not None and Path(save_path).exists():
        ckpt = torch.load(save_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, ckpt.get("info", {})

    if raw_data is None:
        raw_data = load_raw(dataset, data_root=data_root)
    if split_data is None:
        split_data = load_splits(dataset, split_path=split_path)

    x_train, y_train = get_split_tensors(
        dataset, "train", raw_data=raw_data, split_data=split_data, device=dev
    )
    x_val, y_val = get_split_tensors(
        dataset, "val", raw_data=raw_data, split_data=split_data, device=dev
    )

    if max_epochs is None:
        max_epochs = 12 if dataset == "mnist" else 45

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )

    set_seed(seed)
    best_val_acc = -1.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0
    n_train = x_train.shape[0]

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, batch_size):
            idx = perm[start : min(start + batch_size, n_train)]
            bx = to_model_range(x_train[idx])
            by = y_train[idx]

            if dataset == "cifar10":
                # Random crop with padding 4 and random horizontal flip (Section 7.7)
                padded = F.pad(bx, (4, 4, 4, 4), mode="reflect")
                b_cur = bx.shape[0]
                top = torch.randint(0, 9, (b_cur,), device=dev)
                left = torch.randint(0, 9, (b_cur,), device=dev)
                crops = [
                    padded[i, :, top[i] : top[i] + 32, left[i] : left[i] + 32]
                    for i in range(b_cur)
                ]
                bx = torch.stack(crops, dim=0)
                flip_mask = torch.rand((b_cur, 1, 1, 1), device=dev) < 0.5
                bx = torch.where(flip_mask, torch.flip(bx, dims=[-1]), bx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation accuracy check
        val_acc = _compute_classifier_accuracy(model, x_val, y_val, dev)
        if val_acc > best_val_acc + 1e-5:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.to(dev).eval()

    test_acc = evaluate_classifier_test_once(
        model, dataset=dataset, raw_data=raw_data, split_data=split_data, device=dev
    )
    target_acc = 0.99 if dataset == "mnist" else 0.90
    info: Dict[str, Any] = {
        "dataset": dataset,
        "best_val_accuracy": round(best_val_acc, 5),
        "test_accuracy": round(test_acc, 5),
        "target_accuracy": target_acc,
        "meets_target": bool(test_acc >= target_acc),
    }

    if save_path is not None:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "info": info}, out_p)

    return model, info


@torch.no_grad()
def _compute_classifier_accuracy(
    model: nn.Module,
    images_uint8: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    model.eval()
    correct = 0
    total = images_uint8.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        bx = to_model_range(images_uint8[start:end].to(device))
        by = labels[start:end].to(device)
        preds = model(bx).argmax(dim=-1)
        correct += int((preds == by).sum().item())
    return correct / max(1, total)


def evaluate_classifier_test_once(
    model: nn.Module,
    dataset: str,
    raw_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    split_data: Optional[Dict[str, Any]] = None,
    data_root: Union[str, Path] = "./data",
    split_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """Evaluate the classifier on the test split once (Section 6.2 Rule 9 & Section 7.7)."""
    if device is None:
        dev = next(model.parameters()).device
    else:
        dev = torch.device(device)
    with _unlock_test_split("evaluate_classifier_test_once"):
        x_test, y_test = get_split_tensors(
            dataset,
            "test",
            raw_data=raw_data,
            split_data=split_data,
            data_root=data_root,
            split_path=split_path,
        )
    return _compute_classifier_accuracy(model, x_test, y_test, dev)
