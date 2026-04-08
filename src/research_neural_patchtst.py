"""
Optional PatchTST-style neural research model for the XAUUSD cockpit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from pipeline_contract import ensure_parent_dir


HORIZONS = (5, 15, 60)


@dataclass
class SequenceBundle:
    X: np.ndarray
    y: dict[int, np.ndarray]
    time: pd.Series


def _lazy_import_torch() -> tuple[Any, Any, Any, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def build_sequence_bundle(frame: pd.DataFrame, feature_columns: list[str], lookback: int) -> SequenceBundle:
    features = frame[feature_columns].astype(float).to_numpy(dtype=np.float32)
    times = pd.to_datetime(frame["time"]).reset_index(drop=True)
    labels = {horizon: (frame[f"direction_{horizon}m"].astype(int) + 1).to_numpy(dtype=np.int64) for horizon in HORIZONS}

    if len(frame) < lookback:
        raise ValueError(f"PatchTST requires at least {lookback} rows, but only {len(frame)} are available.")

    windows = []
    target_map = {horizon: [] for horizon in HORIZONS}
    aligned_times = []
    for idx in range(lookback - 1, len(frame)):
        windows.append(features[idx - lookback + 1 : idx + 1])
        for horizon in HORIZONS:
            target_map[horizon].append(labels[horizon][idx])
        aligned_times.append(times.iloc[idx])

    return SequenceBundle(
        X=np.asarray(windows, dtype=np.float32),
        y={horizon: np.asarray(values, dtype=np.int64) for horizon, values in target_map.items()},
        time=pd.Series(aligned_times, dtype="datetime64[ns]"),
    )


def _align_proba(raw: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(raw), 3), dtype=np.float32)
    for idx, cls in enumerate(classes):
        aligned[:, int(cls)] = raw[:, idx]
    sums = aligned.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return aligned / sums


def _multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    one_hot = np.eye(3, dtype=np.float32)[y_true]
    return float(np.mean(np.sum((one_hot - proba) ** 2, axis=1)))


def build_splits(n_rows: int, config: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = TimeSeriesSplit(
        n_splits=int(config["walk_forward"]["n_splits"]),
        gap=int(config["walk_forward"]["gap"]),
    )
    min_train_rows = int(config["walk_forward"]["min_train_rows"])
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in splitter.split(np.arange(n_rows)):
        if len(train_idx) < min_train_rows:
            continue
        splits.append((train_idx, test_idx))
    if not splits:
        raise ValueError("Not enough rows to build neural walk-forward splits.")
    return splits


def _device_name(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _count_model_parameters(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def _build_model(input_features: int, config: dict[str, Any]) -> Any:
    torch, nn, _, _ = _lazy_import_torch()
    neural_cfg = config["neural"]
    patch_len = int(neural_cfg["patch_len"])
    stride = int(neural_cfg["stride"])
    d_model = int(neural_cfg["d_model"])
    n_heads = int(neural_cfg["n_heads"])
    n_layers = int(neural_cfg["n_layers"])
    dropout = float(neural_cfg["dropout"])

    class PatchTSTResearchNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_len = patch_len
            self.stride = stride
            patch_dim = patch_len * input_features
            self.input_proj = nn.Linear(patch_dim, d_model)
            self.pos_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleDict({str(horizon): nn.Linear(d_model, 3) for horizon in HORIZONS})

        def patchify(self, x: Any) -> Any:
            if x.shape[1] < self.patch_len:
                pad = self.patch_len - x.shape[1]
                x = torch.nn.functional.pad(x, (0, 0, pad, 0))
            patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            if patches.ndim != 4:
                raise RuntimeError("Unexpected patch tensor rank.")
            return patches.contiguous().view(x.shape[0], patches.shape[1], -1)

        def forward(self, x: Any) -> dict[int, Any]:
            patches = self.patchify(x)
            encoded = self.input_proj(patches)
            encoded = encoded + self.pos_embedding[:, : encoded.shape[1], :]
            encoded = self.encoder(encoded)
            pooled = self.dropout(self.norm(encoded.mean(dim=1)))
            return {horizon: self.heads[str(horizon)](pooled) for horizon in HORIZONS}

    return PatchTSTResearchNet()


def _fit_one_fold(
    X_train: np.ndarray,
    y_train: dict[int, np.ndarray],
    X_valid: np.ndarray,
    config: dict[str, Any],
) -> tuple[Any, dict[int, np.ndarray], dict[str, Any]]:
    torch, nn, DataLoader, TensorDataset = _lazy_import_torch()
    device = torch.device(_device_name(torch))
    neural_cfg = config["neural"]

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0

    X_train_norm = ((X_train - mean) / std).astype(np.float32)
    X_valid_norm = ((X_valid - mean) / std).astype(np.float32)

    model = _build_model(X_train.shape[2], config).to(device)
    total_parameters, trainable_parameters = _count_model_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(neural_cfg["learning_rate"]),
        weight_decay=float(neural_cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train[5], dtype=torch.long),
        torch.tensor(y_train[15], dtype=torch.long),
        torch.tensor(y_train[60], dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=int(neural_cfg["batch_size"]), shuffle=False)

    epochs = int(neural_cfg["epochs"])
    loss_history: list[float] = []
    model.train()
    for _ in range(epochs):
        running = 0.0
        batches = 0
        for batch in loader:
            x_batch, y5_batch, y15_batch, y60_batch = [item.to(device) for item in batch]
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = (
                criterion(outputs[5], y5_batch)
                + criterion(outputs[15], y15_batch)
                + criterion(outputs[60], y60_batch)
            ) / 3.0
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu().item())
            batches += 1
        loss_history.append(running / max(batches, 1))

    model.eval()
    with torch.no_grad():
        valid_tensor = torch.tensor(X_valid_norm, dtype=torch.float32, device=device)
        logits = model(valid_tensor)
        probabilities = {
            horizon: torch.softmax(logits[horizon], dim=1).detach().cpu().numpy().astype(np.float32)
            for horizon in HORIZONS
        }

    state = {
        "model_state": model.state_dict(),
        "normalization_mean": mean.astype(np.float32),
        "normalization_std": std.astype(np.float32),
        "feature_count": X_train.shape[2],
        "config": config["neural"],
        "parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
    }
    return state, probabilities, {"training_loss": loss_history[-1] if loss_history else None, "device": str(device)}


def train_patchtst_research_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    torch, _, _, _ = _lazy_import_torch()
    lookback = int(config["neural"]["lookback"])
    bundle = build_sequence_bundle(frame, feature_columns, lookback=lookback)
    splits = build_splits(len(bundle.X), config)

    oof = {horizon: np.full((len(bundle.X), 3), np.nan, dtype=np.float32) for horizon in HORIZONS}
    fold_metrics: dict[int, list[dict[str, Any]]] = {horizon: [] for horizon in HORIZONS}
    last_state: dict[str, Any] | None = None
    training_notes: list[str] = []

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        state, probabilities, fold_note = _fit_one_fold(
            X_train=bundle.X[train_idx],
            y_train={horizon: bundle.y[horizon][train_idx] for horizon in HORIZONS},
            X_valid=bundle.X[test_idx],
            config=config,
        )
        last_state = state
        training_notes.append(
            f"Fold {fold_number}: device={fold_note['device']}, final_loss={fold_note['training_loss']:.4f}"
            if fold_note["training_loss"] is not None
            else f"Fold {fold_number}: device={fold_note['device']}"
        )

        for horizon in HORIZONS:
            oof[horizon][test_idx] = probabilities[horizon]
            y_true = bundle.y[horizon][test_idx]
            predicted = probabilities[horizon].argmax(axis=1)
            fold_metrics[horizon].append(
                {
                    "fold": fold_number,
                    "samples": int(len(test_idx)),
                    "accuracy": float(accuracy_score(y_true, predicted)),
                    "log_loss": float(log_loss(y_true, probabilities[horizon], labels=[0, 1, 2])),
                    "brier": _multiclass_brier(y_true, probabilities[horizon]),
                }
            )

    if last_state is None:
        raise RuntimeError("PatchTST training did not produce any model state.")

    valid_mask = ~np.isnan(oof[15]).any(axis=1)
    prediction_frame = pd.DataFrame({"time": bundle.time.astype(str)})
    report: dict[str, Any] = {
        "status": "ok",
        "model_family": "PatchTST-inspired multi-horizon classifier",
        "lookback": lookback,
        "prediction_coverage": float(valid_mask.mean()),
        "parameter_count": int(last_state.get("parameter_count", 0)),
        "trainable_parameter_count": int(last_state.get("trainable_parameter_count", 0)),
        "notes": training_notes,
        "horizons": {},
    }

    for horizon in HORIZONS:
        prediction_frame[f"prob_short_{horizon}m"] = oof[horizon][:, 0]
        prediction_frame[f"prob_hold_{horizon}m"] = oof[horizon][:, 1]
        prediction_frame[f"prob_long_{horizon}m"] = oof[horizon][:, 2]
        horizon_mask = ~np.isnan(oof[horizon]).any(axis=1)
        y_true = bundle.y[horizon][horizon_mask]
        proba = oof[horizon][horizon_mask]
        predicted = proba.argmax(axis=1)
        report["horizons"][str(horizon)] = {
            "samples": int(horizon_mask.sum()),
            "accuracy": float(accuracy_score(y_true, predicted)),
            "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
            "brier": _multiclass_brier(y_true, proba),
            "fold_metrics": fold_metrics[horizon],
        }

    output_root = Path(output_dir)
    predictions_path = ensure_parent_dir(output_root / "predictions.csv")
    report_path = ensure_parent_dir(output_root / "report.json")
    model_path = ensure_parent_dir(output_root / "model.pt")

    prediction_frame.to_csv(predictions_path, index=False)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    save_payload = {
        "state_dict": last_state["model_state"],
        "normalization_mean": torch.tensor(last_state["normalization_mean"]),
        "normalization_std": torch.tensor(last_state["normalization_std"]),
        "feature_columns": feature_columns,
        "config": config["neural"],
        "horizons": list(HORIZONS),
    }
    torch.save(save_payload, model_path)

    report["predictions_path"] = str(predictions_path)
    report["report_path"] = str(report_path)
    report["model_path"] = str(model_path)
    return report
