from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import warnings
import sys
import time

import numpy as np
import torch
from torch import nn

import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")

try:
    from .dataloader import (
        FEATURE_COLUMNS,
        YEAR_COLUMN,
        EmbeddingSettings,
        MatchSequenceDataset,
        SequenceBundle,
        SiameseMatchDataset,
        TabularMatchDataset,
        build_sequence_bundle,
        load_match_dataframe,
        split_match_dataframe,
    )
except ImportError:  # pragma: no cover - fallback when run as a standalone script
    from dataloader import (
        FEATURE_COLUMNS,
        YEAR_COLUMN,
        EmbeddingSettings,
        MatchSequenceDataset,
        SequenceBundle,
        SiameseMatchDataset,
        TabularMatchDataset,
        build_sequence_bundle,
        load_match_dataframe,
        split_match_dataframe,
    )

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        # Recebe uma lista de pesos e converte para tensor
        if alpha is None:
            self.alpha = None
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
            
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits brutos da rede [batch_size, num_classes]
        # targets: labels verdadeiros (índices ou One-Hot)

        # 1. Tratar alvos One-Hot (necessário para buscar o peso correto de cada linha)
        if targets.ndim > 1 and targets.shape[1] == inputs.shape[1]:
            targets_indices = torch.argmax(targets, dim=1)
        else:
            targets_indices = targets

        # 2. Cross Entropy pura (SEM pesos) para extrair p_t perfeitamente
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 3. Cálculo base da Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 4. Aplicação do vetor Alpha (Pesos das Classes)
        if self.alpha is not None:
            # Garante que o tensor de pesos esteja na mesma memória (CPU ou GPU) que os dados
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Coleta o peso específico para cada exemplo do lote
            alpha_t = self.alpha[targets_indices]
            
            # Multiplica a perda pelo peso da classe
            focal_loss = alpha_t * focal_loss

        # 5. Redução final
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        
        return focal_loss


@dataclass
class EpochMetrics:
    loss: float = 0.0
    classification_loss: float = 0.0
    regression_loss: float = 0.0
    accuracy: float = 0.0
    f1_score: float = 0.0
    brier_score: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0

try:
    from .LSTM import LSTMBackbone, LSTMClassifier, LSTMRegressor
except ImportError:
    from LSTM import LSTMBackbone, LSTMClassifier, LSTMRegressor

try:
    from .TabularMLP import TabularMLPBackbone, TabularClassifier, TabularRegressor
except ImportError:
    from TabularMLP import TabularMLPBackbone, TabularClassifier, TabularRegressor

try:
    from .SiameseLSTM import SiameseLSTMBackbone, SiameseClassifier, SiameseRegressor
except ImportError:
    from SiameseLSTM import SiameseLSTMBackbone, SiameseClassifier, SiameseRegressor

def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(
    arch: str,
    model_type: str,
    bundle: SequenceBundle,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    runtime_device: torch.device,
) -> nn.Module:
    if arch == "legacy":
        backbone = LSTMBackbone(
            numerical_input_size=len(bundle.numerical_feature_columns),
            embedding_settings=bundle.embedding_settings,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        if model_type == "classifier":
            model = LSTMClassifier(backbone=backbone, hidden_size=hidden_size, dropout=dropout)
        else:
            model = LSTMRegressor(backbone=backbone, hidden_size=hidden_size, dropout=dropout)
    elif arch == "mlp":
        backbone = TabularMLPBackbone(
            numerical_input_size=len(bundle.numerical_feature_columns),
            embedding_settings=bundle.embedding_settings,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        if model_type == "classifier":
            model = TabularClassifier(backbone=backbone, hidden_size=hidden_size, dropout=dropout)
        else:
            model = TabularRegressor(backbone=backbone, hidden_size=hidden_size, dropout=dropout)
    elif arch == "siamese":
        backbone = SiameseLSTMBackbone(
            numerical_input_size=len(bundle.numerical_feature_columns),
            embedding_settings=bundle.embedding_settings,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        if model_type == "classifier":
            model = SiameseClassifier(backbone=backbone, hidden_size=hidden_size, dropout=dropout)
        else:
            model = SiameseRegressor(backbone=backbone, hidden_size=hidden_size, dropout=dropout)
    else:
        raise ValueError(f"arch deve ser 'legacy', 'mlp' ou 'siamese', não '{arch}'")

    return model.to(runtime_device)


def _universal_forward_pass(arch: str, model: nn.Module, batch: tuple, device: torch.device):
    """
    Desempacota o batch de acordo com a arquitetura e realiza o forward pass.
    Retorna: (outputs, class_targets, regression_targets)
    """
    if arch == "siamese":
        inputs, c_targs, r_targs = batch
        h_num, h_cat, a_num, a_cat, m_num, m_cat = inputs
        outputs = model(
            h_num.to(device), h_cat.to(device), a_num.to(device), a_cat.to(device),
            m_num.to(device), m_cat.to(device)
        )
        return outputs, c_targs.to(device), r_targs.to(device)

    num, cat, c_targs, r_targs = batch
    outputs = model(num.to(device), cat.to(device))
    return outputs, c_targs.to(device), r_targs.to(device)


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, remaining_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def _render_progress(epoch: int, epochs: int, step: int, total_steps: int, loss: float, accuracy: float, elapsed: float) -> None:
    if total_steps <= 0:
        return

    width = 28
    filled = int(width * step / total_steps)
    bar = "=" * filled + "." * (width - filled)
    eta = _format_eta((elapsed / step) * (total_steps - step)) if step > 0 else "--:--"
    message = (
        f"\rEpoch {epoch}/{epochs} [{bar}] {step:>4}/{total_steps:<4} "
        f"loss={loss:.4f} acc={accuracy:.4f} eta={eta}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def _collect_validation_predictions(
    model: nn.Module,
    arch: str,
    model_type: str,
    validation_dataframe,
    validation_loader: torch.utils.data.DataLoader | None,
    numerical_feature_columns: list[str],
    categorical_feature_columns: list[str],
    sequence_length: int,
    device: torch.device,
) -> Any:
    predicted_frame = validation_dataframe.copy()
    if model_type == "classifier":
        predicted_frame["pred_resultado_empate"] = np.nan
        predicted_frame["pred_resultado_vitoria_mandante"] = np.nan
        predicted_frame["pred_resultado_vitoria_visitante"] = np.nan
    else:
        predicted_frame["pred_gols_mandante"] = np.nan
        predicted_frame["pred_gols_visitante"] = np.nan

    model.eval()
    if arch == "legacy":
        numerical_batches: list[torch.Tensor] = []
        categorical_batches: list[torch.Tensor] = []
        row_positions: list[int] = []

        ordered = validation_dataframe.sort_values([YEAR_COLUMN])
        for _, season_frame in ordered.groupby(YEAR_COLUMN, sort=True):
            season_frame = season_frame.reset_index()
            if len(season_frame) < sequence_length:
                continue

            numerical_features = torch.tensor(season_frame[numerical_feature_columns].to_numpy(), dtype=torch.float32)
            categorical_features = torch.tensor(season_frame[categorical_feature_columns].to_numpy(), dtype=torch.int64)
            for end_index in range(sequence_length - 1, len(season_frame)):
                start_index = end_index - sequence_length + 1
                numerical_batches.append(numerical_features[start_index : end_index + 1])
                categorical_batches.append(categorical_features[start_index : end_index + 1])
                row_positions.append(int(season_frame.loc[end_index, "index"]))

        if not numerical_batches:
            return predicted_frame

        numerical_batch_tensor = torch.stack(numerical_batches).to(device)
        categorical_batch_tensor = torch.stack(categorical_batches).to(device)
        with torch.no_grad():
            outputs = model(numerical_batch_tensor, categorical_batch_tensor)
            if model_type == "classifier":
                predicted_classes = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                predicted_goals = np.full((len(predicted_classes), 2), np.nan)
            else:
                predicted_goals = torch.exp(outputs).cpu().numpy()
                predicted_classes = np.full((len(predicted_goals), 3), np.nan)

        for row_position, predicted_class, (predicted_home_goals, predicted_away_goals) in zip(row_positions, predicted_classes, predicted_goals):
            if model_type == "classifier":
                predicted_frame.loc[row_position, "pred_resultado_empate"] = predicted_class[0]
                predicted_frame.loc[row_position, "pred_resultado_vitoria_mandante"] = predicted_class[1]
                predicted_frame.loc[row_position, "pred_resultado_vitoria_visitante"] = predicted_class[2]
            else:
                predicted_frame.loc[row_position, "pred_gols_mandante"] = float(predicted_home_goals)
                predicted_frame.loc[row_position, "pred_gols_visitante"] = float(predicted_away_goals)

        return predicted_frame

    predicted_frame = validation_dataframe.reset_index(drop=True).copy()
    if model_type == "classifier":
        predicted_frame["pred_resultado_empate"] = np.nan
        predicted_frame["pred_resultado_vitoria_mandante"] = np.nan
        predicted_frame["pred_resultado_vitoria_visitante"] = np.nan
    else:
        predicted_frame["pred_gols_mandante"] = np.nan
        predicted_frame["pred_gols_visitante"] = np.nan

    if arch == "legacy":
        raise ValueError("legacy validation predictions are handled earlier in this function")

    if validation_loader is None:
        raise ValueError("validation_loader is required for non-legacy validation predictions")

    class_predictions: list[np.ndarray] = []
    goal_predictions: list[np.ndarray] = []

    with torch.no_grad():
        for batch in validation_loader:
            outputs, _, _ = _universal_forward_pass(arch, model, batch, device)
            if model_type == "classifier":
                class_predictions.append(torch.softmax(outputs, dim=1).cpu().numpy())
            else:
                goal_predictions.append(torch.exp(outputs).cpu().numpy())

    if model_type == "classifier":
        predicted_classes = np.concatenate(class_predictions, axis=0) if class_predictions else np.empty((0, 3))
        for row_position, predicted_class in enumerate(predicted_classes):
            predicted_frame.loc[row_position, "pred_resultado_empate"] = predicted_class[0]
            predicted_frame.loc[row_position, "pred_resultado_vitoria_mandante"] = predicted_class[1]
            predicted_frame.loc[row_position, "pred_resultado_vitoria_visitante"] = predicted_class[2]
    else:
        predicted_goals = np.concatenate(goal_predictions, axis=0) if goal_predictions else np.empty((0, 2))
        for row_position, (predicted_home_goals, predicted_away_goals) in enumerate(predicted_goals):
            predicted_frame.loc[row_position, "pred_gols_mandante"] = float(predicted_home_goals)
            predicted_frame.loc[row_position, "pred_gols_visitante"] = float(predicted_away_goals)

    return predicted_frame


def _compute_epoch_metrics(
    model: nn.Module,
    arch: str,
    model_type: str,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_regression_loss = 0.0
    total_classification_loss = 0.0
    total_squared_prob_error = 0.0
    total_correct = 0
    total_samples = 0
    total_abs_error = 0.0
    total_squared_error = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            outputs, class_targets, regression_targets = _universal_forward_pass(arch, model, batch, device)
            batch_size = class_targets.size(0)

            if model_type == "classifier":
                loss = criterion(outputs, class_targets)
                total_classification_loss += float(loss.item()) * batch_size
                
                probabilities = F.softmax(outputs, dim=1)
                brier_error = ((probabilities - class_targets) ** 2).sum()
                total_squared_prob_error += float(brier_error.item())
                
                predictions = torch.argmax(outputs, 1)
                target_classes = torch.argmax(class_targets, 1)
                #print(f"Predictions: {outputs.cpu().numpy()}, Targets: {class_targets.cpu().numpy()}")
                total_correct += int((predictions == target_classes).sum().item())
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target_classes.cpu().numpy())
            else:
                loss = criterion(outputs, regression_targets)
                total_regression_loss += float(loss.item()) * batch_size
                predicted_goals = torch.exp(outputs)
                prediction_error = predicted_goals - regression_targets
                #print(f"Predicted goals: {predicted_goals.cpu().numpy()}, Target goals: {regression_targets.cpu().numpy()}")
                total_abs_error += float(prediction_error.abs().sum().item())
                total_squared_error += float((prediction_error ** 2).sum().item())

            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return EpochMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    regression_elements = total_samples * 2 if model_type == "regressor" else 1
    
    current_f1_score = 0.0
    if model_type == "classifier" and total_samples > 0:
        current_f1_score = float(f1_score(all_targets, all_predictions, average="macro", zero_division=0))

    return EpochMetrics(
        loss=total_loss / total_samples,
        classification_loss=total_classification_loss / total_samples,
        regression_loss=total_regression_loss / total_samples,
        accuracy=total_correct / total_samples if model_type == "classifier" else 0.0,
        f1_score=current_f1_score,
        brier_score=total_squared_prob_error / total_samples if model_type == "classifier" else 0.0,
        mae=total_abs_error / regression_elements if model_type == "regressor" else 0.0,
        rmse=float(np.sqrt(total_squared_error / regression_elements)) if model_type == "regressor" else 0.0,
    )


def train_model(
    csv_path: str | Path = Path(__file__).resolve().parents[1] / "data" / "dataset_scaled.csv",
    arch: str = "mlp",
    model_type: str = "classifier",
    sequence_length: int = 10,
    batch_size: int = 64,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    epochs: int = 30,
    early_stopping_patience: int = 10,
    num_workers: int = 0,
    device: str | None = None,
    save_path: str | Path | None = None,
    best_model_path: str | Path | None = None,
    validation_predictions_path: str | Path = Path(__file__).resolve().parents[0] / "respostas_lstm.csv",
) -> dict[str, Any]:
    
    if arch not in ("legacy", "mlp", "siamese"):
        raise ValueError(f"arch deve ser 'legacy', 'mlp' ou 'siamese', não '{arch}'")
    if model_type not in ("classifier", "regressor"):
        raise ValueError(f"model_type deve ser 'classifier' ou 'regressor', não '{model_type}'")

    checkpoint_path = best_model_path or save_path
    bundle: SequenceBundle = build_sequence_bundle(
        csv_path=csv_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        num_workers=num_workers,
        arch=arch,
    )
    runtime_device = get_device(device)

    pesos_classes = [1.01, 1.03, 1.02]
    model = _build_model(
        arch=arch,
        model_type=model_type,
        bundle=bundle,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        runtime_device=runtime_device,
    )
    if model_type == "classifier":
        criterion = FocalLoss(alpha=pesos_classes, gamma=2.0, reduction='mean')
    else:
        criterion = nn.PoissonNLLLoss(log_input=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, cooldown=1)
    use_amp = runtime_device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_test_performance = float("inf")
    prev_train_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_test_metrics: EpochMetrics | None = None
    patience_counter = 0

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        running_brier_score = 0.0
        running_correct = 0
        running_samples = 0
        all_train_predictions = []
        all_train_targets = []
        epoch_start = time.time()
        total_steps = len(bundle.train_loader)

        for step, batch in enumerate(bundle.train_loader, start=1):
            optimizer.zero_grad(set_to_none=False)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs, class_targets, regression_targets = _universal_forward_pass(arch, model, batch, runtime_device)
                if model_type == "classifier":
                    loss = criterion(outputs, class_targets)
                    classification_loss = loss
                    regression_loss = torch.tensor(0.0)
                else:
                    loss = criterion(outputs, regression_targets)
                    regression_loss = loss
                    classification_loss = torch.tensor(0.0)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size_actual = class_targets.size(0)
            running_samples += batch_size_actual
            running_loss += float(loss.item()) * batch_size_actual
            running_classification_loss += float(classification_loss.item()) * batch_size_actual
            running_regression_loss += float(regression_loss.item()) * batch_size_actual

            if model_type == "classifier":
                with torch.no_grad():
                    _, predicted_classes = torch.max(outputs, 1)
                    _, target_classes = torch.max(class_targets, 1)
                    running_correct += int((predicted_classes == target_classes).sum().item())
                    all_train_predictions.extend(predicted_classes.cpu().numpy())
                    all_train_targets.extend(target_classes.cpu().numpy())

                    probabilities = F.softmax(outputs, dim=1)
                    brier_error = ((probabilities - class_targets) ** 2).sum()
                    running_brier_score += float(brier_error.item())

            _render_progress(
                epoch=epoch,
                epochs=epochs,
                step=step,
                total_steps=total_steps,
                loss=running_loss / running_samples if running_samples else 0.0,
                accuracy=running_correct / running_samples if running_samples and model_type == "classifier" else 0.0,
                elapsed=time.time() - epoch_start,
            )

        sys.stdout.write("\n")
        sys.stdout.flush()

        train_f1_score = 0.0
        if model_type == "classifier" and running_samples > 0:
            train_f1_score = float(f1_score(all_train_targets, all_train_predictions, average="macro", zero_division=0))

        train_metrics = EpochMetrics(
            loss=running_loss / running_samples if running_samples else 0.0,
            classification_loss=running_classification_loss / running_samples if running_samples else 0.0,
            regression_loss=running_regression_loss / running_samples if running_samples else 0.0,
            accuracy=running_correct / running_samples if running_samples and model_type == "classifier" else 0.0,
            f1_score=train_f1_score,
            brier_score=running_brier_score / running_samples if running_samples and model_type == "classifier" else 0.0,
            mae=0.0,
            rmse=0.0,
        )

        validation_metrics = _compute_epoch_metrics(
            model=model,
            arch=arch,
            model_type=model_type,
            loader=bundle.validation_loader,
            device=runtime_device,
            criterion=criterion,
        )

        test_metrics = _compute_epoch_metrics(
            model=model,
            arch=arch,
            model_type=model_type,
            loader=bundle.test_loader,
            device=runtime_device,
            criterion=criterion,
        )

        if model_type == "classifier":
            summary_line = (
                f"Epoch {epoch}/{epochs} final | "
                f"train_class_loss={train_metrics.classification_loss:.6f} train_acc={train_metrics.accuracy:.6f} train_f1={train_metrics.f1_score:.6f} train_brier={train_metrics.brier_score:.6f} | "
                f"test_acc={test_metrics.accuracy:.6f} test_f1={test_metrics.f1_score:.6f} test_brier={test_metrics.brier_score:.6f}"
            )
        else:
            summary_line = (
                f"Epoch {epoch}/{epochs} final | "
                f"train_reg_loss={train_metrics.regression_loss:.6f} train_mae={train_metrics.mae:.6f} train_rmse={train_metrics.rmse:.6f} | "
                f"test_mae={test_metrics.mae:.6f} test_rmse={test_metrics.rmse:.6f}"
            )
        sys.stdout.write(summary_line + "\n")
        sys.stdout.flush()

        scheduler.step(test_metrics.loss)

        history.append(
            {
                "epoch": epoch,
                "train": asdict(train_metrics),
                "test": asdict(test_metrics),
                "validation": asdict(validation_metrics),
            }
        )

        if (test_metrics.loss < best_test_performance and model_type == "classifier") or (model_type == "regressor" and test_metrics.mae < best_test_performance):
            best_test_performance = test_metrics.loss if model_type == "classifier" else test_metrics.mae
            best_state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            best_epoch = epoch
            best_test_metrics = test_metrics
            patience_counter = 0
            if checkpoint_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_state_dict,
                        "numerical_feature_columns": bundle.numerical_feature_columns,
                        "categorical_feature_columns": bundle.categorical_feature_columns,
                        "embedding_settings": bundle.embedding_settings,
                        "sequence_length": sequence_length,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "best_epoch": best_epoch,
                        "best_test_metrics": asdict(best_test_metrics),
                        "model_type": model_type,
                    },
                    checkpoint_path,
                )
        else:
            # Overfitting: loss de treino continuou caindo, mas a de validação não
            if train_metrics.loss < test_metrics.loss:
                patience_counter += 1

        prev_train_loss = train_metrics.loss if train_metrics.loss < prev_train_loss else prev_train_loss

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            break

    if checkpoint_path is not None and checkpoint_path.exists():
        with torch.serialization.safe_globals([EmbeddingSettings]):
            best_checkpoint = torch.load(checkpoint_path, map_location=runtime_device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
    elif best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    full_dataframe = load_match_dataframe(csv_path)
    validation_dataframe = split_match_dataframe(full_dataframe).validation
    test_dataframe = split_match_dataframe(full_dataframe).test
    
    validation_predictions_path = Path(validation_predictions_path)
    validation_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    validation_predictions = _collect_validation_predictions(
        model=model,
        arch=arch,
        model_type=model_type,
        validation_dataframe=validation_dataframe,
        validation_loader=bundle.validation_loader,
        numerical_feature_columns=bundle.numerical_feature_columns,
        categorical_feature_columns=bundle.categorical_feature_columns,
        sequence_length=sequence_length,
        device=runtime_device,
    )
    validation_predictions.to_csv(validation_predictions_path, index=False)

    test_metrics = _compute_epoch_metrics(
        model=model,
        arch=arch,
        model_type=model_type,
        loader=bundle.test_loader,
        device=runtime_device,
        criterion=criterion,
    )

    validation_metrics = _compute_epoch_metrics(
        model=model,
        arch=arch,
        model_type=model_type,
        loader=bundle.validation_loader,
        device=runtime_device,
        criterion=criterion,
    )

    result: dict[str, Any] = {
        "model": model,
        "history": history,
        "test_metrics": asdict(test_metrics),
        "validation_metrics": asdict(validation_metrics),
        "best_epoch": best_epoch,
        "best_test_metrics": asdict(best_test_metrics) if best_test_metrics is not None else None,
        "validation_predictions_path": str(validation_predictions_path),
        "bundle": bundle,
        "device": str(runtime_device),
    }

    if checkpoint_path is not None and checkpoint_path.exists():
        with torch.serialization.safe_globals([EmbeddingSettings]):
            checkpoint = torch.load(checkpoint_path, map_location=runtime_device, weights_only=False)
        checkpoint.update(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_test_metrics": asdict(best_test_metrics) if best_test_metrics is not None else None,
                "validation_predictions_path": str(validation_predictions_path),
                "test_metrics": asdict(test_metrics),
            }
        )
        torch.save(checkpoint, checkpoint_path)

    return result


def plot_metrics(output: dict[str, Any], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    history = output["history"]
    test_metrics = output["test_metrics"]
    
    epochs = [h["epoch"] for h in history]
    
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["validation"]["loss"] for h in history]
    test_loss = [h['test']['loss'] for h in history]
    
    train_acc = [h["train"]["accuracy"] for h in history]
    val_acc = [h["validation"]["accuracy"] for h in history]
    test_acc = [h['test']['accuracy'] for h in history]
    
    train_f1 = [h["train"]["f1_score"] for h in history]
    val_f1 = [h["validation"]["f1_score"] for h in history]
    test_f1 = [h['test']['f1_score'] for h in history]
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Treino")
    plt.plot(epochs, test_loss, label="Teste")
    plt.plot(epochs, val_loss, label="Validação")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Evolução da Loss (Perda)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "loss_curve.png")
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label="Treino")
    plt.plot(epochs, test_acc, label="Teste")
    plt.plot(epochs, val_acc, label="Validação")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.title("Evolução da Acurácia")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "accuracy_curve.png")
    plt.close()
    
    # Plot F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_f1, label="Treino")
    plt.plot(epochs, test_f1, label="Teste")
    plt.plot(epochs, val_f1, label="Validação")
    plt.xlabel("Época")
    plt.ylabel("F1 Score")
    plt.title("Evolução do F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "f1_score_curve.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train football predictor models across multiple architectures.")
    parser.add_argument("--arch", choices=["legacy", "mlp", "siamese"], default="mlp")
    parser.add_argument("--model_type", choices=["classifier", "regressor"], default="classifier")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "dataset_preprocessed.csv"),
    )
    args = parser.parse_args()
    results_name = f"respostas_{args.arch}.csv"
    checkpoint_name = f"{args.arch}_{args.model_type}.pth"
    output = train_model(
        csv_path=Path(args.csv_path),
        arch=args.arch,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.lr,
        num_workers=args.workers,
        save_path=Path(__file__).resolve().parents[0] / "checkpoints" / checkpoint_name,
        best_model_path=Path(__file__).resolve().parents[0] / "checkpoints" / f"best_{checkpoint_name}",
        validation_predictions_path=Path(__file__).resolve().parents[0] / "results" / results_name,
        early_stopping_patience=args.early_stopping_patience,
    )


    print("Dispositivo:", output["device"])
    print("Métricas de teste:", output["test_metrics"])
    print("Métricas de validação:", output["validation_metrics"])
    print("Melhor época:", output["best_epoch"])   
    
    plots_dir = Path(__file__).resolve().parents[0] / "plots"
    plot_metrics(output, plots_dir)
    print(f"Gráficos salvos em: {plots_dir}")


if __name__ == "__main__":
    main()