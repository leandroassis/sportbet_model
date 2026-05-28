from __future__ import annotations

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
        SequenceBundle,
        build_sequence_bundle,
        load_match_dataframe,
        split_match_dataframe,
    )
except ImportError:  # pragma: no cover - fallback when run as a standalone script
    from dataloader import (
        FEATURE_COLUMNS,
        YEAR_COLUMN,
        EmbeddingSettings,
        SequenceBundle,
        build_sequence_bundle,
        load_match_dataframe,
        split_match_dataframe,
    )


@dataclass
class EpochMetrics:
    loss: float = 0.0
    classification_loss: float = 0.0
    regression_loss: float = 0.0
    accuracy: float = 0.0
    f1_score: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Calcula o peso de cada timestep da LSTM para formar um vetor de contexto focado,
    dando mais importância a rodadas chave na sequência histórica.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        # lstm_outputs shape: [batch_size, seq_len, hidden_size]
        attn_weights = self.attention(lstm_outputs) # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Multiplica os outputs pelos pesos e soma ao longo da dimensão temporal
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1) # [batch_size, hidden_size]
        return context_vector


class LSTMBackbone(nn.Module):
    def __init__(
        self,
        numerical_input_size: int,
        embedding_settings, # Tipo assumido: EmbeddingSettings
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings, dim)
            for name, (num_embeddings, dim) in embedding_settings.dimensions.items()
        })
        
        # Dropout extra para evitar overfitting precoce nos embeddings categóricos
        self.embedding_dropout = nn.Dropout(dropout * 0.5)
        
        lstm_input_size = numerical_input_size + sum(dim for _, dim in embedding_settings.dimensions.values())
        self.input_ln = nn.LayerNorm(lstm_input_size)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Módulo de atenção temporal instanciado
        self.attention = TemporalAttention(hidden_size)
        
        self.latent_dense1 = nn.Linear(hidden_size, hidden_size)
        self.latent_ln1 = nn.LayerNorm(hidden_size)
        self.latent_act1 = nn.ReLU()
        self.latent_drop1 = nn.Dropout(dropout)
        
        self.latent_dense2 = nn.Linear(hidden_size, hidden_size)
        self.latent_ln2 = nn.LayerNorm(hidden_size)
        self.latent_act2 = nn.ReLU()
        self.latent_drop2 = nn.Dropout(dropout)

    def forward(self, numerical_inputs: torch.Tensor, categorical_inputs: torch.Tensor) -> torch.Tensor:
        embeddings = [
            self.embedding_layers[name](categorical_inputs[..., i])
            for i, name in enumerate(self.embedding_layers)
        ]
        
        cat_embeddings = self.embedding_dropout(torch.cat(embeddings, dim=-1))
        combined_input = torch.cat([numerical_inputs, cat_embeddings], dim=-1)
        combined_input = self.input_ln(combined_input)
        
        outputs, _ = self.lstm(combined_input)
        
        # Substituição da captura estática pelo vetor de contexto com atenção
        context = self.attention(outputs)
        
        x = self.latent_dense1(context)
        x = self.latent_ln1(x)
        x = self.latent_act1(x)
        x = self.latent_drop1(x)
        
        residual = x
        x = self.latent_dense2(x)
        x = self.latent_ln2(x)
        x = x + residual
        x = self.latent_act2(x)
        x = self.latent_drop2(x)
        
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, backbone: LSTMBackbone, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), # LayerNorm adicionado para estabilidade
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 3)
        )

    def forward(self, numerical_inputs: torch.Tensor, categorical_inputs: torch.Tensor) -> torch.Tensor:
        backbone_output = self.backbone(numerical_inputs, categorical_inputs)
        return self.classification_head(backbone_output)


class LSTMRegressor(nn.Module):
    def __init__(self, backbone: LSTMBackbone, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), # LayerNorm adicionado para estabilidade
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 2),
            nn.Softplus() # Garante predição de gols não-negativa para funções como PoissonNLLLoss
        )

    def forward(self, numerical_inputs: torch.Tensor, categorical_inputs: torch.Tensor) -> torch.Tensor:
        backbone_output = self.backbone(numerical_inputs, categorical_inputs)
        return self.regression_head(backbone_output)

def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device):
    sequences, categorical_sequences, class_targets, regression_targets = batch
    return (
        sequences.to(device, non_blocking=True),
        categorical_sequences.to(device, non_blocking=True),
        class_targets.to(device, non_blocking=True),
        regression_targets.to(device, non_blocking=True),
    )


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
    model_type: str,
    validation_dataframe,
    numerical_feature_columns: list[str],
    categorical_feature_columns: list[str],
    sequence_length: int,
    device: torch.device,
) -> Any:
    predicted_frame = validation_dataframe.copy()
    predicted_frame["pred_resultado_partida"] = np.nan
    predicted_frame["pred_gols_mandante"] = np.nan
    predicted_frame["pred_gols_visitante"] = np.nan

    model.eval()
    numerical_batches: list[torch.Tensor] = []
    categorical_batches: list[torch.Tensor] = []
    row_positions: list[int] = []

    ordered = validation_dataframe.sort_values([YEAR_COLUMN, "data", "rodada"])
    for _, season_frame in ordered.groupby(YEAR_COLUMN, sort=True):
        season_frame = season_frame.sort_values(["data", "rodada"]).reset_index()
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
            _, predicted_classes = torch.max(outputs, 1)
            predicted_classes = predicted_classes.cpu().numpy()
            predicted_goals = np.full((len(predicted_classes), 2), np.nan)
        else:
            predicted_goals = torch.exp(outputs).cpu().numpy()
            predicted_classes = np.full(len(predicted_goals), np.nan)

    for row_position, predicted_class, (predicted_home_goals, predicted_away_goals) in zip(row_positions, predicted_classes, predicted_goals):
        if not np.isnan(predicted_class):
            predicted_frame.loc[row_position, "pred_resultado_partida"] = int(predicted_class)
        if not np.isnan(predicted_home_goals):
            predicted_frame.loc[row_position, "pred_gols_mandante"] = float(predicted_home_goals)
            predicted_frame.loc[row_position, "pred_gols_visitante"] = float(predicted_away_goals)

    return predicted_frame


def _compute_epoch_metrics(
    model: nn.Module,
    model_type: str,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_regression_loss = 0.0
    total_classification_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_abs_error = 0.0
    total_squared_error = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            sequences, categorical_sequences, class_targets, regression_targets = _move_batch(batch, device)
            outputs = model(sequences, categorical_sequences)
            batch_size = sequences.size(0)

            if model_type == "classifier":
                loss = criterion(outputs, class_targets)
                total_classification_loss += float(loss.item()) * batch_size
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
        return EpochMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

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
        mae=total_abs_error / regression_elements if model_type == "regressor" else 0.0,
        rmse=float(np.sqrt(total_squared_error / regression_elements)) if model_type == "regressor" else 0.0,
    )


def train_model(
    csv_path: str | Path = Path(__file__).resolve().parents[1] / "data" / "dataset_scaled.csv",
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
    
    if model_type not in ("classifier", "regressor"):
        raise ValueError(f"model_type deve ser 'classifier' ou 'regressor', não '{model_type}'")

    checkpoint_path = best_model_path or save_path
    bundle: SequenceBundle = build_sequence_bundle(
        csv_path=csv_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    runtime_device = get_device(device)

    backbone = LSTMBackbone(
        numerical_input_size=len(bundle.numerical_feature_columns),
        embedding_settings=bundle.embedding_settings,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(runtime_device)

    if model_type == "classifier":
        model = LSTMClassifier(backbone=backbone, hidden_size=hidden_size, dropout=dropout).to(runtime_device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.3, 0.7, 1.4], device=runtime_device), label_smoothing=0.05)  # Pesos iguais para as classes
    else:
        model = LSTMRegressor(backbone=backbone, hidden_size=hidden_size, dropout=dropout).to(runtime_device)
        criterion = nn.SmoothL1Loss(beta=2)#nn.PoissonNLLLoss(log_input=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=25, factor=0.5)
    use_amp = runtime_device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    prev_train_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_metrics: EpochMetrics | None = None
    patience_counter = 0

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        running_correct = 0
        running_samples = 0
        all_train_predictions = []
        all_train_targets = []
        epoch_start = time.time()
        total_steps = len(bundle.train_loader)

        for step, batch in enumerate(bundle.train_loader, start=1):
            sequences, categorical_sequences, class_targets, regression_targets = _move_batch(batch, runtime_device)
            optimizer.zero_grad(set_to_none=False)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(sequences, categorical_sequences)
                if model_type == "classifier":
                    loss = criterion(outputs, class_targets)
                    classification_loss = loss
                    regression_loss = torch.tensor(0.0)
                else:
                    loss = criterion(outputs, regression_targets)
                    regression_loss = loss
                    classification_loss = torch.tensor(0.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_actual = sequences.size(0)
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
            mae=0.0,
            rmse=0.0,
        )

        validation_metrics = _compute_epoch_metrics(
            model=model,
            model_type=model_type,
            loader=bundle.validation_loader,
            device=runtime_device,
            criterion=criterion,
        )

        if model_type == "classifier":
            summary_line = (
                f"Epoch {epoch}/{epochs} final | "
                f"train_class_loss={train_metrics.classification_loss:.6f} train_acc={train_metrics.accuracy:.6f} train_f1={train_metrics.f1_score:.6f} | "
                f"val_acc={validation_metrics.accuracy:.6f} val_f1={validation_metrics.f1_score:.6f}"
            )
        else:
            summary_line = (
                f"Epoch {epoch}/{epochs} final | "
                f"train_reg_loss={train_metrics.regression_loss:.6f} train_mae={train_metrics.mae:.6f} train_rmse={train_metrics.rmse:.6f} | "
                f"val_mae={validation_metrics.mae:.6f} val_rmse={validation_metrics.rmse:.6f}"
            )
        sys.stdout.write(summary_line + "\n")
        sys.stdout.flush()

        scheduler.step(validation_metrics.loss)

        history.append(
            {
                "epoch": epoch,
                "train": asdict(train_metrics),
                "validation": asdict(validation_metrics),
            }
        )

        if validation_metrics.loss < best_validation_loss:
            best_validation_loss = validation_metrics.loss
            best_state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            best_epoch = epoch
            best_validation_metrics = validation_metrics
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
                        "best_validation_metrics": asdict(best_validation_metrics),
                        "model_type": model_type,
                    },
                    checkpoint_path,
                )
        else:
            # Overfitting: loss de treino continuou caindo, mas a de validação não
            if train_metrics.loss < prev_train_loss:
                patience_counter += 1

        prev_train_loss = train_metrics.loss

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
    test_and_validation_dataframe = pd.concat([validation_dataframe, test_dataframe], ignore_index=True)
    validation_predictions_path = Path(validation_predictions_path)
    validation_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    validation_predictions = _collect_validation_predictions(
        model=model,
        model_type=model_type,
        validation_dataframe=test_and_validation_dataframe,
        numerical_feature_columns=bundle.numerical_feature_columns,
        categorical_feature_columns=bundle.categorical_feature_columns,
        sequence_length=sequence_length,
        device=runtime_device,
    )
    validation_predictions.to_csv(validation_predictions_path, index=False)

    test_metrics = _compute_epoch_metrics(
        model=model,
        model_type=model_type,
        loader=bundle.test_loader,
        device=runtime_device,
        criterion=criterion,
    )

    result: dict[str, Any] = {
        "model": model,
        "history": history,
        "test_metrics": asdict(test_metrics),
        "best_epoch": best_epoch,
        "best_validation_metrics": asdict(best_validation_metrics) if best_validation_metrics is not None else None,
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
                "best_validation_metrics": asdict(best_validation_metrics) if best_validation_metrics is not None else None,
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
    test_loss = test_metrics["loss"]
    
    train_acc = [h["train"]["accuracy"] for h in history]
    val_acc = [h["validation"]["accuracy"] for h in history]
    test_acc = test_metrics["accuracy"]
    
    train_f1 = [h["train"]["f1_score"] for h in history]
    val_f1 = [h["validation"]["f1_score"] for h in history]
    test_f1 = test_metrics["f1_score"]
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Treino")
    plt.plot(epochs, val_loss, label="Validação")
    plt.axhline(y=test_loss, color='r', linestyle='--', label=f"Teste Final: {test_loss:.4f}")
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
    plt.plot(epochs, val_acc, label="Validação")
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f"Teste Final: {test_acc:.4f}")
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
    plt.plot(epochs, val_f1, label="Validação")
    plt.axhline(y=test_f1, color='r', linestyle='--', label=f"Teste Final: {test_f1:.4f}")
    plt.xlabel("Época")
    plt.ylabel("F1 Score")
    plt.title("Evolução do F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "f1_score_curve.png")
    plt.close()


def main() -> None:
    output = train_model(epochs=10000, sequence_length=100, batch_size=64, hidden_size=16,
                         num_layers=1, dropout=0.2, learning_rate=1e-3, model_type="classifier",
                         save_path=Path(__file__).resolve().parents[0] / "checkpoints" / "lstm_checkpoint.pth",
                         best_model_path=Path(__file__).resolve().parents[0] / "checkpoints" / "lstm_best.pth",
                         validation_predictions_path=Path(__file__).resolve().parents[0] / "results" / "respostas_lstm.csv",
                         early_stopping_patience=25, 
                         csv_path=Path(__file__).resolve().parents[1] / "data" / "dataset_preprocessed.csv")


    print("Dispositivo:", output["device"])
    print("Métricas de teste:", output["test_metrics"])
    print("Melhor época:", output["best_epoch"])   
    
    plots_dir = Path(__file__).resolve().parents[0] / "plots"
    plot_metrics(output, plots_dir)
    print(f"Gráficos salvos em: {plots_dir}")


if __name__ == "__main__":
    main()