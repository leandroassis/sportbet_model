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

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")

try:
    from .dataloader import (
        CLASS_LABEL_MAP,
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
        CLASS_LABEL_MAP,
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
    loss: float
    classification_loss: float
    regression_loss: float
    accuracy: float
    mae: float
    rmse: float


class LSTMBackbone(nn.Module):
    def __init__(
        self,
        numerical_input_size: int,
        embedding_settings: EmbeddingSettings,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_layers = nn.ModuleDict(
            {
                name: nn.Embedding(num_embeddings, dim)
                for name, (num_embeddings, dim) in embedding_settings.dimensions.items()
            }
        )
        
        lstm_input_size = numerical_input_size + sum(dim for _, dim in embedding_settings.dimensions.values())
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classification_head = nn.Linear(hidden_size, 3)
        # Mantido linear sem ativação. A PoissonNLLLoss(log_input=True) calculará os gradientes
        # de contagem de gols diretamente sob a transformação logarítmica estável.
        self.regression_head = nn.Linear(hidden_size, 2)

    def forward(self, numerical_inputs: torch.Tensor, categorical_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        embeddings = [
            self.embedding_layers[name](categorical_inputs[..., i])
            for i, name in enumerate(self.embedding_layers)
        ]
        
        combined_input = torch.cat([numerical_inputs] + embeddings, dim=-1)
        
        outputs, _ = self.lstm(combined_input)
        last_state = outputs[:, -1, :]
        latent = self.head(last_state)
        classification_logits = self.classification_head(latent)
        regression_outputs = self.regression_head(latent)
        return classification_logits, regression_outputs


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
        classification_logits, regression_outputs = model(numerical_batch_tensor, categorical_batch_tensor)
        predicted_classes = classification_logits.argmax(dim=1).cpu().numpy()
        # Modificado: Como a rede prevê em espaço logarítmico, aplicamos a exponencial para retornar
        # as predições para a escala real de gols ao salvar o arquivo final.
        predicted_goals = torch.exp(regression_outputs).cpu().numpy()

    for row_position, predicted_class, (predicted_home_goals, predicted_away_goals) in zip(row_positions, predicted_classes, predicted_goals):
        predicted_frame.loc[row_position, "pred_resultado_partida"] = int(predicted_class)
        predicted_frame.loc[row_position, "pred_gols_mandante"] = float(predicted_home_goals)
        predicted_frame.loc[row_position, "pred_gols_visitante"] = float(predicted_away_goals)

    return predicted_frame


def _compute_epoch_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion_classification: nn.Module,
    criterion_regression: nn.Module,
    regression_weight: float,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_regression_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_abs_error = 0.0
    total_squared_error = 0.0

    with torch.no_grad():
        for batch in loader:
            sequences, categorical_sequences, class_targets, regression_targets = _move_batch(batch, device)
            classification_logits, regression_outputs = model(sequences, categorical_sequences)

            classification_loss = criterion_classification(classification_logits, class_targets)
            regression_loss = criterion_regression(regression_outputs, regression_targets)
            loss = classification_loss + regression_weight * regression_loss

            batch_size = sequences.size(0)
            total_loss += float(loss.item()) * batch_size
            total_classification_loss += float(classification_loss.item()) * batch_size
            total_regression_loss += float(regression_loss.item()) * batch_size
            total_samples += batch_size

            predictions = classification_logits.argmax(dim=1)
            total_correct += int((predictions == class_targets).sum().item())

            # Modificado: Para calcular métricas reais (MAE/RMSE) de gols decimais comparados com inteiros,
            # aplicamos o mapeamento exponencial nas predições saídas do espaço logarítmico da rede.
            real_scale_predictions = torch.exp(regression_outputs)
            prediction_error = real_scale_predictions - regression_targets
            total_abs_error += float(prediction_error.abs().sum().item())
            total_squared_error += float((prediction_error ** 2).sum().item())

    if total_samples == 0:
        return EpochMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    regression_elements = total_samples * 2
    return EpochMetrics(
        loss=total_loss / total_samples,
        classification_loss=total_classification_loss / total_samples,
        regression_loss=total_regression_loss / total_samples,
        accuracy=total_correct / total_samples,
        mae=total_abs_error / regression_elements,
        rmse=float(np.sqrt(total_squared_error / regression_elements)),
    )


def train_model(
    csv_path: str | Path = Path(__file__).resolve().parents[1] / "data" / "dataset_scaled.csv",
    sequence_length: int = 10,
    batch_size: int = 64,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    epochs: int = 30,
    regression_weight: float = 1.0,
    early_stopping_patience: int = 10,
    num_workers: int = 0,
    device: str | None = None,
    save_path: str | Path | None = None,
    best_model_path: str | Path | None = None,
    validation_predictions_path: str | Path = Path(__file__).resolve().parents[0] / "respostas_lstm.csv",
) -> dict[str, Any]:
    checkpoint_path = best_model_path or save_path
    bundle: SequenceBundle = build_sequence_bundle(
        csv_path=csv_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    runtime_device = get_device(device)

    model = LSTMBackbone(
        numerical_input_size=len(bundle.numerical_feature_columns),
        embedding_settings=bundle.embedding_settings,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(runtime_device)

    criterion_classification = nn.CrossEntropyLoss()
    # Modificado: Especificado explicitamente log_input=True. Isso faz o PyTorch aceitar as
    # saídas lineares sem ativação e aplicar a transformação exponencial estável internamente na perda.
    criterion_regression = nn.PoissonNLLLoss(log_input=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.7)
    use_amp = runtime_device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
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
        running_classification_loss = 0.0
        running_regression_loss = 0.0
        running_correct = 0
        running_samples = 0
        epoch_start = time.time()
        total_steps = len(bundle.train_loader)

        for step, batch in enumerate(bundle.train_loader, start=1):
            sequences, categorical_sequences, class_targets, regression_targets = _move_batch(batch, runtime_device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                classification_logits, regression_outputs = model(sequences, categorical_sequences)
                classification_loss = criterion_classification(classification_logits, class_targets)
                regression_loss = criterion_regression(regression_outputs, regression_targets)
                loss = classification_loss + regression_weight * regression_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_actual = sequences.size(0)
            running_samples += batch_size_actual
            running_loss += float(loss.item()) * batch_size_actual
            running_classification_loss += float(classification_loss.item()) * batch_size_actual
            running_regression_loss += float(regression_loss.item()) * batch_size_actual
            running_correct += int((classification_logits.argmax(dim=1) == class_targets).sum().item())

            _render_progress(
                epoch=epoch,
                epochs=epochs,
                step=step,
                total_steps=total_steps,
                loss=running_loss / running_samples if running_samples else 0.0,
                accuracy=running_correct / running_samples if running_samples else 0.0,
                elapsed=time.time() - epoch_start,
            )

        sys.stdout.write("\n")
        sys.stdout.flush()

        train_metrics = EpochMetrics(
            loss=running_loss / running_samples if running_samples else 0.0,
            classification_loss=running_classification_loss / running_samples if running_samples else 0.0,
            regression_loss=running_regression_loss / running_samples if running_samples else 0.0,
            accuracy=running_correct / running_samples if running_samples else 0.0,
            mae=0.0,
            rmse=0.0,
        )

        validation_metrics = _compute_epoch_metrics(
            model=model,
            loader=bundle.validation_loader,
            device=runtime_device,
            criterion_classification=criterion_classification,
            criterion_regression=criterion_regression,
            regression_weight=regression_weight,
        )

        summary_line = (
            f"Epoch {epoch}/{epochs} final | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} "
            f"val_acc={validation_metrics.accuracy:.4f}"
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

        if validation_metrics.accuracy > best_validation_loss:
            best_validation_loss = validation_metrics.accuracy
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
                        "class_mapping": bundle.class_mapping,
                        "best_epoch": best_epoch,
                        "best_validation_metrics": asdict(best_validation_metrics),
                    },
                    checkpoint_path,
                )
        else:
            patience_counter += 1

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
        validation_dataframe=test_and_validation_dataframe,
        numerical_feature_columns=bundle.numerical_feature_columns,
        categorical_feature_columns=bundle.categorical_feature_columns,
        sequence_length=sequence_length,
        device=runtime_device,
    )
    validation_predictions.to_csv(validation_predictions_path, index=False)

    test_metrics = _compute_epoch_metrics(
        model=model,
        loader=bundle.test_loader,
        device=runtime_device,
        criterion_classification=criterion_classification,
        criterion_regression=criterion_regression,
        regression_weight=regression_weight,
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


def main() -> None:
    output = train_model(epochs=10000, sequence_length=4, batch_size=128, hidden_size=32,
                         num_layers=2, dropout=0.1, learning_rate=5e-3, regression_weight=0.2,
                         save_path=Path(__file__).resolve().parents[0] / "checkpoints" / "lstm_checkpoint.pth",
                         best_model_path=Path(__file__).resolve().parents[0] / "checkpoints" / "lstm_best.pth",
                         validation_predictions_path=Path(__file__).resolve().parents[0] / "results" / "respostas_lstm.csv",
                         early_stopping_patience=500)


    print("Dispositivo:", output["device"])
    print("Métricas de teste:", output["test_metrics"])
    print("Melhor época:", output["best_epoch"])   


if __name__ == "__main__":
    main()