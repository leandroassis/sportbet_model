from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
import xgboost as xgb  # type: ignore[import-not-found]

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

try:
    from .dataloader import (
        CLASS_LABEL_MAP,
        FEATURE_COLUMNS,
        load_match_dataframe,
        split_match_dataframe,
        EmbeddingSettings,
    )
except ImportError:  # pragma: no cover
    from dataloader import (
        CLASS_LABEL_MAP,
        FEATURE_COLUMNS,
        load_match_dataframe,
        split_match_dataframe,
        EmbeddingSettings,
    )


YEAR_COLUMN = "ano_campeonato"
INV_CLASS_LABEL_MAP = {0: -1, 1: 0, 2: 1}
TARGET_COLUMNS = ("resultado_partida", "gols_mandante", "gols_visitante")


@dataclass(frozen=True)
class TrainedModels:
    classifier: xgb.Booster
    regressor_home: xgb.Booster
    regressor_away: xgb.Booster


@dataclass(frozen=True)
class ModelMetrics:
    accuracy: float
    mae_home: float
    mae_away: float
    rmse_home: float
    rmse_away: float


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_settings: EmbeddingSettings):
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(cardinality, dim)
                for name, (cardinality, dim) in embedding_settings.dimensions.items()
            }
        )
        total_embedding_dim = sum(dim for _, dim in embedding_settings.dimensions.values())
        self.classifier = nn.Linear(total_embedding_dim, 3)
        self.regressor = nn.Linear(total_embedding_dim, 2)

    def forward(self, x_cat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cat_embeds = [self.embeddings[name](x_cat[:, i]) for i, name in enumerate(self.embeddings)]
        cat_embeds = torch.cat(cat_embeds, dim=1)
        
        class_out = self.classifier(cat_embeds)
        reg_out = self.regressor(cat_embeds)
        return class_out, reg_out

    def get_embeddings(self, x_cat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cat_embeds = [self.embeddings[name](x_cat[:, i]) for i, name in enumerate(self.embeddings)]
            return torch.cat(cat_embeds, dim=1)


def _train_embedding_model(
    embedding_net: EmbeddingNet,
    x_cat: np.ndarray,
    y_cls: np.ndarray,
    y_home: np.ndarray,
    y_away: np.ndarray,
    epochs: int = 5,
    batch_size: int = 256,
) -> None:
    dataset = TensorDataset(
        torch.from_numpy(x_cat).long(),
        torch.from_numpy(y_cls).long(),
        torch.from_numpy(np.stack([y_home, y_away], axis=1)).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(embedding_net.parameters(), lr=1e-2)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.PoissonNLLLoss()

    print("\n--- Pre-training Embeddings ---")
    for epoch in range(1, epochs + 1):
        for x_batch, y_cls_batch, y_reg_batch in loader:
            optimizer.zero_grad()
            cls_out, reg_out = embedding_net(x_batch)
            loss = criterion_reg(reg_out, y_reg_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")


def _prepare_xy(
    dataframe: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_num = dataframe[numerical_cols].to_numpy(dtype=np.float32)
    x_cat = dataframe[categorical_cols].to_numpy(dtype=np.int64)
    #y_class = dataframe[TARGET_COLUMNS[0]].astype(int).map(CLASS_LABEL_MAP).to_numpy(dtype=np.int64)
    y_class = dataframe[TARGET_COLUMNS[0]].astype(int).to_numpy(dtype=np.int64)
    y_home_goals = dataframe[TARGET_COLUMNS[1]].to_numpy(dtype=np.float32)
    y_away_goals = dataframe[TARGET_COLUMNS[2]].to_numpy(dtype=np.float32)
    return x_num, x_cat, y_class, y_home_goals, y_away_goals


def _compute_metrics_from_preds(
    y_true_home: np.ndarray, y_pred_home: np.ndarray,
    y_true_away: np.ndarray, y_pred_away: np.ndarray,
    threshold: float = 0.5,
) -> ModelMetrics:
    gol_diff_pred = y_pred_home - y_pred_away
    gol_diff_true = y_true_home - y_true_away

    pred_cls = ((np.abs(gol_diff_pred) > threshold).astype(int) *
                (2 * (gol_diff_pred > 0).astype(int) - 1) + 1)
    true_cls = ((np.abs(gol_diff_true) > 0.5).astype(int) *
                (2 * (gol_diff_true > 0).astype(int) - 1) + 1)

    acc = float(np.mean(pred_cls == true_cls))
    mae_h = float(np.mean(np.abs(y_pred_home - y_true_home)))
    mae_a = float(np.mean(np.abs(y_pred_away - y_true_away)))
    rmse_h = float(np.sqrt(np.mean((y_pred_home - y_true_home) ** 2)))
    rmse_a = float(np.sqrt(np.mean((y_pred_away - y_true_away) ** 2)))
    return ModelMetrics(accuracy=acc, mae_home=mae_h, mae_away=mae_a, rmse_home=rmse_h, rmse_away=rmse_a)


def _predict_booster(model: xgb.Booster, dmatrix: xgb.DMatrix, is_classifier: bool = False) -> np.ndarray:
    # O XGBoost usa automaticamente a melhor iteração salva se early_stopping foi ativado
    predictions = model.predict(dmatrix)
    if is_classifier:
        return predictions.argmax(axis=1).astype(np.int64)
    return np.asarray(predictions)


def _save_checkpoints(result: dict[str, Any], save_path: str | Path | None, best_model_path: str | Path | None) -> None:
    checkpoint_paths = [path for path in (save_path, best_model_path) if path is not None]
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        dump(result, checkpoint_path)


def train_model(
    csv_path: str | Path = Path(__file__).resolve().parents[1] / "data" / "dataset_scaled.csv",
    random_state: int = 42,
    n_estimators: int = 700,
    max_depth: int = 6,
    learning_rate: float = 0.03,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    early_stopping_rounds: int = 50,
    embedding_epochs: int = 5,
    save_path: str | Path | None = None,
    best_model_path: str | Path | None = None,
    validation_predictions_path: str | Path = Path(__file__).resolve().parents[0] / "results" / "respostas_xgboost.csv",
) -> dict[str, Any]:
    
    # Stage 1: load and split
    dataframe = load_match_dataframe(csv_path)
    splits = split_match_dataframe(dataframe)

    categorical_feature_columns = [
        col for col in [
            "arbitro", "estadio", "tecnico_mandante", "tecnico_visitante", "time_mandante", "time_visitante"
        ] if col in dataframe.columns
    ]
    numerical_feature_columns = [
        col for col in FEATURE_COLUMNS if col not in categorical_feature_columns and col in dataframe.columns
    ]

    cardinalities = {col: int(dataframe[col].max() + 1) for col in categorical_feature_columns}
    embedding_dimensions = {
        col: (cardinalities[col], min(50, (cardinalities[col] + 1) // 2))
        for col in categorical_feature_columns
    }
    embedding_settings = EmbeddingSettings(cardinalities=cardinalities, dimensions=embedding_dimensions)

    x_train_num, x_train_cat, y_train_cls, y_train_home, y_train_away = _prepare_xy(splits.train, numerical_feature_columns, categorical_feature_columns)
    x_val_num, x_val_cat, y_val_cls, y_val_home, y_val_away = _prepare_xy(splits.validation, numerical_feature_columns, categorical_feature_columns)
    x_test_num, x_test_cat, y_test_cls, y_test_home, y_test_away = _prepare_xy(splits.test, numerical_feature_columns, categorical_feature_columns)

    if len(x_val_num) == 0:
        raise ValueError("validation split is empty; check the year reconstruction logic before training")

    # Pre-train embeddings
    embedding_net = EmbeddingNet(embedding_settings)
    _train_embedding_model(embedding_net, x_train_cat, y_train_cls, y_train_home, y_train_away, epochs=embedding_epochs)

    # Get embedding features
    train_embeds = embedding_net.get_embeddings(torch.from_numpy(x_train_cat).long()).numpy()
    val_embeds = embedding_net.get_embeddings(torch.from_numpy(x_val_cat).long()).numpy()
    test_embeds = embedding_net.get_embeddings(torch.from_numpy(x_test_cat).long()).numpy()

    # Combine numerical and embedding features
    x_train = np.concatenate([x_train_num, train_embeds], axis=1)
    x_val = np.concatenate([x_val_num, val_embeds], axis=1)
    x_test = np.concatenate([x_test_num, test_embeds], axis=1)

    # Montando as DMatrices estruturadas
    dtrain_home = xgb.DMatrix(x_train, label=y_train_home)
    dtrain_away = xgb.DMatrix(x_train, label=y_train_away)

    dval_home = xgb.DMatrix(x_val, label=y_val_home)
    dval_away = xgb.DMatrix(x_val, label=y_val_away)

    dtest_home = xgb.DMatrix(x_test)
    dtest_away = xgb.DMatrix(x_test)

    # Parâmetros base compartilhados
    base_params = {
        "max_depth": max_depth,
        "eta": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "alpha": reg_alpha,
        "lambda": reg_lambda,
        "seed": random_state,
        "tree_method": "auto",
        "device": "cuda",
        "grow_policy": "depthwise",
    }

    # 1. Defina a função de decaimento (ex: decaimento de 5% a cada 100 rodadas)
    def lr_decay_function(boosting_round: int) -> float:
        decay_rate = 0.95
        step_size = 100
        return learning_rate * (decay_rate ** (boosting_round // step_size))

    # 2. Instancie o callback nativo do XGBoost
    lr_scheduler = xgb.callback.LearningRateScheduler(lr_decay_function)

    # Stage 2: Treinamento Nativo usando Early Stopping do XGBoost
    print("\n=== GPU ACCELERATION ENABLED ===")
    print(f"XGBoost: tree_method={base_params['tree_method']}, device={base_params['device']}")
    print("==================================\n")

    print("\n--- Treinando Regressor (Mandante) ---")
    regressor_home = xgb.train(
        params={**base_params, "objective": "reg:squarederror", "eval_metric": "rmse"},
        dtrain=dtrain_home,
        num_boost_round=n_estimators,
        evals=[(dtrain_home, "train"), (dval_home, "val")],
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
        verbose_eval=n_estimators // 10,
        callbacks=[lr_scheduler]
    )

    print("\n--- Treinando Regressor (Visitante) ---")
    regressor_away = xgb.train(
        params={**base_params, "objective": "reg:squarederror", "eval_metric": "rmse"},
        dtrain=dtrain_away,
        num_boost_round=n_estimators,
        evals=[(dtrain_away, "train"), (dval_away, "val")],
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
        verbose_eval=n_estimators // 10,
        callbacks=[lr_scheduler]
    )

    # Stage 3: Predições e Métricas (Validação e Teste)
    val_pred_home = _predict_booster(regressor_home, dval_home)
    val_pred_away = _predict_booster(regressor_away, dval_away)

    test_pred_home = _predict_booster(regressor_home, dtest_home)
    test_pred_away = _predict_booster(regressor_away, dtest_away)

    validation_metrics = _compute_metrics_from_preds(
        y_val_home, val_pred_home, y_val_away, val_pred_away
    )
    test_metrics = _compute_metrics_from_preds(
        y_test_home, test_pred_home, y_test_away, test_pred_away
    )

    # Coleta de melhores iterações nativas
    best_iterations = {
        "regressor_home": getattr(regressor_home, "best_iteration", n_estimators - 1),
        "regressor_away": getattr(regressor_away, "best_iteration", n_estimators - 1),
    }

    # Apenas para compatibilidade com o retorno anterior
    best_epoch = int(np.max(list(best_iterations.values()))) + 1

    # Stage 4: Calcular classificações post-hoc com base no threshold
    threshold = 0.5
    val_gol_diff = val_pred_home - val_pred_away
    test_gol_diff = test_pred_home - test_pred_away

    val_pred_cls = ((np.abs(val_gol_diff) > threshold).astype(int) *
                    (2 * (val_gol_diff > 0).astype(int) - 1) + 1)
    test_pred_cls = ((np.abs(test_gol_diff) > threshold).astype(int) *
                     (2 * (test_gol_diff > 0).astype(int) - 1) + 1)

    # Stage 5: Exportação de predições
    predicted_frame = pd.concat([splits.validation, splits.test], ignore_index=True)
    predicted_frame["pred_resultado_partida"] = np.concatenate([val_pred_cls, test_pred_cls]).astype(int)
    predicted_frame["pred_gols_mandante"] = np.concatenate([val_pred_home, test_pred_home]).astype(float)
    predicted_frame["pred_gols_visitante"] = np.concatenate([val_pred_away, test_pred_away]).astype(float)

    validation_predictions_path = Path(validation_predictions_path)
    validation_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predicted_frame.to_csv(validation_predictions_path, index=False)

    result: dict[str, Any] = {
        "models": {
            "regressor_home": regressor_home,
            "regressor_away": regressor_away,
        },
        "embedding_net_state_dict": embedding_net.state_dict(),
        "embedding_settings": embedding_settings,
        "numerical_feature_columns": numerical_feature_columns,
        "categorical_feature_columns": categorical_feature_columns,
        "validation_metrics": validation_metrics.__dict__,
        "test_metrics": test_metrics.__dict__,
        "feature_columns": numerical_feature_columns + list(embedding_settings.dimensions.keys()),
        "class_mapping": CLASS_LABEL_MAP,
        "inverse_class_mapping": INV_CLASS_LABEL_MAP,
        "best_iterations": best_iterations,
        "best_epoch": best_epoch,
        "validation_predictions_path": str(validation_predictions_path),
        "threshold": threshold,
        "device": "cpu",
    }

    # Stage 5: Salvar checkpoint (.joblib)
    _save_checkpoints(result, save_path, best_model_path)

    return result


def main() -> None:
    output = train_model(
        n_estimators=20000, max_depth=4, learning_rate=5e-4, subsample=0.9,
        colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0, early_stopping_rounds=50,
        embedding_epochs=500,
        save_path=Path(__file__).resolve().parents[0] / "checkpoints" / "xgboost_checkpoint.joblib",
        best_model_path=Path(__file__).resolve().parents[0] / "checkpoints" / "xgboost_best.joblib",
        validation_predictions_path=Path(__file__).resolve().parents[0] / "results" / "respostas_xgboost.csv",
    )

    print("\n\n--- RESULTADOS FINAIS ---")
    print("Dispositivo:", output["device"])
    print("Melhores iterações:", output["best_iterations"])
    print("Validação:", output["validation_metrics"])
    print("Teste:", output["test_metrics"])


if __name__ == "__main__":
    main()