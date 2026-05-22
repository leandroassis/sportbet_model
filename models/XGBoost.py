from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
import xgboost as xgb  # type: ignore[import-not-found]

try:
    from .dataloader import CLASS_LABEL_MAP, FEATURE_COLUMNS, load_match_dataframe, split_match_dataframe
except ImportError:  # pragma: no cover
    from dataloader import CLASS_LABEL_MAP, FEATURE_COLUMNS, load_match_dataframe, split_match_dataframe


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


def _prepare_xy(dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_columns = [column for column in FEATURE_COLUMNS if column in dataframe.columns]
    x_values = dataframe[feature_columns].to_numpy(dtype=np.float32)
    y_class = dataframe[TARGET_COLUMNS[0]].astype(int).map(CLASS_LABEL_MAP).to_numpy(dtype=np.int64)
    y_home_goals = dataframe[TARGET_COLUMNS[1]].to_numpy(dtype=np.float32)
    y_away_goals = dataframe[TARGET_COLUMNS[2]].to_numpy(dtype=np.float32)
    return x_values, y_class, y_home_goals, y_away_goals


def _compute_metrics_from_preds(
    y_true_cls: np.ndarray, y_pred_cls: np.ndarray,
    y_true_home: np.ndarray, y_pred_home: np.ndarray,
    y_true_away: np.ndarray, y_pred_away: np.ndarray,
) -> ModelMetrics:
    # Substituindo sklearn por numpy puro para remover dependências externas externas
    acc = float(np.mean(y_true_cls == y_pred_cls))
    mae_h = float(np.mean(np.abs(y_true_home - y_pred_home)))
    mae_a = float(np.mean(np.abs(y_true_away - y_pred_away)))
    rmse_h = float(np.sqrt(np.mean((y_true_home - y_pred_home) ** 2)))
    rmse_a = float(np.sqrt(np.mean((y_true_away - y_pred_away) ** 2)))
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
    save_path: str | Path | None = None,
    best_model_path: str | Path | None = None,
    validation_predictions_path: str | Path = Path(__file__).resolve().parents[0] / "results" / "respostas_xgboost.csv",
) -> dict[str, Any]:
    
    # Stage 1: load and split
    dataframe = load_match_dataframe(csv_path)
    splits = split_match_dataframe(dataframe)

    x_train, y_train_cls, y_train_home, y_train_away = _prepare_xy(splits.train)
    x_val, y_val_cls, y_val_home, y_val_away = _prepare_xy(splits.validation)
    x_test, y_test_cls, y_test_home, y_test_away = _prepare_xy(splits.test)

    if len(x_val) == 0:
        raise ValueError("validation split is empty; check the year reconstruction logic before training")

    # Montando as DMatrices estruturadas
    dtrain_cls = xgb.DMatrix(x_train, label=y_train_cls)
    dtrain_home = xgb.DMatrix(x_train, label=y_train_home)
    dtrain_away = xgb.DMatrix(x_train, label=y_train_away)
    
    dval_cls = xgb.DMatrix(x_val, label=y_val_cls)
    dval_home = xgb.DMatrix(x_val, label=y_val_home)
    dval_away = xgb.DMatrix(x_val, label=y_val_away)
    
    dtest_cls = xgb.DMatrix(x_test)
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
        "tree_method": "hist",
    }

    # 1. Defina a função de decaimento (ex: decaimento de 5% a cada 100 rodadas)
    def lr_decay_function(boosting_round: int) -> float:
        decay_rate = 0.95
        step_size = 100
        return learning_rate * (decay_rate ** (boosting_round // step_size))

    # 2. Instancie o callback nativo do XGBoost
    lr_scheduler = xgb.callback.LearningRateScheduler(lr_decay_function)

    # Stage 2: Treinamento Nativo usando Early Stopping do XGBoost
    print("\n--- Treinando Classificador ---")
    classifier = xgb.train(
        params={**base_params, "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss"},
        dtrain=dtrain_cls,
        num_boost_round=n_estimators,
        evals=[(dtrain_cls, "train"), (dval_cls, "val")],
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
        verbose_eval=n_estimators // 10,
        callbacks=[lr_scheduler]
    )

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
    val_pred_cls = _predict_booster(classifier, dval_cls, is_classifier=True)
    val_pred_home = _predict_booster(regressor_home, dval_home)
    val_pred_away = _predict_booster(regressor_away, dval_away)

    test_pred_cls = _predict_booster(classifier, dtest_cls, is_classifier=True)
    test_pred_home = _predict_booster(regressor_home, dtest_home)
    test_pred_away = _predict_booster(regressor_away, dtest_away)

    validation_metrics = _compute_metrics_from_preds(
        y_val_cls, val_pred_cls, y_val_home, val_pred_home, y_val_away, val_pred_away
    )
    test_metrics = _compute_metrics_from_preds(
        y_test_cls, test_pred_cls, y_test_home, test_pred_home, y_test_away, test_pred_away
    )

    # Coleta de melhores iterações nativas
    best_iterations = {
        "classifier": getattr(classifier, "best_iteration", n_estimators - 1),
        "regressor_home": getattr(regressor_home, "best_iteration", n_estimators - 1),
        "regressor_away": getattr(regressor_away, "best_iteration", n_estimators - 1),
    }
    
    # Apenas para compatibilidade com o retorno anterior
    best_epoch = int(np.max(list(best_iterations.values()))) + 1

    # Stage 4: Exportação de predições
    predicted_frame = pd.concat([splits.validation, splits.test], ignore_index=True)
    predicted_frame["pred_resultado_partida"] = np.concatenate([val_pred_cls, test_pred_cls]).astype(int)
    predicted_frame["pred_gols_mandante"] = np.concatenate([val_pred_home, test_pred_home]).astype(float)
    predicted_frame["pred_gols_visitante"] = np.concatenate([val_pred_away, test_pred_away]).astype(float)

    validation_predictions_path = Path(validation_predictions_path)
    validation_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predicted_frame.to_csv(validation_predictions_path, index=False)

    result: dict[str, Any] = {
        "models": {
            "classifier": classifier,
            "regressor_home": regressor_home,
            "regressor_away": regressor_away,
        },
        "validation_metrics": validation_metrics.__dict__,
        "test_metrics": test_metrics.__dict__,
        "feature_columns": [column for column in FEATURE_COLUMNS if column in dataframe.columns],
        "class_mapping": CLASS_LABEL_MAP,
        "inverse_class_mapping": INV_CLASS_LABEL_MAP,
        "best_iterations": best_iterations,
        "best_epoch": best_epoch,
        "validation_predictions_path": str(validation_predictions_path),
        "device": "cpu",
    }

    # Stage 5: Salvar checkpoint (.joblib)
    _save_checkpoints(result, save_path, best_model_path)

    return result


def main() -> None:
    output = train_model(
        n_estimators=20000, max_depth=4, learning_rate=5e-4, subsample=0.9,
        colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0, early_stopping_rounds=50,
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