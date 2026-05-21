from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from xgboost import DMatrix, XGBClassifier, XGBRegressor, train  # type: ignore[import-not-found]
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

try:
    from .dataloader import CLASS_LABEL_MAP, FEATURE_COLUMNS, load_match_dataframe, split_match_dataframe
except ImportError:  # pragma: no cover - fallback when run as a standalone script
    from dataloader import CLASS_LABEL_MAP, FEATURE_COLUMNS, load_match_dataframe, split_match_dataframe


YEAR_COLUMN = "ano_campeonato"
INV_CLASS_LABEL_MAP = {0: -1, 1: 0, 2: 1}
TARGET_COLUMNS = ("resultado_partida", "gols_mandante", "gols_visitante")


def _prepare_xy(dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_columns = [column for column in FEATURE_COLUMNS if column in dataframe.columns]
    x_values = dataframe[feature_columns].to_numpy(dtype=np.float32)
    y_class = dataframe[TARGET_COLUMNS[0]].astype(int).map(CLASS_LABEL_MAP).to_numpy(dtype=np.int64)
    y_home_goals = dataframe[TARGET_COLUMNS[1]].to_numpy(dtype=np.float32)
    y_away_goals = dataframe[TARGET_COLUMNS[2]].to_numpy(dtype=np.float32)
    return x_values, y_class, y_home_goals, y_away_goals


def _train_booster(
    *,
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int,
    early_stopping_rounds: int,
) -> Any:
    dtrain = DMatrix(x_train, label=y_train)
    dval = DMatrix(x_val, label=y_val)
    booster = train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return booster


def _predict_classifier(model: Any, features: np.ndarray) -> np.ndarray:
    dmatrix = DMatrix(features)
    if getattr(model, "best_iteration", None) is not None:
        return np.asarray(model.predict(dmatrix, iteration_range=(0, model.best_iteration + 1)))
    return np.asarray(model.predict(dmatrix))


def _predict_regressor(model: Any, features: np.ndarray) -> np.ndarray:
    dmatrix = DMatrix(features)
    if getattr(model, "best_iteration", None) is not None:
        return np.asarray(model.predict(dmatrix, iteration_range=(0, model.best_iteration + 1)))
    return np.asarray(model.predict(dmatrix))


def _build_validation_predictions(
    validation_dataframe: pd.DataFrame,
    predicted_class_labels: np.ndarray,
    predicted_home_goals: np.ndarray,
    predicted_away_goals: np.ndarray,
) -> pd.DataFrame:
    predicted_frame = validation_dataframe.copy()
    predicted_frame["pred_resultado_partida"] = [INV_CLASS_LABEL_MAP[int(label)] for label in predicted_class_labels]
    predicted_frame["pred_gols_mandante"] = predicted_home_goals.astype(float)
    predicted_frame["pred_gols_visitante"] = predicted_away_goals.astype(float)
    return predicted_frame


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
    dataframe = load_match_dataframe(csv_path)
    splits = split_match_dataframe(dataframe)

    x_train, y_train_cls, y_train_home, y_train_away = _prepare_xy(splits.train)
    x_val, y_val_cls, y_val_home, y_val_away = _prepare_xy(splits.validation)
    x_test, y_test_cls, y_test_home, y_test_away = _prepare_xy(splits.test)

    if len(x_val) == 0:
        raise ValueError("validation split is empty; check the year reconstruction logic before training")

    classifier_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": max_depth,
        "eta": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "alpha": reg_alpha,
        "lambda": reg_lambda,
        "seed": random_state,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
    }
    regressor_params = {
        "objective": "reg:squarederror",
        "max_depth": max_depth,
        "eta": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "alpha": reg_alpha,
        "lambda": reg_lambda,
        "seed": random_state,
        "tree_method": "hist",
        "eval_metric": "rmse",
    }

    classifier = _train_booster(
        params=classifier_params,
        x_train=x_train,
        y_train=y_train_cls,
        x_val=x_val,
        y_val=y_val_cls,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )
    regressor_home = _train_booster(
        params=regressor_params,
        x_train=x_train,
        y_train=y_train_home,
        x_val=x_val,
        y_val=y_val_home,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )
    regressor_away = _train_booster(
        params=regressor_params,
        x_train=x_train,
        y_train=y_train_away,
        x_val=x_val,
        y_val=y_val_away,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )

    val_pred_cls = _predict_classifier(classifier, x_val)
    val_pred_home = _predict_regressor(regressor_home, x_val)
    val_pred_away = _predict_regressor(regressor_away, x_val)

    test_pred_cls = _predict_classifier(classifier, x_test)
    test_pred_home = _predict_regressor(regressor_home, x_test)
    test_pred_away = _predict_regressor(regressor_away, x_test)

    validation_metrics = {
        "accuracy": float(accuracy_score(y_val_cls, val_pred_cls)),
        "mae_home": float(mean_absolute_error(y_val_home, val_pred_home)),
        "mae_away": float(mean_absolute_error(y_val_away, val_pred_away)),
        "rmse_home": float(np.sqrt(mean_squared_error(y_val_home, val_pred_home))),
        "rmse_away": float(np.sqrt(mean_squared_error(y_val_away, val_pred_away))),
    }
    test_metrics = {
        "accuracy": float(accuracy_score(y_test_cls, test_pred_cls)),
        "mae_home": float(mean_absolute_error(y_test_home, test_pred_home)),
        "mae_away": float(mean_absolute_error(y_test_away, test_pred_away)),
        "rmse_home": float(np.sqrt(mean_squared_error(y_test_home, test_pred_home))),
        "rmse_away": float(np.sqrt(mean_squared_error(y_test_away, test_pred_away))),
    }

    best_iterations = {
        "classifier": getattr(classifier, "best_iteration", None),
        "regressor_home": getattr(regressor_home, "best_iteration", None),
        "regressor_away": getattr(regressor_away, "best_iteration", None),
    }
    resolved_best_iterations = [iteration for iteration in best_iterations.values() if iteration is not None]
    best_epoch = int(max(resolved_best_iterations) + 1) if resolved_best_iterations else n_estimators

    validation_predictions = _build_validation_predictions(
        validation_dataframe=splits.validation,
        predicted_class_labels=val_pred_cls,
        predicted_home_goals=val_pred_home,
        predicted_away_goals=val_pred_away,
    )

    validation_predictions_path = Path(validation_predictions_path)
    validation_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    validation_predictions.to_csv(validation_predictions_path, index=False)

    result: dict[str, Any] = {
        "models": {
            "classifier": classifier,
            "regressor_home": regressor_home,
            "regressor_away": regressor_away,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "feature_columns": [column for column in FEATURE_COLUMNS if column in dataframe.columns],
        "class_mapping": CLASS_LABEL_MAP,
        "inverse_class_mapping": INV_CLASS_LABEL_MAP,
        "best_iterations": best_iterations,
        "best_epoch": best_epoch,
        "validation_predictions_path": str(validation_predictions_path),
        "device": "cpu",
    }

    checkpoint_paths = [path for path in (save_path, best_model_path) if path is not None]
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        dump(result, checkpoint_path)

    return result


def main() -> None:
    output = train_model(
        save_path=Path(__file__).resolve().parents[0] / "checkpoints" / "xgboost_checkpoint.joblib",
        best_model_path=Path(__file__).resolve().parents[0] / "checkpoints" / "xgboost_best.joblib",
        validation_predictions_path=Path(__file__).resolve().parents[0] / "results" / "respostas_xgboost.csv",
    )

    print("Dispositivo:", output["device"])
    print("Melhor época:", output["best_epoch"])
    print("Melhores iterações:", output["best_iterations"])
    print("Validação:", output["validation_metrics"])
    print("Teste:", output["test_metrics"])
    print("Respostas de validação salvas em:", output["validation_predictions_path"])


if __name__ == "__main__":
    main()
