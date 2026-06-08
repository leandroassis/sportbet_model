from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import numpy as np


TARGET_COLUMNS = ("resultado_empate", 'resultado_vitoria_mandante', 'resultado_vitoria_visitante',
                  "gols_mandante", "gols_visitante")
YEAR_COLUMN = "ano_campeonato"
FEATURE_COLUMNS = [
    "data",
    "rodada",
    "publico",
    "publico_max",
    "colocacao_mandante",
    "colocacao_visitante",
    "valor_equipe_titular_mandante",
    "valor_equipe_titular_visitante",
    "idade_media_titular_mandante",
    "idade_media_titular_visitante",
    "AvgCH", "AvgCD", "AvgCA",
    "gols_pro_mandante",
    "gols_pro_visitante",
    "gols_sofridos_mandante",
    "gols_sofridos_visitante",
    "saldo_gols_mandante",
    "saldo_gols_visitante",
    "vitorias_mandante",
    "vitorias_visitante",
    "empates_mandante",
    "empates_visitante",
    "derrotas_mandante",
    "derrotas_visitante",
    "dia",
    "mes",
    "gols_sofridos_media_mandante",
    "gols_sofridos_media_visitante",
    "gols_marcados_media_mandante",
    "gols_marcados_media_visitante",
    "vitorias_confronto_mandante",
    "vitorias_confronto_visitante",
    "empates_confronto",
    "vitorias_seguidas_mandante",
    "vitorias_seguidas_visitante",
    "colocacao_media_mandante",
    'colocacao_media_visitante',
    'derrotas_seguidas_mandante',
    'derrotas_seguidas_visitante',
    'elo_mandante',
    'elo_visitante',
]

FEATURES_TO_DROP = ['gols_sofridos_media_visitante', 'gols_marcados_media_visitante', 'gols_pro_mandante', 'vitorias_confronto_mandante', 'empates_confronto',
                    'derrotas_mandante', 'colocacao_media_mandante', 'publico', 'rodada', 'dia', 'valor_equipe_titular_visitante', 'idade_media_titular_mandante',
                    'idade_media_titular_visitante', 'valor_equipe_titular_mandante', 'missing_colocacao_visitante', 'publico_max', 'missing_rodada']

@dataclass(frozen=True)
class EmbeddingSettings:
    cardinalities: dict[str, int]
    dimensions: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class SequenceBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    validation_loader: DataLoader
    numerical_feature_columns: list[str]
    categorical_feature_columns: list[str]
    input_size: int
    embedding_settings: EmbeddingSettings


class MatchSequenceDataset(Dataset):
    def __init__(
        self,
        numerical_dataframe: pd.DataFrame,
        categorical_dataframe: pd.DataFrame,
        numerical_feature_columns: list[str],
        categorical_feature_columns: list[str],
        sequence_length: int,
    ) -> None:
        self.sequences: list[torch.Tensor] = []
        self.categorical_sequences: list[torch.Tensor] = []
        self.class_targets: list[torch.Tensor] = []
        self.regression_targets: list[torch.Tensor] = []

        # Combine the dataframes for sorting, keeping the original index
        num_df_reset = numerical_dataframe.reset_index(drop=True)
        cat_df_reset = categorical_dataframe[categorical_feature_columns].reset_index(drop=True)
        combined_df = pd.concat([num_df_reset, cat_df_reset], axis=1)
        ordered = combined_df.sort_values([YEAR_COLUMN])

        for _, season_frame in ordered.groupby(YEAR_COLUMN, sort=True):
            if len(season_frame) < sequence_length:
                continue

            numerical_features = torch.tensor(season_frame[numerical_feature_columns].to_numpy(), dtype=torch.float32)
            categorical_features = torch.tensor(season_frame[categorical_feature_columns].to_numpy(), dtype=torch.int64)

            class_values = torch.tensor(season_frame[['resultado_empate', 'resultado_vitoria_mandante', 'resultado_vitoria_visitante']].to_numpy(), dtype=torch.float32)
            regression_values = torch.tensor(season_frame[["gols_mandante", "gols_visitante"]].to_numpy(), dtype=torch.float32)

            for end_index in range(sequence_length - 1, len(season_frame)):
                start_index = end_index - sequence_length + 1
                self.sequences.append(numerical_features[start_index : end_index + 1])
                self.categorical_sequences.append(categorical_features[start_index : end_index + 1])
                self.class_targets.append(class_values[end_index])
                self.regression_targets.append(regression_values[end_index])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.sequences[index],
            self.categorical_sequences[index],
            self.class_targets[index],
            self.regression_targets[index],
        )


class TabularMatchDataset(Dataset):
    def __init__(
        self,
        numerical_dataframe: pd.DataFrame,
        categorical_dataframe: pd.DataFrame,
        numerical_feature_columns: list[str],
        categorical_feature_columns: list[str],
        sequence_length: int,
    ) -> None:
        del sequence_length
        num_df_reset = numerical_dataframe.reset_index(drop=True)
        cat_df_reset = categorical_dataframe[categorical_feature_columns].reset_index(drop=True)

        self.numerical_features = torch.tensor(
            num_df_reset[numerical_feature_columns].to_numpy(), dtype=torch.float32
        )
        self.categorical_features = torch.tensor(cat_df_reset.to_numpy(), dtype=torch.int64)
        self.class_targets = torch.tensor(
            num_df_reset[["resultado_empate", "resultado_vitoria_mandante", "resultado_vitoria_visitante"]].to_numpy(),
            dtype=torch.float32,
        )
        self.regression_targets = torch.tensor(
            num_df_reset[["gols_mandante", "gols_visitante"]].to_numpy(), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.numerical_features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.numerical_features[index],
            self.categorical_features[index],
            self.class_targets[index],
            self.regression_targets[index],
        )


class SiameseMatchDataset(Dataset):
    def __init__(
        self,
        numerical_dataframe: pd.DataFrame,
        categorical_dataframe: pd.DataFrame,
        numerical_feature_columns: list[str],
        categorical_feature_columns: list[str],
        sequence_length: int,
    ) -> None:
        self.sequence_length = sequence_length
        num_df_reset = numerical_dataframe.reset_index(drop=True)
        cat_df_reset = categorical_dataframe[categorical_feature_columns].reset_index(drop=True)
        combined_df = pd.concat([num_df_reset, cat_df_reset], axis=1)

        self.numerical_features = torch.tensor(
            combined_df[numerical_feature_columns].to_numpy(), dtype=torch.float32
        )
        self.categorical_features = torch.tensor(
            combined_df[categorical_feature_columns].to_numpy(), dtype=torch.int64
        )
        self.class_targets = torch.tensor(
            combined_df[["resultado_empate", "resultado_vitoria_mandante", "resultado_vitoria_visitante"]].to_numpy(),
            dtype=torch.float32,
        )
        self.regression_targets = torch.tensor(
            combined_df[["gols_mandante", "gols_visitante"]].to_numpy(), dtype=torch.float32
        )

        self.home_team_col_idx = categorical_feature_columns.index("time_mandante")
        self.away_team_col_idx = categorical_feature_columns.index("time_visitante")
        self.team_history: dict[int, list[int]] = {}

        for global_idx, row in cat_df_reset.iterrows():
            global_index = cast(int, global_idx)
            home_team_id = int(row["time_mandante"])
            away_team_id = int(row["time_visitante"])
            self.team_history.setdefault(home_team_id, []).append(global_index)
            self.team_history.setdefault(away_team_id, []).append(global_index)

    def _get_team_sequence(self, team_id: int, current_global_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        history = self.team_history.get(team_id, [])
        past_indices = [idx for idx in history if idx < current_global_idx]
        sequence_indices = past_indices[-self.sequence_length:]

        numerical_sequence = [self.numerical_features[idx] for idx in sequence_indices]
        categorical_sequence = [self.categorical_features[idx] for idx in sequence_indices]

        pad_len = self.sequence_length - len(sequence_indices)
        if pad_len > 0:
            numerical_padding = torch.zeros((pad_len, self.numerical_features.shape[1]), dtype=torch.float32)
            categorical_padding = torch.zeros((pad_len, self.categorical_features.shape[1]), dtype=torch.int64)
            if numerical_sequence:
                numerical_tensor = torch.cat([numerical_padding, torch.stack(numerical_sequence)], dim=0)
                categorical_tensor = torch.cat([categorical_padding, torch.stack(categorical_sequence)], dim=0)
            else:
                numerical_tensor = numerical_padding
                categorical_tensor = categorical_padding
        else:
            numerical_tensor = torch.stack(numerical_sequence)
            categorical_tensor = torch.stack(categorical_sequence)

        return numerical_tensor, categorical_tensor

    def __len__(self) -> int:
        return len(self.numerical_features)

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        match_num = self.numerical_features[index]
        match_cat = self.categorical_features[index]
        class_targets = self.class_targets[index]
        regression_targets = self.regression_targets[index]

        home_team_id = int(match_cat[self.home_team_col_idx].item())
        away_team_id = int(match_cat[self.away_team_col_idx].item())
        home_num, home_cat = self._get_team_sequence(home_team_id, index)
        away_num, away_cat = self._get_team_sequence(away_team_id, index)

        return ((home_num, home_cat, away_num, away_cat, match_num, match_cat), class_targets, regression_targets)


def load_match_dataframe(csv_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)

    return dataframe.sort_values([YEAR_COLUMN, "data", "rodada"]).reset_index(drop=True)


def scale_features(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame, validation_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    dist_normais_labels = ['saldo_gols_mandante', 'saldo_gols_visitante', 'idade_media_titular_mandante',
                        'idade_media_titular_visitante', 'colocacao_media_mandante', 'colocacao_media_visitante', 'elo_mandante', 'elo_visitante']

    dist_cauda_longa_labels = ['publico', 'publico_max', 'valor_equipe_titular_mandante',
                            'valor_equipe_titular_visitante']

    dist_poisson_labels = ['vitorias_mandante', 'vitorias_visitante', 'empates_mandante', 
                        'empates_visitante', 'derrotas_mandante', 'derrotas_visitante', 
                        'gols_pro_mandante', 'gols_pro_visitante', 'gols_sofridos_mandante',
                        'gols_sofridos_visitante', 'empates_confronto', 'vitorias_confronto_mandante', 
                        'vitorias_confronto_visitante',
                        'gols_marcados_media_mandante', 'gols_marcados_media_visitante',
                            'gols_sofridos_media_mandante', 'gols_sofridos_media_visitante',
                            'vitorias_seguidas_mandante', 'vitorias_seguidas_visitante',
                            'derrotas_seguidas_mandante', 'derrotas_seguidas_visitante',
                            'AvgCH', 'AvgCD', 'AvgCA']

    dist_categoricas_ciclicas_labels = ['data', 'rodada', 'colocacao_mandante', 'colocacao_visitante', 'dia', 'mes']

    # Initialize scalers
    standard_scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()

    # Apply StandardScaler to normally distributed columns
    for col in dist_normais_labels:
        if col in train_dataframe.columns and col in test_dataframe.columns and col in validation_dataframe.columns:
            train_dataframe[col] = standard_scaler.fit_transform(train_dataframe[[col]])
            test_dataframe[col] = standard_scaler.transform(test_dataframe[[col]])
            validation_dataframe[col] = standard_scaler.transform(validation_dataframe[[col]])

    # Apply log transformation and then MinMaxScaler to long-tail distributed columns
    for col in dist_cauda_longa_labels:
        if col in train_dataframe.columns and col in test_dataframe.columns and col in validation_dataframe.columns:
            train_dataframe[col] = np.log1p(train_dataframe[col])
            train_dataframe[col] = min_max_scaler.fit_transform(train_dataframe[[col]])

            test_dataframe[col] = np.log1p(test_dataframe[col])
            test_dataframe[col] = min_max_scaler.transform(test_dataframe[[col]])

            validation_dataframe[col] = np.log1p(validation_dataframe[col])
            validation_dataframe[col] = min_max_scaler.transform(validation_dataframe[[col]])

    # Combine Poisson and cyclical categorical labels
    dist_minmax_labels = dist_poisson_labels + dist_categoricas_ciclicas_labels

    # Apply MinMaxScaler to the combined list of columns
    for col in dist_minmax_labels:
        if col in test_dataframe.columns and col in train_dataframe.columns and col in validation_dataframe.columns:
            train_dataframe[col] = min_max_scaler.fit_transform(train_dataframe[[col]])
            test_dataframe[col] = min_max_scaler.transform(test_dataframe[[col]])
            validation_dataframe[col] = min_max_scaler.transform(validation_dataframe[[col]])

    return train_dataframe, test_dataframe, validation_dataframe

def split_match_dataframe(dataframe: pd.DataFrame) -> TemporalSplit:   
    minmaxscaler = MinMaxScaler()
    minmaxscaler.fit(dataframe[[YEAR_COLUMN]])

    train=dataframe[dataframe[YEAR_COLUMN] <= 2024].copy()
    test=dataframe[dataframe[YEAR_COLUMN].isin([2025])].copy()
    validation=dataframe[dataframe[YEAR_COLUMN].isin([2026])].copy()

    train[YEAR_COLUMN] = minmaxscaler.transform(train[[YEAR_COLUMN]])
    test[YEAR_COLUMN] = minmaxscaler.transform(test[[YEAR_COLUMN]])
    validation[YEAR_COLUMN] = minmaxscaler.transform(validation[[YEAR_COLUMN]])

    train, test, validation = scale_features(train, test, validation)

    return TemporalSplit(
        train=train,
        test=test,
        validation=validation,
    )


def _build_loader(
    numerical_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    numerical_feature_columns: list[str],
    categorical_feature_columns: list[str],
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    is_train: bool = True,
    arch: str = "legacy",
) -> DataLoader:

    if arch == "mlp":
        dataset = TabularMatchDataset(
            numerical_dataframe=numerical_df,
            categorical_dataframe=categorical_df,
            numerical_feature_columns=numerical_feature_columns,
            categorical_feature_columns=categorical_feature_columns,
            sequence_length=sequence_length,
        )
    elif arch in ["siamese", "hybrid"]:
        dataset = SiameseMatchDataset(
            numerical_dataframe=numerical_df,
            categorical_dataframe=categorical_df,
            numerical_feature_columns=numerical_feature_columns,
            categorical_feature_columns=categorical_feature_columns,
            sequence_length=sequence_length,
        )
    else:
        dataset = MatchSequenceDataset(
            numerical_dataframe=numerical_df,
            categorical_dataframe=categorical_df,
            numerical_feature_columns=numerical_feature_columns,
            categorical_feature_columns=categorical_feature_columns,
            sequence_length=sequence_length,
        )

    if is_train:
        # Pega os targets (assumindo que class_targets é One-Hot)
        if arch in ["siamese", "hybrid"]:
            targets = torch.stack([item[1] for item in dataset])
        else:
            targets = torch.stack([item[2] for item in dataset])
        class_indices = torch.argmax(targets, dim=1)
        
        # Calcula o peso de cada amostra (inverso da frequência da classe)
        class_counts = torch.bincount(class_indices)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[class_indices]
        
        # Cria o sampler (ele fará o shuffle internamente baseado nos pesos)
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(), 
            num_samples=len(sample_weights), 
            replacement=True
        )
        
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def build_sequence_bundle(
    csv_path: str | Path,
    sequence_length: int = 8,
    batch_size: int = 64,
    num_workers: int = 0,
    arch: str = "legacy",
) -> SequenceBundle:
    
    dataframe = load_match_dataframe(csv_path)
    dataframe = dataframe.sort_values([YEAR_COLUMN, "data", "rodada"]).reset_index(drop=True)
    dataframe = dataframe.drop(columns=FEATURES_TO_DROP)
    
    categorical_feature_columns = [
        "arbitro", "estadio", "tecnico_mandante", "tecnico_visitante",
        "time_mandante", "time_visitante", 'missing_colocacao_mandante',
        'missing_colocacao_visitante', 'missing_rodada'
    ]
    
    # 1. Create a separate dataframe for categorical features
    categorical_feature_columns_in_df = [col for col in categorical_feature_columns if col in dataframe.columns]
    categorical_feature_columns = categorical_feature_columns_in_df.copy()
    categorical_features_df = dataframe[categorical_feature_columns].copy()

    # 2. Calculate cardinalities from the separated categorical features dataframe
    cardinalities = {col: int(categorical_features_df[col].max() + 1) for col in categorical_feature_columns}
    embedding_dimensions = {
        col: (cardinalities[col], min(50, (cardinalities[col]) + 1 // 2))
        for col in categorical_feature_columns
    }
    embedding_settings = EmbeddingSettings(cardinalities=cardinalities, dimensions=embedding_dimensions)

    # 3. Define numerical features
    numerical_feature_columns = [
        col for col in FEATURE_COLUMNS if col not in categorical_feature_columns and col not in TARGET_COLUMNS
    ]

    numerical_feature_columns_in_df = [col for col in numerical_feature_columns if col in dataframe.columns]
    numerical_feature_columns = numerical_feature_columns_in_df.copy()
    
    # 4. The main dataframe for splitting should contain numerical features, targets, and sorting keys
    main_df_cols = numerical_feature_columns + list(TARGET_COLUMNS) + [YEAR_COLUMN]
    # Remove duplicates and ensure all columns exist in the dataframe
    main_df_cols = list(dict.fromkeys([col for col in main_df_cols if col in dataframe.columns]))
    main_df = dataframe[main_df_cols].copy()

    # 5. Perform the temporal split on the main (mostly numerical) data
    splits = split_match_dataframe(main_df)

    # 6. Use the indices from the splits to slice the categorical features
    train_categorical_features = categorical_features_df.loc[splits.train.index]
    test_categorical_features = categorical_features_df.loc[splits.test.index]
    validation_categorical_features = categorical_features_df.loc[splits.validation.index]

    # 7. Pass the correct dataframes to the loader builder
    train_loader = _build_loader(
        splits.train, train_categorical_features, numerical_feature_columns, categorical_feature_columns,
        sequence_length, batch_size, shuffle=True, num_workers=num_workers, is_train=True, arch=arch
    )
    test_loader = _build_loader(
        splits.test, test_categorical_features, numerical_feature_columns, categorical_feature_columns,
        sequence_length, batch_size, shuffle=False, num_workers=num_workers, arch=arch
    )
    validation_loader = _build_loader(
        splits.validation, validation_categorical_features, numerical_feature_columns, categorical_feature_columns,
        sequence_length, batch_size, shuffle=False, num_workers=num_workers, arch=arch
    )

    # Calculate the final input size for the models
    numerical_input_size = len(numerical_feature_columns)
    embedding_input_size = sum(dim for _, dim in embedding_dimensions.values())
    input_size = numerical_input_size + embedding_input_size

    return SequenceBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        validation_loader=validation_loader,
        numerical_feature_columns=numerical_feature_columns,
        categorical_feature_columns=categorical_feature_columns,
        input_size=input_size,
        embedding_settings=embedding_settings
)
