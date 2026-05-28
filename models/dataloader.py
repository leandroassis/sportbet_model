from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset


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
    'derrotas_seguidas_visitante' 
]


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
        self.class_targets: list[int] = []
        self.regression_targets: list[torch.Tensor] = []

        # Combine the dataframes for sorting, keeping the original index
        num_df_reset = numerical_dataframe.reset_index(drop=True)
        cat_df_reset = categorical_dataframe[categorical_feature_columns].reset_index(drop=True)
        combined_df = pd.concat([num_df_reset, cat_df_reset], axis=1)
        ordered = combined_df.sort_values([YEAR_COLUMN, "data", "rodada"])

        for _, season_frame in ordered.groupby(YEAR_COLUMN, sort=True):
            season_frame = season_frame.sort_values(["data", "rodada"]).reset_index(drop=True)
            if len(season_frame) < sequence_length:
                continue

            numerical_features = torch.tensor(season_frame[numerical_feature_columns].to_numpy(), dtype=torch.float32)
            categorical_features = torch.tensor(season_frame[categorical_feature_columns].to_numpy(), dtype=torch.int64)

            class_values = season_frame[['resultado_empate', 'resultado_vitoria_mandante', 'resultado_vitoria_visitante']].astype(int).to_numpy()
            regression_values = torch.tensor(season_frame[["gols_mandante", "gols_visitante"]].to_numpy(), dtype=torch.float32)

            for end_index in range(sequence_length - 1, len(season_frame)):
                start_index = end_index - sequence_length + 1
                self.sequences.append(numerical_features[start_index : end_index + 1])
                self.categorical_sequences.append(categorical_features[start_index : end_index + 1])
                self.class_targets.append(int(class_values[end_index]))
                self.regression_targets.append(regression_values[end_index])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        return (
            self.sequences[index],
            self.categorical_sequences[index],
            self.class_targets[index],
            self.regression_targets[index],
        )


def load_match_dataframe(csv_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)

    return dataframe.sort_values([YEAR_COLUMN, "data", "rodada"]).reset_index(drop=True)


def split_match_dataframe(dataframe: pd.DataFrame) -> TemporalSplit:
    
    minmaxscaler = MinMaxScaler()
    minmaxscaler.fit(dataframe[[YEAR_COLUMN]])

    train=dataframe[dataframe[YEAR_COLUMN] <= 2022].copy()
    test=dataframe[dataframe[YEAR_COLUMN].isin([2023, 2024])].copy()
    validation=dataframe[dataframe[YEAR_COLUMN].isin([2025, 2026])].copy()

    train[YEAR_COLUMN] = minmaxscaler.transform(train[[YEAR_COLUMN]])
    test[YEAR_COLUMN] = minmaxscaler.transform(test[[YEAR_COLUMN]])
    validation[YEAR_COLUMN] = minmaxscaler.transform(validation[[YEAR_COLUMN]])

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
) -> DataLoader:
    dataset = MatchSequenceDataset(
        numerical_dataframe=numerical_df,
        categorical_dataframe=categorical_df,
        numerical_feature_columns=numerical_feature_columns,
        categorical_feature_columns=categorical_feature_columns,
        sequence_length=sequence_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_sequence_bundle(
    csv_path: str | Path,
    sequence_length: int = 8,
    batch_size: int = 64,
    num_workers: int = 0,
) -> SequenceBundle:
    
    dataframe = load_match_dataframe(csv_path)
    
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
        col: (cardinalities[col], min(50, (cardinalities[col]) // 2))
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
        splits.train, train_categorical_features, numerical_feature_columns, categorical_feature_columns, sequence_length, batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = _build_loader(
        splits.test, test_categorical_features, numerical_feature_columns, categorical_feature_columns, sequence_length, batch_size, shuffle=False, num_workers=num_workers
    )
    validation_loader = _build_loader(
        splits.validation, validation_categorical_features, numerical_feature_columns, categorical_feature_columns, sequence_length, batch_size, shuffle=False, num_workers=num_workers
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