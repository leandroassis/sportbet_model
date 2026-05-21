from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset


TARGET_COLUMNS = ("resultado_partida", "gols_mandante", "gols_visitante")
CLASS_LABEL_MAP = {-1: 0, 0: 1, 1: 2}
YEAR_COLUMN = "ano_campeonato"
FEATURE_COLUMNS = [
    "data",
    "rodada",
    "estadio",
    "arbitro",
    "publico",
    "publico_max",
    "time_mandante",
    "time_visitante",
    "tecnico_mandante",
    "tecnico_visitante",
    "colocacao_mandante",
    "colocacao_visitante",
    "valor_equipe_titular_mandante",
    "valor_equipe_titular_visitante",
    "idade_media_titular_mandante",
    "idade_media_titular_visitante",
    "AvgCH",
    "AvgCD",
    "AvgCA",
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
]


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
    feature_columns: list[str]
    input_size: int
    class_mapping: dict[int, int]


class MatchSequenceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, feature_columns: list[str], sequence_length: int) -> None:
        self.sequences: list[torch.Tensor] = []
        self.class_targets: list[int] = []
        self.regression_targets: list[torch.Tensor] = []

        ordered = dataframe.sort_values([YEAR_COLUMN, "data", "rodada"])
        for _, season_frame in ordered.groupby(YEAR_COLUMN, sort=True):
            season_frame = season_frame.sort_values(["data", "rodada"]).reset_index(drop=True)
            if len(season_frame) < sequence_length:
                continue

            features = torch.tensor(season_frame[feature_columns].to_numpy(), dtype=torch.float32)
            class_values = season_frame["resultado_partida"].astype(int).map(CLASS_LABEL_MAP).to_numpy()
            regression_values = torch.tensor(season_frame[["gols_mandante", "gols_visitante"]].to_numpy(), dtype=torch.float32)

            for end_index in range(sequence_length - 1, len(season_frame)):
                start_index = end_index - sequence_length + 1
                self.sequences.append(features[start_index : end_index + 1])
                self.class_targets.append(int(class_values[end_index]))
                self.regression_targets.append(regression_values[end_index])

    def __len__(self) -> int:
        return len(self.class_targets)

    def __getitem__(self, index: int):
        return (
            self.sequences[index],
            torch.tensor(self.class_targets[index], dtype=torch.long),
            self.regression_targets[index],
        )


def load_match_dataframe(csv_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)

    return dataframe.sort_values([YEAR_COLUMN, "data", "rodada"]).reset_index(drop=True)


def split_match_dataframe(dataframe: pd.DataFrame) -> TemporalSplit:
    
    minmaxscaler = MinMaxScaler()
    minmaxscaler.fit_transform(dataframe[[YEAR_COLUMN]])

    train=dataframe[dataframe[YEAR_COLUMN] <= 2023].copy()
    test=dataframe[dataframe[YEAR_COLUMN].isin([2024, 2025])].copy()
    validation=dataframe[dataframe[YEAR_COLUMN] == 2026].copy()

    train[YEAR_COLUMN] = minmaxscaler.transform(train[[YEAR_COLUMN]])
    test[YEAR_COLUMN] = minmaxscaler.transform(test[[YEAR_COLUMN]])
    validation[YEAR_COLUMN] = minmaxscaler.transform(validation[[YEAR_COLUMN]])

    return TemporalSplit(
        train=train,
        test=test,
        validation=validation,
    )


def _build_loader(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = MatchSequenceDataset(dataframe=dataframe, feature_columns=feature_columns, sequence_length=sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def build_sequence_bundle(
    csv_path: str | Path,
    sequence_length: int = 8,
    batch_size: int = 64,
    num_workers: int = 0,
) -> SequenceBundle:
    
    dataframe = load_match_dataframe(csv_path)
    splits = split_match_dataframe(dataframe)
    feature_columns = [column for column in FEATURE_COLUMNS if column in dataframe.columns]

    train_loader = _build_loader(splits.train, feature_columns, sequence_length, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = _build_loader(splits.test, feature_columns, sequence_length, batch_size, shuffle=False, num_workers=num_workers)
    validation_loader = _build_loader(splits.validation, feature_columns, sequence_length, batch_size, shuffle=False, num_workers=num_workers)

    return SequenceBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        validation_loader=validation_loader,
        feature_columns=feature_columns,
        input_size=len(feature_columns),
        class_mapping=CLASS_LABEL_MAP.copy(),
    )