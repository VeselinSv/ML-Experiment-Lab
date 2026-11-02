"""Предварителна обработка на данни за ML експерименти."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_experiment_lab.datasets import LoadedDataset


class PreprocessingError(ValueError):
    """Грешка при подготовката на данните за ML експеримент."""


@dataclass(frozen=True)
class PreparedExperimentData:
    """Подготвени входни и целеви данни за ML експеримент."""

    X: np.ndarray
    y: np.ndarray
    preprocessor: ColumnTransformer
    feature_columns: tuple[str, ...]
    target_column: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    excluded_rows: int


def _to_number(value: str) -> float | None:
    """Опитва да преобразува текстова стойност във валидно число."""
    value = value.strip()

    if not value:
        return None

    try:
        number = float(value)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def _validate_selection(
    dataset: LoadedDataset,
    target_column: str,
    feature_columns: Sequence[str],
) -> tuple[str, ...]:
    """Проверява избора на целева и входни колони."""

    if target_column not in dataset.columns:
        raise PreprocessingError(
            f"Целевата колона „{target_column}“ не съществува."
        )

    if isinstance(feature_columns, (str, bytes)):
        raise PreprocessingError(
            "Входните колони трябва да бъдат подадени като списък."
        )

    selected_features = tuple(feature_columns)

    if not selected_features:
        raise PreprocessingError(
            "Трябва да изберете поне една входна колона."
        )

    if len(set(selected_features)) != len(selected_features):
        raise PreprocessingError(
            "Не можете да изберете една и съща входна колона повече от веднъж."
        )

    if target_column in selected_features:
        raise PreprocessingError(
            "Целевата колона не може да бъде и входна характеристика."
        )

    unknown_columns = [
        column
        for column in selected_features
        if column not in dataset.columns
    ]

    if unknown_columns:
        raise PreprocessingError(
            "Следните входни колони не съществуват: "
            + ", ".join(unknown_columns)
        )

    return selected_features


def _detect_feature_types(
    rows: tuple[tuple[str, ...], ...],
    feature_indexes: tuple[int, ...],
    feature_columns: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Разделя входните колони на числови и категориални."""

    numeric_columns = []
    categorical_columns = []

    for column_name, column_index in zip(
        feature_columns,
        feature_indexes,
    ):
        values = [
            row[column_index].strip()
            for row in rows
            if row[column_index].strip()
        ]

        is_numeric = bool(values) and all(
            _to_number(value) is not None
            for value in values
        )

        if is_numeric:
            numeric_columns.append(column_name)
        else:
            categorical_columns.append(column_name)

    return (
        tuple(numeric_columns),
        tuple(categorical_columns),
    )


def _build_preprocessor(
    numeric_indexes: tuple[int, ...],
    categorical_indexes: tuple[int, ...],
) -> ColumnTransformer:
    """Създава pipeline за числовите и категориалните колони."""

    transformers = []

    if numeric_indexes:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                list(numeric_indexes),
            )
        )

    if categorical_indexes:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="ЛИПСВАЩА_СТОЙНОСТ",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                list(categorical_indexes),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def prepare_experiment_data(
    dataset: LoadedDataset,
    target_column: str,
    feature_columns: Sequence[str],
) -> PreparedExperimentData:
    """Подготвя X, y и необучен трансформатор за експеримент.

    Числовите характеристики се преобразуват в числа.
    Категориалните характеристики остават текстови.
    Празните входни стойности се представят като NaN.

    Редовете с липсваща целева стойност се изключват.

    Важно:
        Трансформаторът не се обучава тук. След разделяне на
        данните трябва да се извика fit_transform() върху
        обучаващата извадка и transform() върху тестовата.
    """

    selected_features = _validate_selection(
        dataset,
        target_column,
        feature_columns,
    )

    target_index = dataset.columns.index(target_column)

    feature_indexes = tuple(
        dataset.columns.index(column)
        for column in selected_features
    )

    # Проверяваме дали всички редове имат правилния брой стойности.
    for row_number, row in enumerate(dataset.rows, start=1):
        if len(row) != dataset.column_count:
            raise PreprocessingError(
                f"Ред {row_number} има неправилен брой стойности."
            )

    # Изключваме редовете без целева стойност.
    valid_rows = []
    excluded_rows = 0

    for row in dataset.rows:
        target_value = row[target_index].strip()

        if not target_value:
            excluded_rows += 1
            continue

        valid_rows.append(row)

    if not valid_rows:
        raise PreprocessingError(
            "Няма редове с попълнена целева колона."
        )

    valid_rows_tuple = tuple(valid_rows)

    numeric_columns, categorical_columns = _detect_feature_types(
        valid_rows_tuple,
        feature_indexes,
        selected_features,
    )

    # Индексите тук са спрямо новата матрица X,
    # а не спрямо първоначалния набор от данни.
    numeric_indexes = tuple(
        index
        for index, column in enumerate(selected_features)
        if column in numeric_columns
    )

    categorical_indexes = tuple(
        index
        for index, column in enumerate(selected_features)
        if column in categorical_columns
    )

    feature_rows = []
    target_values = []

    for row in valid_rows_tuple:
        prepared_row = []

        for column, source_index in zip(
            selected_features,
            feature_indexes,
        ):
            value = row[source_index].strip()

            if column in numeric_columns:
                number = _to_number(value)

                prepared_row.append(
                    np.nan if number is None else number
                )
            else:
                prepared_row.append(
                    value if value else np.nan
                )

        feature_rows.append(prepared_row)
        target_values.append(row[target_index].strip())

    X = np.asarray(
        feature_rows,
        dtype=object,
    )

    y = np.asarray(
        target_values,
        dtype=object,
    )

    preprocessor = _build_preprocessor(
        numeric_indexes,
        categorical_indexes,
    )

    return PreparedExperimentData(
        X=X,
        y=y,
        preprocessor=preprocessor,
        feature_columns=selected_features,
        target_column=target_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        excluded_rows=excluded_rows,
    )