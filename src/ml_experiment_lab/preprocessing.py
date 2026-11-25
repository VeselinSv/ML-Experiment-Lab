"""Предварителна обработка на данни за ML експерименти."""

from __future__ import annotations

import math
import re
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


# Поддържани числови формати:
# 25, -25, 25.50, 25,50, .5, ,5, 1e3, 1,5e3.
NUMERIC_PATTERN = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*|,[0-9]+)?|[.,][0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?"
)


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

    mixed_columns: tuple[str, ...] = ()
    empty_columns: tuple[str, ...] = ()


def _to_number(value: str) -> float | None:
    """Преобразува текстова стойност във валидно число.

    Поддържа десетична точка, десетична запетая,
    отрицателни числа и научна нотация.
    """

    value = value.strip()

    if not value:
        return None

    if NUMERIC_PATTERN.fullmatch(value) is None:
        return None

    normalized_value = value.replace(",", ".")

    try:
        number = float(normalized_value)
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
            "Не можете да изберете една и съща входна колона "
            "повече от веднъж."
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
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Разпознава числови, категориални, смесени и празни колони."""

    numeric_columns = []
    categorical_columns = []
    mixed_columns = []
    empty_columns = []

    for column_name, column_index in zip(
        feature_columns,
        feature_indexes,
    ):
        values = [
            row[column_index].strip()
            for row in rows
            if row[column_index].strip()
        ]

        # Изцяло празна колона.
        if not values:
            empty_columns.append(column_name)
            categorical_columns.append(column_name)
            continue

        numeric_count = sum(
            _to_number(value) is not None
            for value in values
        )

        # Всички непразни стойности са числа.
        if numeric_count == len(values):
            numeric_columns.append(column_name)

        # Нито една стойност не е число.
        elif numeric_count == 0:
            categorical_columns.append(column_name)

        # Колоната съдържа едновременно числа и текст.
        else:
            mixed_columns.append(column_name)

            # Засега запазваме стойностите като категории,
            # за да не губим информация автоматично.
            categorical_columns.append(column_name)

    return (
        tuple(numeric_columns),
        tuple(categorical_columns),
        tuple(mixed_columns),
        tuple(empty_columns),
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

    Трансформаторът не се обучава тук.
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

    # Проверяваме структурата на всички редове.
    for row_number, row in enumerate(dataset.rows, start=1):
        if len(row) != dataset.column_count:
            raise PreprocessingError(
                f"Ред {row_number} има неправилен брой стойности."
            )

    # Изключваме редовете с липсваща целева стойност.
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

    (
        numeric_columns,
        categorical_columns,
        mixed_columns,
        empty_columns,
    ) = _detect_feature_types(
        valid_rows_tuple,
        feature_indexes,
        selected_features,
    )

    numeric_column_set = set(numeric_columns)
    categorical_column_set = set(categorical_columns)

    # Индекси спрямо новата матрица X.
    numeric_indexes = tuple(
        index
        for index, column in enumerate(selected_features)
        if column in numeric_column_set
    )

    categorical_indexes = tuple(
        index
        for index, column in enumerate(selected_features)
        if column in categorical_column_set
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

            if column in numeric_column_set:
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
        mixed_columns=mixed_columns,
        empty_columns=empty_columns,
    )