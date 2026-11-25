"""Предварителна обработка на данни за ML експерименти."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from ml_experiment_lab.datasets import LoadedDataset

from ml_experiment_lab.datetime_features import (
    DATETIME_FEATURE_NAMES,
    extract_datetime_features,
    is_datetime_value,
)


class PreprocessingError(ValueError):
    """Грешка при подготовката на данните за ML експеримент."""


NumericImputationStrategy = Literal[
    "mean",
    "median",
    "most_frequent",
    "constant",
]

CategoricalImputationStrategy = Literal[
    "most_frequent",
    "constant",
]


NUMERIC_IMPUTATION_STRATEGIES = {
    "mean": "Средноаритметична стойност",
    "median": "Медиана",
    "most_frequent": "Най-често срещана стойност",
    "constant": "Константна стойност",
}

CATEGORICAL_IMPUTATION_STRATEGIES = {
    "most_frequent": "Най-често срещана категория",
    "constant": "Константна стойност",
}


# Поддържани числови формати:
# 25, -25, 25.50, 25,50, .5, ,5, 1e3, 1,5e3.
NUMERIC_PATTERN = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*|,[0-9]+)?|[.,][0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?"
)


@dataclass(frozen=True)
class PreprocessingConfiguration:
    """Настройки за предварителна обработка на ML данни."""

    numeric_imputation_strategy: NumericImputationStrategy = "median"

    categorical_imputation_strategy: (
        CategoricalImputationStrategy
    ) = "constant"

    numeric_fill_value: float = 0.0

    categorical_fill_value: str = "ЛИПСВАЩА_СТОЙНОСТ"

    def __post_init__(self) -> None:
        """Проверява валидността на настройките."""

        if (
            self.numeric_imputation_strategy
            not in NUMERIC_IMPUTATION_STRATEGIES
        ):
            raise PreprocessingError(
                "Неподдържана стратегия за обработка "
                "на липсващи числови стойности."
            )

        if (
            self.categorical_imputation_strategy
            not in CATEGORICAL_IMPUTATION_STRATEGIES
        ):
            raise PreprocessingError(
                "Неподдържана стратегия за обработка "
                "на липсващи категориални стойности."
            )

        if (
            isinstance(self.numeric_fill_value, bool)
            or not isinstance(
                self.numeric_fill_value,
                (int, float),
            )
        ):
            raise PreprocessingError(
                "Числовата константа трябва да бъде число."
            )

        try:
            is_finite = math.isfinite(
                self.numeric_fill_value
            )
        except OverflowError:
            is_finite = False

        if not is_finite:
            raise PreprocessingError(
                "Числовата константа трябва да бъде "
                "крайна числова стойност."
            )

        if (
            not isinstance(
                self.categorical_fill_value,
                str,
            )
            or not self.categorical_fill_value.strip()
        ):
            raise PreprocessingError(
                "Категориалната константа трябва "
                "да бъде непразна текстова стойност."
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

    preprocessing_configuration: PreprocessingConfiguration = field(
        default_factory=PreprocessingConfiguration
    )

    datetime_columns: tuple[str, ...] = ()


def _to_number(value: str) -> float | None:
    """Преобразува текстова стойност във валидно число."""

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


def parse_numeric_value(value: str) -> float | None:
    """Преобразува стойност чрез общия числов parser."""

    return _to_number(value)


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
    tuple[str, ...],
]:
    """Разпознава числови, категориални и датни колони.

    Допълнително открива смесени и изцяло празни колони.
    """

    numeric_columns = []
    categorical_columns = []
    datetime_columns = []
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

        datetime_count = sum(
            is_datetime_value(value)
            for value in values
        )

        # Всички непразни стойности са числа.
        if numeric_count == len(values):
            numeric_columns.append(column_name)

        # Всички непразни стойности са валидни дати.
        elif datetime_count == len(values):
            datetime_columns.append(column_name)

        # Колоната съдържа само обикновен текст.
        elif numeric_count == 0 and datetime_count == 0:
            categorical_columns.append(column_name)

        # Колоната съдържа смесени типове.
        else:
            mixed_columns.append(column_name)

            # Запазваме смесените колони като категории,
            # за да не губим информация автоматично.
            categorical_columns.append(column_name)

    return (
        tuple(numeric_columns),
        tuple(categorical_columns),
        tuple(mixed_columns),
        tuple(empty_columns),
        tuple(datetime_columns),
    )


def _detect_training_feature_types(
    X: np.ndarray,
    feature_columns: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Разпознава типовете от необработени тренировъчни данни."""

    rows = tuple(
        tuple(
            value.strip() if isinstance(value, str) else ""
            for value in row
        )
        for row in X
    )

    feature_indexes = tuple(
        range(len(feature_columns))
    )

    return _detect_feature_types(
        rows,
        feature_indexes,
        feature_columns,
    )


def _select_training_numeric_indexes(
    X: np.ndarray,
) -> list[int]:
    """Определя числовите колони при обучението."""

    feature_columns = tuple(
        str(index)
        for index in range(X.shape[1])
    )

    numeric_columns, _, _, _, _ = (
        _detect_training_feature_types(
            X,
            feature_columns,
        )
    )

    return [
        int(column)
        for column in numeric_columns
    ]


def _select_training_categorical_indexes(
    X: np.ndarray,
) -> list[int]:
    """Определя категориалните колони при обучението."""

    feature_columns = tuple(
        str(index)
        for index in range(X.shape[1])
    )

    _, categorical_columns, _, _, _ = (
        _detect_training_feature_types(
            X,
            feature_columns,
        )
    )

    return [
        int(column)
        for column in categorical_columns
    ]


def _select_training_datetime_indexes(
    X: np.ndarray,
) -> list[int]:
    """Определя датните колони при обучението."""

    feature_columns = tuple(
        str(index)
        for index in range(X.shape[1])
    )

    _, _, _, _, datetime_columns = (
        _detect_training_feature_types(
            X,
            feature_columns,
        )
    )

    return [
        int(column)
        for column in datetime_columns
    ]


def _select_categorical_by_empty(
    X: np.ndarray,
    *,
    fixed_indexes: tuple[int, ...] | None,
    empty: bool,
) -> list[int]:
    """Разделя категориалните колони на празни и непразни."""

    if fixed_indexes is None:
        categorical_indexes = (
            _select_training_categorical_indexes(X)
        )
    else:
        categorical_indexes = list(fixed_indexes)

    selected_indexes = []

    for index in categorical_indexes:
        has_observed_values = any(
            isinstance(value, str) and bool(value.strip())
            for value in X[:, index]
        )

        is_empty = not has_observed_values

        if is_empty == empty:
            selected_indexes.append(index)

    return selected_indexes


def _convert_numeric_features(
    X: np.ndarray,
) -> np.ndarray:
    """Преобразува входните стойности в числа."""

    values = np.asarray(X, dtype=object)

    converted = np.full(
        values.shape,
        np.nan,
        dtype=float,
    )

    for index, value in np.ndenumerate(values):
        if not isinstance(value, str):
            continue

        number = _to_number(value)

        if number is not None:
            converted[index] = number

    return converted


def _convert_datetime_features(
    X: np.ndarray,
) -> np.ndarray:
    """Преобразува датните колони в числови характеристики.

    Всяка дата се разширява до девет характеристики.
    Липсващите и невалидните дати се преобразуват в NaN.
    """

    values = np.asarray(
        X,
        dtype=object,
    )

    feature_count = len(
        DATETIME_FEATURE_NAMES
    )

    converted = np.full(
        (
            values.shape[0],
            values.shape[1] * feature_count,
        ),
        np.nan,
        dtype=float,
    )

    for row_index in range(values.shape[0]):

        for column_index in range(values.shape[1]):

            value = values[
                row_index,
                column_index,
            ]

            if not isinstance(value, str):
                continue

            datetime_features = extract_datetime_features(
                value
            )

            start_index = (
                column_index * feature_count
            )

            end_index = (
                start_index + feature_count
            )

            converted[
                row_index,
                start_index:end_index,
            ] = datetime_features

    return converted


def _get_datetime_feature_names(
    transformer: FunctionTransformer,
    input_features: Sequence[str],
) -> np.ndarray:
    """Създава имена за извлечените датни характеристики."""

    del transformer

    return np.asarray(
        [
            f"{column_name}_{feature_name}"
            for column_name in input_features
            for feature_name in DATETIME_FEATURE_NAMES
        ],
        dtype=object,
    )


def with_training_feature_types(
    prepared_data: PreparedExperimentData,
    X_train: np.ndarray,
) -> PreparedExperimentData:
    """Записва типовете, разпознати от Training данните."""

    (
        numeric_columns,
        categorical_columns,
        mixed_columns,
        empty_columns,
        datetime_columns,
    ) = _detect_training_feature_types(
        X_train,
        prepared_data.feature_columns,
    )

    return replace(
        prepared_data,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        mixed_columns=mixed_columns,
        empty_columns=empty_columns,
        datetime_columns=datetime_columns,
    )


def _build_numeric_imputer(
    configuration: PreprocessingConfiguration,
) -> SimpleImputer:
    """Създава конфигурируем imputer за числа.

    Използва се както за обикновените числови колони,
    така и за извлечените датни характеристики.
    """

    fill_value = (
        float(configuration.numeric_fill_value)
        if configuration.numeric_imputation_strategy == "constant"
        else None
    )

    return SimpleImputer(
        strategy=configuration.numeric_imputation_strategy,
        fill_value=fill_value,
        keep_empty_features=True,
    )


def _build_numeric_pipeline(
    configuration: PreprocessingConfiguration,
    defer_type_detection: bool,
) -> Pipeline:
    """Създава pipeline за числовите характеристики."""

    numeric_steps = []

    if defer_type_detection:
        numeric_steps.append(
            (
                "parser",
                FunctionTransformer(
                    _convert_numeric_features,
                    feature_names_out="one-to-one",
                ),
            )
        )

    numeric_steps.extend(
        [
            (
                "imputer",
                _build_numeric_imputer(configuration),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return Pipeline(
        steps=numeric_steps
    )


def _build_categorical_pipeline(
    configuration: PreprocessingConfiguration,
    *,
    force_constant: bool = False,
) -> Pipeline:
    """Създава pipeline за категориалните характеристики."""

    strategy = (
        "constant"
        if force_constant
        else configuration.categorical_imputation_strategy
    )

    fill_value = (
        configuration.categorical_fill_value
        if strategy == "constant"
        else None
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=strategy,
                    fill_value=fill_value,
                    keep_empty_features=True,
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


def _build_datetime_pipeline(
    configuration: PreprocessingConfiguration,
) -> Pipeline:
    """Създава pipeline за датните характеристики."""

    return Pipeline(
        steps=[
            (
                "extractor",
                FunctionTransformer(
                    _convert_datetime_features,
                    feature_names_out=(
                        _get_datetime_feature_names
                    ),
                ),
            ),
            (
                "imputer",
                _build_numeric_imputer(
                    configuration
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


def _build_preprocessor(
    numeric_indexes: tuple[int, ...] = (),
    categorical_indexes: tuple[int, ...] = (),
    *,
    datetime_indexes: tuple[int, ...] = (),
    defer_type_detection: bool = False,
    configuration: PreprocessingConfiguration | None = None,
) -> ColumnTransformer:
    """Създава конфигурируем pipeline за обработка.

    Поддържа числови, категориални и датни колони.

    При отложено разпознаване типовете се определят
    при fit() само от тренировъчните данни.
    """

    if configuration is None:
        configuration = PreprocessingConfiguration()

    transformers = []

    # Обработка на числовите характеристики.
    if numeric_indexes or defer_type_detection:

        numeric_pipeline = _build_numeric_pipeline(
            configuration=configuration,
            defer_type_detection=defer_type_detection,
        )

        numeric_selector = (
            _select_training_numeric_indexes
            if defer_type_detection
            else list(numeric_indexes)
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_selector,
            )
        )

    # Обработка на категориалните характеристики.
    if categorical_indexes or defer_type_detection:

        if (
            configuration.categorical_imputation_strategy
            == "most_frequent"
        ):
            # Изцяло празна колона няма най-честа категория.
            # Затова използваме отделен pipeline с константа.

            fixed_indexes = (
                None
                if defer_type_detection
                else categorical_indexes
            )

            nonempty_selector = partial(
                _select_categorical_by_empty,
                fixed_indexes=fixed_indexes,
                empty=False,
            )

            empty_selector = partial(
                _select_categorical_by_empty,
                fixed_indexes=fixed_indexes,
                empty=True,
            )

            transformers.append(
                (
                    "categorical",
                    _build_categorical_pipeline(
                        configuration=configuration,
                    ),
                    nonempty_selector,
                )
            )

            transformers.append(
                (
                    "categorical_empty",
                    _build_categorical_pipeline(
                        configuration=configuration,
                        force_constant=True,
                    ),
                    empty_selector,
                )
            )

        else:

            categorical_selector = (
                _select_training_categorical_indexes
                if defer_type_detection
                else list(categorical_indexes)
            )

            transformers.append(
                (
                    "categorical",
                    _build_categorical_pipeline(
                        configuration=configuration,
                    ),
                    categorical_selector,
                )
            )

    # Обработка на датните характеристики.
    if datetime_indexes or defer_type_detection:

        datetime_pipeline = _build_datetime_pipeline(
            configuration=configuration,
        )

        datetime_selector = (
            _select_training_datetime_indexes
            if defer_type_detection
            else list(datetime_indexes)
        )

        transformers.append(
            (
                "datetime",
                datetime_pipeline,
                datetime_selector,
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
    *,
    defer_type_detection: bool = False,
    preprocessing_configuration: (
        PreprocessingConfiguration | None
    ) = None,
) -> PreparedExperimentData:
    """Подготвя X, y и необучен трансформатор.

    Поддържа числови, категориални и датни колони.

    При defer_type_detection=True типовете се разпознават
    след разделянето на Training и Test.

    Настройките за предварителна обработка могат
    да бъдат зададени чрез PreprocessingConfiguration.

    Редовете с липсваща целева стойност се изключват.

    Трансформаторът не се обучава тук.
    """

    if preprocessing_configuration is None:
        preprocessing_configuration = (
            PreprocessingConfiguration()
        )

    if not isinstance(
        preprocessing_configuration,
        PreprocessingConfiguration,
    ):
        raise PreprocessingError(
            "Невалидна конфигурация за предварителна обработка."
        )

    selected_features = _validate_selection(
        dataset,
        target_column,
        feature_columns,
    )

    target_index = dataset.columns.index(
        target_column
    )

    feature_indexes = tuple(
        dataset.columns.index(column)
        for column in selected_features
    )

    # Проверяваме структурата на всички редове.
    for row_number, row in enumerate(
        dataset.rows,
        start=1,
    ):
        if len(row) != dataset.column_count:
            raise PreprocessingError(
                f"Ред {row_number} има неправилен брой стойности."
            )

    # Изключваме редовете с липсващ Target.
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

    # При реално обучение типовете ще бъдат определени
    # след разделяне на тренировъчна и тестова извадка.
    if defer_type_detection:

        numeric_columns = ()
        categorical_columns = ()
        mixed_columns = ()
        empty_columns = ()
        datetime_columns = ()

    else:

        (
            numeric_columns,
            categorical_columns,
            mixed_columns,
            empty_columns,
            datetime_columns,
        ) = _detect_feature_types(
            valid_rows_tuple,
            feature_indexes,
            selected_features,
        )

    numeric_column_set = set(
        numeric_columns
    )

    categorical_column_set = set(
        categorical_columns
    )

    datetime_column_set = set(
        datetime_columns
    )

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

    datetime_indexes = tuple(
        index
        for index, column in enumerate(selected_features)
        if column in datetime_column_set
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

            if (
                not defer_type_detection
                and column in numeric_column_set
            ):
                number = _to_number(value)

                prepared_row.append(
                    np.nan if number is None else number
                )

            else:
                # Категориалните и датните стойности
                # се запазват като текст до трансформацията.
                prepared_row.append(
                    value if value else np.nan
                )

        feature_rows.append(
            prepared_row
        )

        target_values.append(
            row[target_index].strip()
        )

    X = np.asarray(
        feature_rows,
        dtype=object,
    )

    y = np.asarray(
        target_values,
        dtype=object,
    )

    preprocessor = _build_preprocessor(
        numeric_indexes=numeric_indexes,
        categorical_indexes=categorical_indexes,
        datetime_indexes=datetime_indexes,
        defer_type_detection=defer_type_detection,
        configuration=preprocessing_configuration,
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
        datetime_columns=datetime_columns,
        preprocessing_configuration=(
            preprocessing_configuration
        ),
    )