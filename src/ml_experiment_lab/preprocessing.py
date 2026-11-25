"""Предварителна обработка на данни за ML експерименти."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, Sequence

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from sklearn.utils.validation import check_is_fitted

from ml_experiment_lab.datasets import LoadedDataset

from ml_experiment_lab.datetime_features import (
    DATETIME_FEATURE_NAMES,
    extract_datetime_features,
    is_datetime_value,
)

from ml_experiment_lab.encoding import FrequencyEncoder


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

ScalingMethod = Literal[
    "standard",
    "minmax",
    "robust",
    "maxabs",
    "none",
]

EncodingMethod = Literal[
    "onehot",
    "ordinal",
    "frequency",
]

FeatureType = Literal[
    "numeric",
    "categorical",
    "datetime",
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

SCALING_METHODS = {
    "standard": "Стандартизиране (StandardScaler)",
    "minmax": "Нормализиране (MinMaxScaler)",
    "robust": "Устойчиво мащабиране (RobustScaler)",
    "maxabs": "Максимална абсолютна стойност (MaxAbsScaler)",
    "none": "Без мащабиране",
}

ENCODING_METHODS = {
    "onehot": "Бинарно кодиране (One-Hot Encoding)",
    "ordinal": "Поредно кодиране (Ordinal Encoding)",
    "frequency": "Честотно кодиране (Frequency Encoding)",
}


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

    drop_empty_columns: bool = False

    drop_constant_features: bool = False

    scaling_method: ScalingMethod = "standard"

    encoding_method: EncodingMethod = "onehot"

    def __post_init__(self) -> None:
        """Проверява валидността на настройките."""

        if (
            not isinstance(
                self.numeric_imputation_strategy,
                str,
            )
            or self.numeric_imputation_strategy
            not in NUMERIC_IMPUTATION_STRATEGIES
        ):
            raise PreprocessingError(
                "Неподдържана стратегия за обработка "
                "на липсващи числови стойности."
            )

        if (
            not isinstance(
                self.categorical_imputation_strategy,
                str,
            )
            or self.categorical_imputation_strategy
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

        if not isinstance(
            self.drop_empty_columns,
            bool,
        ):
            raise PreprocessingError(
                "Настройката за премахване на празни "
                "колони трябва да бъде булева стойност."
            )

        if not isinstance(
            self.drop_constant_features,
            bool,
        ):
            raise PreprocessingError(
                "Настройката за премахване на постоянни "
                "характеристики трябва да бъде булева стойност."
            )

        if (
            not isinstance(self.scaling_method, str)
            or self.scaling_method not in SCALING_METHODS
        ):
            raise PreprocessingError(
                "Неподдържан метод за мащабиране "
                "на числовите характеристики."
            )

        if (
            not isinstance(self.encoding_method, str)
            or self.encoding_method not in ENCODING_METHODS
        ):
            raise PreprocessingError(
                "Неподдържан метод за кодиране "
                "на категориалните характеристики."
            )


@dataclass(frozen=True)
class PreparedExperimentData:
    """Подготвени входни и целеви данни за ML експеримент."""

    X: np.ndarray
    y: np.ndarray

    preprocessor: ColumnTransformer | Pipeline

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


@dataclass(frozen=True)
class FeatureSelectionReport:
    """Информация за премахнатите характеристики."""

    removed_empty_columns: tuple[str, ...] = ()

    removed_constant_features: tuple[str, ...] = ()

    @property
    def removed_empty_count(self) -> int:
        """Връща броя на премахнатите празни колони."""

        return len(self.removed_empty_columns)

    @property
    def removed_constant_count(self) -> int:
        """Връща броя на премахнатите постоянни характеристики."""

        return len(self.removed_constant_features)


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

        if numeric_count == len(values):
            numeric_columns.append(column_name)

        elif datetime_count == len(values):
            datetime_columns.append(column_name)

        elif numeric_count == 0 and datetime_count == 0:
            categorical_columns.append(column_name)

        else:
            mixed_columns.append(column_name)
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


def _has_observed_values(
    X: np.ndarray,
    column_index: int,
) -> bool:
    """Проверява дали колоната има поне една непразна стойност."""

    for value in X[:, column_index]:

        if isinstance(value, str):
            if value.strip():
                return True

        elif isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            if not np.isnan(value):
                return True

    return False


def _select_feature_indexes(
    X: np.ndarray,
    *,
    feature_type: FeatureType,
    fixed_indexes: tuple[int, ...] | None,
    drop_empty_columns: bool,
    empty: bool | None = None,
) -> list[int]:
    """Избира колони според типа и настройките за филтриране."""

    # При отложено разпознаване определяме
    # типовете само от подадените Training данни.
    if fixed_indexes is None:

        if feature_type == "numeric":
            indexes = _select_training_numeric_indexes(X)

        elif feature_type == "categorical":
            indexes = _select_training_categorical_indexes(X)

        else:
            indexes = _select_training_datetime_indexes(X)

    else:
        indexes = list(fixed_indexes)

    # Определяме кои входни колони са изцяло празни.
    if drop_empty_columns or empty is not None:

        observed_indexes = {
            index
            for index in range(X.shape[1])
            if _has_observed_values(X, index)
        }

        if drop_empty_columns:

            if not observed_indexes:
                raise PreprocessingError(
                    "След премахването на празните колони "
                    "не остава нито една входна характеристика."
                )

            indexes = [
                index
                for index in indexes
                if index in observed_indexes
            ]

        # Разделяме категориалните колони
        # при Most Frequent Imputation.
        if empty is not None:

            indexes = [
                index
                for index in indexes
                if (index not in observed_indexes) == empty
            ]

    return indexes


def _convert_numeric_features(
    X: np.ndarray,
) -> np.ndarray:
    """Преобразува входните стойности в числа."""

    values = np.asarray(
        X,
        dtype=object,
    )

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
    """Преобразува датните колони в числови характеристики."""

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
    """Създава конфигурируем imputer за числа."""

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


def _build_scaler(
    configuration: PreprocessingConfiguration,
) -> (
    StandardScaler
    | MinMaxScaler
    | RobustScaler
    | MaxAbsScaler
    | Literal["passthrough"]
):
    """Създава избрания метод за мащабиране."""

    method = configuration.scaling_method

    if method == "standard":
        return StandardScaler()

    if method == "minmax":
        return MinMaxScaler(
            feature_range=(0, 1),
            clip=False,
        )

    if method == "robust":
        return RobustScaler(
            quantile_range=(25.0, 75.0),
        )

    if method == "maxabs":
        return MaxAbsScaler()

    if method == "none":
        return "passthrough"

    raise PreprocessingError(
        f"Неподдържан метод за мащабиране: „{method}“."
    )


def _build_encoder(
    configuration: PreprocessingConfiguration,
) -> (
    OneHotEncoder
    | OrdinalEncoder
    | FrequencyEncoder
):
    """Създава избрания метод за категориално кодиране."""

    method = configuration.encoding_method

    if method == "onehot":

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    if method == "ordinal":

        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )

    if method == "frequency":

        return FrequencyEncoder()

    raise PreprocessingError(
        f"Неподдържан метод за кодиране: „{method}“."
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
                _build_scaler(configuration),
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
                _build_encoder(configuration),
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
                _build_numeric_imputer(configuration),
            ),
            (
                "scaler",
                _build_scaler(configuration),
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
) -> ColumnTransformer | Pipeline:
    """Създава конфигурируем трансформатор за ML данни."""

    if configuration is None:
        configuration = PreprocessingConfiguration()

    transformers = []

    def make_selector(
        feature_type: FeatureType,
        indexes: tuple[int, ...],
        *,
        empty: bool | None = None,
    ):
        """Създава селектор за дадена група характеристики."""

        if (
            not defer_type_detection
            and not configuration.drop_empty_columns
            and empty is None
        ):
            return list(indexes)

        return partial(
            _select_feature_indexes,
            feature_type=feature_type,
            fixed_indexes=(
                None
                if defer_type_detection
                else indexes
            ),
            drop_empty_columns=(
                configuration.drop_empty_columns
            ),
            empty=empty,
        )

    # Числови характеристики.
    if numeric_indexes or defer_type_detection:

        transformers.append(
            (
                "numeric",
                _build_numeric_pipeline(
                    configuration=configuration,
                    defer_type_detection=defer_type_detection,
                ),
                make_selector(
                    "numeric",
                    numeric_indexes,
                ),
            )
        )

    # Категориални характеристики.
    if categorical_indexes or defer_type_detection:

        if (
            configuration.categorical_imputation_strategy
            == "most_frequent"
        ):

            transformers.append(
                (
                    "categorical",
                    _build_categorical_pipeline(
                        configuration=configuration,
                    ),
                    make_selector(
                        "categorical",
                        categorical_indexes,
                        empty=False,
                    ),
                )
            )

            # Резервна стратегия за изцяло празни колони.
            if not configuration.drop_empty_columns:

                transformers.append(
                    (
                        "categorical_empty",
                        _build_categorical_pipeline(
                            configuration=configuration,
                            force_constant=True,
                        ),
                        make_selector(
                            "categorical",
                            categorical_indexes,
                            empty=True,
                        ),
                    )
                )

        else:

            transformers.append(
                (
                    "categorical",
                    _build_categorical_pipeline(
                        configuration=configuration,
                    ),
                    make_selector(
                        "categorical",
                        categorical_indexes,
                    ),
                )
            )

    # Датни характеристики.
    if datetime_indexes or defer_type_detection:

        transformers.append(
            (
                "datetime",
                _build_datetime_pipeline(
                    configuration=configuration,
                ),
                make_selector(
                    "datetime",
                    datetime_indexes,
                ),
            )
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

    # Запазваме досегашната структура,
    # когато Feature Selection е изключен.
    if not configuration.drop_constant_features:
        return column_transformer

    return Pipeline(
        steps=[
            (
                "columns",
                column_transformer,
            ),
            (
                "variance",
                VarianceThreshold(
                    threshold=0.0,
                ),
            ),
        ]
    )


def get_feature_selection_report(
    prepared_data: PreparedExperimentData,
    fitted_preprocessor: ColumnTransformer | Pipeline,
) -> FeatureSelectionReport:
    """Връща информация за действително премахнатите колони."""

    configuration = (
        prepared_data.preprocessing_configuration
    )

    if isinstance(
        fitted_preprocessor,
        Pipeline,
    ):
        column_transformer = (
            fitted_preprocessor.named_steps["columns"]
        )

    else:
        column_transformer = fitted_preprocessor

    check_is_fitted(column_transformer)

    removed_empty_columns = []
    removed_constant_features = []

    # Празните колони са останали извън всички
    # групи на обучения ColumnTransformer.
    if configuration.drop_empty_columns:

        for name, _, indexes in (
            column_transformer.transformers_
        ):

            if name == "remainder":

                removed_empty_columns.extend(
                    prepared_data.feature_columns[int(index)]
                    for index in indexes
                )

                break

    # Определяме кои трансформирани характеристики
    # са премахнати от VarianceThreshold.
    if configuration.drop_constant_features:

        variance_selector = (
            fitted_preprocessor.named_steps["variance"]
        )

        check_is_fitted(variance_selector)

        feature_names = (
            column_transformer.get_feature_names_out()
        )

        support_mask = (
            variance_selector.get_support()
        )

        removed_constant_features.extend(
            str(feature_name)
            for feature_name, selected in zip(
                feature_names,
                support_mask,
            )
            if not selected
        )

    return FeatureSelectionReport(
        removed_empty_columns=tuple(
            removed_empty_columns
        ),
        removed_constant_features=tuple(
            removed_constant_features
        ),
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

    Поддържа:
        - числови характеристики;
        - категориални характеристики;
        - датни характеристики;
        - конфигурируемо попълване на липсващи стойности;
        - премахване на изцяло празни колони;
        - премахване на постоянни характеристики;
        - конфигурируемо мащабиране;
        - конфигурируемо категориално кодиране.

    При defer_type_detection=True типовете се разпознават
    след разделянето на Training и Test.

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

    # При реално обучение отлагаме разпознаването
    # на типовете до Training.
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