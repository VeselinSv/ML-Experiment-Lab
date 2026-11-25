"""Подробен отчет за предварителната обработка на ML данни."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TYPE_CHECKING

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ml_experiment_lab.datetime_features import (
    DATETIME_FEATURE_NAMES,
    is_datetime_value,
)

from ml_experiment_lab.encoding import FrequencyEncoder

from ml_experiment_lab.preprocessing import (
    PreprocessingConfiguration,
    get_feature_selection_report,
    parse_numeric_value,
)

if TYPE_CHECKING:
    from ml_experiment_lab.training import TrainedExperiment


ColumnType = Literal[
    "numeric",
    "categorical",
    "datetime",
    "empty",
]

ReportValue = str | float


COLUMN_TYPE_NAMES = {
    "numeric": "Числова",
    "categorical": "Категориална",
    "datetime": "Датна",
    "empty": "Изцяло празна",
}


DATETIME_FEATURE_LABELS = {
    "year": "година",
    "month": "месец",
    "day": "ден",
    "day_of_week": "ден от седмицата",
    "quarter": "тримесечие",
    "day_of_year": "ден от годината",
    "hour": "час",
    "minute": "минута",
    "second": "секунда",
}


class PreprocessingReportError(ValueError):
    """Грешка при създаването на Preprocessing отчет."""


@dataclass(frozen=True)
class CategoryReport:
    """Информация за една научена категория."""

    category: str

    encoded_value: float | None = None

    feature_name: str | None = None


@dataclass(frozen=True)
class FeatureReport:
    """Информация за една трансформирана характеристика."""

    technical_name: str

    display_name: str

    source_column: str

    removed_as_constant: bool = False

    imputation_value: float | None = None

    scaling_parameters: tuple[
        tuple[str, float],
        ...,
    ] = ()

    @property
    def is_retained(self) -> bool:
        """Проверява дали характеристиката е запазена."""

        return not self.removed_as_constant


@dataclass(frozen=True)
class ColumnPreprocessingReport:
    """Информация за обработката на една входна колона."""

    column_name: str

    column_type: ColumnType

    is_mixed: bool = False

    is_empty: bool = False

    removed_as_empty: bool = False

    training_missing_count: int = 0

    test_missing_count: int = 0

    training_invalid_count: int = 0

    test_invalid_count: int = 0

    training_unknown_count: int = 0

    test_unknown_count: int = 0

    imputation_strategy: str | None = None

    categorical_imputation_value: str | None = None

    scaling_method: str | None = None

    encoding_method: str | None = None

    categories: tuple[CategoryReport, ...] = ()

    features: tuple[FeatureReport, ...] = ()

    @property
    def total_missing_count(self) -> int:
        """Връща общия брой липсващи стойности."""

        return (
            self.training_missing_count
            + self.test_missing_count
        )

    @property
    def total_invalid_count(self) -> int:
        """Връща общия брой невалидни стойности."""

        return (
            self.training_invalid_count
            + self.test_invalid_count
        )

    @property
    def total_unknown_count(self) -> int:
        """Връща общия брой непознати категории."""

        return (
            self.training_unknown_count
            + self.test_unknown_count
        )

    @property
    def generated_feature_count(self) -> int:
        """Връща броя генерирани характеристики."""

        return len(self.features)

    @property
    def retained_feature_count(self) -> int:
        """Връща броя запазени характеристики."""

        return sum(
            feature.is_retained
            for feature in self.features
        )

    @property
    def removed_constant_count(self) -> int:
        """Връща броя премахнати постоянни характеристики."""

        return sum(
            feature.removed_as_constant
            for feature in self.features
        )

    @property
    def learned_category_count(self) -> int:
        """Връща броя научени категории."""

        return len(self.categories)


@dataclass(frozen=True)
class PreprocessingReport:
    """Цялостен отчет за предварителната обработка."""

    configuration: PreprocessingConfiguration

    target_column: str

    original_row_count: int

    excluded_row_count: int

    training_row_count: int

    test_row_count: int

    random_state: int

    requested_test_size: float

    columns: tuple[ColumnPreprocessingReport, ...]

    generated_features: tuple[FeatureReport, ...]

    removed_empty_columns: tuple[str, ...] = ()

    warnings: tuple[str, ...] = ()

    @property
    def valid_row_count(self) -> int:
        """Връща броя използваеми наблюдения."""

        return (
            self.training_row_count
            + self.test_row_count
        )

    @property
    def actual_test_size(self) -> float:
        """Връща действителния дял на тестовата извадка."""

        if self.valid_row_count == 0:
            return 0.0

        return (
            self.test_row_count
            / self.valid_row_count
        )

    @property
    def input_feature_count(self) -> int:
        """Връща първоначалния брой входни колони."""

        return len(self.columns)

    @property
    def generated_feature_count(self) -> int:
        """Връща броя характеристики преди Feature Selection."""

        return len(self.generated_features)

    @property
    def retained_features(self) -> tuple[FeatureReport, ...]:
        """Връща всички запазени характеристики."""

        return tuple(
            feature
            for feature in self.generated_features
            if feature.is_retained
        )

    @property
    def final_feature_count(self) -> int:
        """Връща крайния брой характеристики."""

        return len(self.retained_features)

    @property
    def final_feature_names(self) -> tuple[str, ...]:
        """Връща четимите имена на крайните характеристики."""

        return tuple(
            feature.display_name
            for feature in self.retained_features
        )

    @property
    def final_technical_feature_names(self) -> tuple[str, ...]:
        """Връща техническите имена на крайните характеристики."""

        return tuple(
            feature.technical_name
            for feature in self.retained_features
        )

    @property
    def removed_constant_features(self) -> tuple[str, ...]:
        """Връща четимите имена на постоянните характеристики."""

        return tuple(
            feature.display_name
            for feature in self.generated_features
            if feature.removed_as_constant
        )

    @property
    def removed_constant_technical_names(self) -> tuple[str, ...]:
        """Връща техническите имена на премахнатите характеристики."""

        return tuple(
            feature.technical_name
            for feature in self.generated_features
            if feature.removed_as_constant
        )

    @property
    def removed_empty_count(self) -> int:
        """Връща броя премахнати празни входни колони."""

        return len(self.removed_empty_columns)

    @property
    def removed_constant_count(self) -> int:
        """Връща броя премахнати постоянни характеристики."""

        return len(self.removed_constant_features)

    @property
    def numeric_column_count(self) -> int:
        """Връща броя числови колони."""

        return sum(
            column.column_type == "numeric"
            for column in self.columns
        )

    @property
    def categorical_column_count(self) -> int:
        """Връща броя непразни категориални колони."""

        return sum(
            column.column_type == "categorical"
            for column in self.columns
        )

    @property
    def datetime_column_count(self) -> int:
        """Връща броя датни колони."""

        return sum(
            column.column_type == "datetime"
            for column in self.columns
        )

    @property
    def empty_column_count(self) -> int:
        """Връща броя изцяло празни тренировъчни колони."""

        return sum(
            column.is_empty
            for column in self.columns
        )

    @property
    def mixed_column_count(self) -> int:
        """Връща броя колони със смесени типове."""

        return sum(
            column.is_mixed
            for column in self.columns
        )

    @property
    def total_missing_count(self) -> int:
        """Връща общия брой липсващи входни стойности."""

        return sum(
            column.total_missing_count
            for column in self.columns
        )

    @property
    def total_invalid_count(self) -> int:
        """Връща общия брой невалидни входни стойности."""

        return sum(
            column.total_invalid_count
            for column in self.columns
        )

    @property
    def total_unknown_count(self) -> int:
        """Връща общия брой непознати категориални стойности."""

        return sum(
            column.total_unknown_count
            for column in self.columns
        )

    def get_column_report(
        self,
        column_name: str,
    ) -> ColumnPreprocessingReport:
        """Намира отчета за конкретна входна колона."""

        for column in self.columns:

            if column.column_name == column_name:
                return column

        raise PreprocessingReportError(
            f"Колоната „{column_name}“ не присъства в отчета."
        )


def _is_missing(value: object) -> bool:
    """Проверява липсваща стойност в необработените данни."""

    if isinstance(value, str):
        return not value.strip()

    if value is None:
        return True

    if isinstance(
        value,
        (float, np.floating),
    ):
        return bool(np.isnan(value))

    return False


def _count_missing_and_invalid(
    values: np.ndarray,
    column_type: ColumnType,
) -> tuple[int, int]:
    """Преброява липсващите и невалидните стойности."""

    missing_count = 0
    invalid_count = 0

    for value in values:

        if _is_missing(value):
            missing_count += 1
            continue

        if column_type == "numeric":

            if parse_numeric_value(str(value)) is None:
                invalid_count += 1

        elif column_type == "datetime":

            if not is_datetime_value(str(value)):
                invalid_count += 1

    return missing_count, invalid_count


def _count_unknown_categories(
    values: np.ndarray,
    known_categories: set[str],
    imputation_value: str,
) -> int:
    """Преброява непознатите категории след Imputation."""

    unknown_count = 0

    for value in values:

        if _is_missing(value):
            category = imputation_value

        else:
            category = str(value).strip()

        if category not in known_categories:
            unknown_count += 1

    return unknown_count


def _get_column_type(
    prepared_data,
    column_name: str,
) -> ColumnType:
    """Определя вече разпознатия тип на входна колона."""

    if column_name in prepared_data.empty_columns:
        return "empty"

    if column_name in prepared_data.numeric_columns:
        return "numeric"

    if column_name in prepared_data.datetime_columns:
        return "datetime"

    if column_name in prepared_data.categorical_columns:
        return "categorical"

    raise PreprocessingReportError(
        f"Типът на колоната „{column_name}“ не е определен."
    )


def _get_scaling_parameters(
    scaler,
    method: str,
    feature_index: int,
) -> tuple[tuple[str, float], ...]:
    """Извлича научените параметри за една характеристика."""

    if method == "none":
        return ()

    attribute_names = {
        "standard": (
            ("mean", "mean_"),
            ("variance", "var_"),
            ("scale", "scale_"),
        ),
        "minmax": (
            ("minimum", "data_min_"),
            ("maximum", "data_max_"),
            ("range", "data_range_"),
            ("scale", "scale_"),
            ("offset", "min_"),
        ),
        "robust": (
            ("center", "center_"),
            ("scale", "scale_"),
        ),
        "maxabs": (
            ("max_abs", "max_abs_"),
            ("scale", "scale_"),
        ),
    }

    if method not in attribute_names:
        raise PreprocessingReportError(
            f"Неподдържан метод за мащабиране: „{method}“."
        )

    return tuple(
        (
            parameter_name,
            float(
                getattr(
                    scaler,
                    attribute_name,
                )[feature_index]
            ),
        )
        for parameter_name, attribute_name
        in attribute_names[method]
    )


def _create_feature_report(
    *,
    technical_name: str,
    display_name: str,
    source_column: str,
    retained: bool,
    imputation_value: float | None = None,
    scaling_parameters: tuple[
        tuple[str, float],
        ...,
    ] = (),
) -> FeatureReport:
    """Създава отчет за една трансформирана характеристика."""

    return FeatureReport(
        technical_name=technical_name,
        display_name=display_name,
        source_column=source_column,
        removed_as_constant=not retained,
        imputation_value=imputation_value,
        scaling_parameters=scaling_parameters,
    )


def _create_numeric_or_datetime_report(
    *,
    column_report: ColumnPreprocessingReport,
    pipeline: Pipeline,
    group_name: str,
    local_column_index: int,
    output_names: tuple[str, ...],
    output_support: np.ndarray,
    configuration: PreprocessingConfiguration,
) -> ColumnPreprocessingReport:
    """Създава отчет за числова или датна колона."""

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]

    if group_name == "datetime":

        component_names = DATETIME_FEATURE_NAMES

        start_index = (
            local_column_index
            * len(component_names)
        )

    else:

        component_names = (None,)

        start_index = local_column_index

    features = []

    for component_index, component_name in enumerate(
        component_names
    ):

        local_feature_index = (
            start_index + component_index
        )

        if component_name is None:

            display_name = column_report.column_name

        else:

            component_label = (
                DATETIME_FEATURE_LABELS[component_name]
            )

            display_name = (
                f"{column_report.column_name}"
                f" — {component_label}"
            )

        feature = _create_feature_report(
            technical_name=(
                output_names[local_feature_index]
            ),
            display_name=display_name,
            source_column=column_report.column_name,
            retained=bool(
                output_support[local_feature_index]
            ),
            imputation_value=float(
                imputer.statistics_[local_feature_index]
            ),
            scaling_parameters=_get_scaling_parameters(
                scaler=scaler,
                method=configuration.scaling_method,
                feature_index=local_feature_index,
            ),
        )

        features.append(feature)

    return replace(
        column_report,
        imputation_strategy=imputer.strategy,
        scaling_method=configuration.scaling_method,
        features=tuple(features),
    )


def _create_categorical_report(
    *,
    column_report: ColumnPreprocessingReport,
    pipeline: Pipeline,
    local_column_index: int,
    output_names: tuple[str, ...],
    output_support: np.ndarray,
    training_values: np.ndarray,
    test_values: np.ndarray,
    configuration: PreprocessingConfiguration,
) -> ColumnPreprocessingReport:
    """Създава отчет за една категориална колона."""

    imputer = pipeline.named_steps["imputer"]
    encoder = pipeline.named_steps["encoder"]

    method = configuration.encoding_method

    imputation_value = str(
        imputer.statistics_[local_column_index]
    )

    features = []
    categories = []

    if method == "onehot":

        learned_categories = tuple(
            str(category)
            for category
            in encoder.categories_[local_column_index]
        )

        # One-Hot създава по една характеристика
        # за всяка научена категория.
        start_index = sum(
            len(column_categories)
            for column_categories
            in encoder.categories_[:local_column_index]
        )

        for category_index, category in enumerate(
            learned_categories
        ):

            local_feature_index = (
                start_index + category_index
            )

            technical_name = (
                output_names[local_feature_index]
            )

            feature = _create_feature_report(
                technical_name=technical_name,
                display_name=(
                    f"{column_report.column_name}"
                    f" — {category}"
                ),
                source_column=column_report.column_name,
                retained=bool(
                    output_support[local_feature_index]
                ),
            )

            features.append(feature)

            categories.append(
                CategoryReport(
                    category=category,
                    feature_name=technical_name,
                )
            )

    elif method == "ordinal":

        learned_categories = tuple(
            str(category)
            for category
            in encoder.categories_[local_column_index]
        )

        technical_name = (
            output_names[local_column_index]
        )

        features.append(
            _create_feature_report(
                technical_name=technical_name,
                display_name=column_report.column_name,
                source_column=column_report.column_name,
                retained=bool(
                    output_support[local_column_index]
                ),
            )
        )

        for category_index, category in enumerate(
            learned_categories
        ):

            categories.append(
                CategoryReport(
                    category=category,
                    encoded_value=float(category_index),
                    feature_name=technical_name,
                )
            )

    elif method == "frequency":

        if not isinstance(encoder, FrequencyEncoder):
            raise PreprocessingReportError(
                "Очаква се обучен FrequencyEncoder."
            )

        frequency_map = encoder.frequency_maps_[
            local_column_index
        ]

        technical_name = (
            output_names[local_column_index]
        )

        features.append(
            _create_feature_report(
                technical_name=technical_name,
                display_name=column_report.column_name,
                source_column=column_report.column_name,
                retained=bool(
                    output_support[local_column_index]
                ),
            )
        )

        for category, frequency in sorted(
            frequency_map.items(),
            key=lambda item: str(item[0]),
        ):

            categories.append(
                CategoryReport(
                    category=str(category),
                    encoded_value=float(frequency),
                    feature_name=technical_name,
                )
            )

        learned_categories = tuple(
            category.category
            for category in categories
        )

    else:

        raise PreprocessingReportError(
            f"Неподдържан метод за кодиране: „{method}“."
        )

    known_categories = set(
        learned_categories
    )

    return replace(
        column_report,
        imputation_strategy=imputer.strategy,
        categorical_imputation_value=imputation_value,
        encoding_method=method,
        categories=tuple(categories),
        features=tuple(features),
        training_unknown_count=_count_unknown_categories(
            values=training_values,
            known_categories=known_categories,
            imputation_value=imputation_value,
        ),
        test_unknown_count=_count_unknown_categories(
            values=test_values,
            known_categories=known_categories,
            imputation_value=imputation_value,
        ),
    )


def _build_warnings(
    columns: tuple[ColumnPreprocessingReport, ...],
    configuration: PreprocessingConfiguration,
    excluded_row_count: int,
) -> tuple[str, ...]:
    """Създава предупреждения за предварителната обработка."""

    warnings = []

    if excluded_row_count > 0:

        warnings.append(
            f"Изключени са {excluded_row_count} реда "
            "поради липсваща целева стойност."
        )

    for column in columns:

        name = column.column_name

        if column.is_mixed:

            warnings.append(
                f"Колоната „{name}“ съдържа смесени типове "
                "в тренировъчните данни и е обработена "
                "като категориална."
            )

        if column.is_empty:

            if column.removed_as_empty:

                warnings.append(
                    f"Колоната „{name}“ е премахната, "
                    "защото е изцяло празна в Training."
                )

            else:

                warnings.append(
                    f"Колоната „{name}“ е изцяло празна "
                    "в Training, но е запазена за обработка."
                )

        if column.total_invalid_count > 0:

            warnings.append(
                f"Колоната „{name}“ съдържа "
                f"{column.total_invalid_count} невалидни "
                "стойности, обработени като липсващи."
            )

        if column.test_unknown_count > 0:

            warnings.append(
                f"Колоната „{name}“ съдържа "
                f"{column.test_unknown_count} непознати "
                "категориални стойности в Test."
            )

        if column.removed_constant_count > 0:

            warnings.append(
                f"От колоната „{name}“ са премахнати "
                f"{column.removed_constant_count} "
                "постоянни характеристики."
            )

        if column.encoding_method == "frequency":

            encoded_values = [
                category.encoded_value
                for category in column.categories
            ]

            if len(set(encoded_values)) < len(encoded_values):

                warnings.append(
                    f"При честотното кодиране на колоната "
                    f"„{name}“ различни категории получават "
                    "еднаква числова стойност."
                )

    if (
        configuration.encoding_method == "ordinal"
        and any(
            column.encoding_method == "ordinal"
            for column in columns
        )
    ):

        warnings.append(
            "Поредното кодиране може да въведе изкуствена "
            "числова подредба между категориите."
        )

    return tuple(warnings)


def build_preprocessing_report(
    experiment: TrainedExperiment,
) -> PreprocessingReport:
    """Създава подробен отчет от обучен ML експеримент.

    Отчетът използва съществуващите:
        - PreparedExperimentData;
        - ExperimentDataSplit;
        - обучен трансформатор;
        - TransformedExperimentSplit.

    Не обучава повторно трансформаторите.
    Не променя оригиналните данни.
    Не променя обучения модел.
    """

    prepared_data = experiment.prepared_data
    data_split = experiment.data_split
    transformed_data = experiment.transformed_data

    configuration = (
        prepared_data.preprocessing_configuration
    )

    fitted_preprocessor = (
        transformed_data.fitted_preprocessor
    )

    X_train = np.asarray(
        data_split.X_train,
        dtype=object,
    )

    X_test = np.asarray(
        data_split.X_test,
        dtype=object,
    )

    if X_train.ndim != 2 or X_test.ndim != 2:

        raise PreprocessingReportError(
            "Невалидна структура на входните матрици."
        )

    expected_column_count = len(
        prepared_data.feature_columns
    )

    if (
        X_train.shape[1] != expected_column_count
        or X_test.shape[1] != expected_column_count
    ):

        raise PreprocessingReportError(
            "Броят входни колони не съответства "
            "на конфигурацията на експеримента."
        )

    # Извличаме отчета за Feature Selection.
    selection_report = get_feature_selection_report(
        prepared_data=prepared_data,
        fitted_preprocessor=fitted_preprocessor,
    )

    removed_empty_columns = set(
        selection_report.removed_empty_columns
    )

    # При включен VarianceThreshold имаме общ Pipeline.
    # В противен случай имаме директен ColumnTransformer.
    if isinstance(fitted_preprocessor, Pipeline):

        column_transformer = (
            fitted_preprocessor.named_steps["columns"]
        )

        variance_selector = (
            fitted_preprocessor.named_steps["variance"]
        )

        check_is_fitted(variance_selector)

        support_mask = variance_selector.get_support()

    else:

        column_transformer = fitted_preprocessor

        support_mask = None

    check_is_fitted(column_transformer)

    # Имена преди премахването на постоянни характеристики.
    generated_names = tuple(
        str(name)
        for name
        in column_transformer.get_feature_names_out()
    )

    if support_mask is None:

        support_mask = np.ones(
            len(generated_names),
            dtype=bool,
        )

    else:

        support_mask = np.asarray(
            support_mask,
            dtype=bool,
        )

    if len(support_mask) != len(generated_names):

        raise PreprocessingReportError(
            "Невалидна информация за избраните характеристики."
        )

    column_reports = {}

    # Създаваме основен отчет за всяка оригинална колона.
    for column_index, column_name in enumerate(
        prepared_data.feature_columns
    ):

        column_type = _get_column_type(
            prepared_data,
            column_name,
        )

        training_missing, training_invalid = (
            _count_missing_and_invalid(
                X_train[:, column_index],
                column_type,
            )
        )

        test_missing, test_invalid = (
            _count_missing_and_invalid(
                X_test[:, column_index],
                column_type,
            )
        )

        column_reports[column_index] = (
            ColumnPreprocessingReport(
                column_name=column_name,
                column_type=column_type,
                is_mixed=(
                    column_name
                    in prepared_data.mixed_columns
                ),
                is_empty=(
                    column_name
                    in prepared_data.empty_columns
                ),
                removed_as_empty=(
                    column_name
                    in removed_empty_columns
                ),
                training_missing_count=training_missing,
                test_missing_count=test_missing,
                training_invalid_count=training_invalid,
                test_invalid_count=test_invalid,
            )
        )

    generated_features = []

    # Използваме действително обучените групи от колони.
    for group_name, pipeline, column_indexes in (
        column_transformer.transformers_
    ):

        if group_name == "remainder":
            continue

        if len(column_indexes) == 0:
            continue

        if not isinstance(pipeline, Pipeline):

            raise PreprocessingReportError(
                f"Невалиден трансформатор: „{group_name}“."
            )

        # Определяме позициите на характеристиките,
        # създадени от текущата група.
        output_slice = (
            column_transformer.output_indices_[group_name]
        )

        output_names = generated_names[
            output_slice
        ]

        output_support = support_mask[
            output_slice
        ]

        for local_column_index, column_index in enumerate(
            column_indexes
        ):

            column_index = int(column_index)

            column_report = column_reports[
                column_index
            ]

            if group_name in (
                "numeric",
                "datetime",
            ):

                updated_report = (
                    _create_numeric_or_datetime_report(
                        column_report=column_report,
                        pipeline=pipeline,
                        group_name=group_name,
                        local_column_index=local_column_index,
                        output_names=output_names,
                        output_support=output_support,
                        configuration=configuration,
                    )
                )

            elif group_name in (
                "categorical",
                "categorical_empty",
            ):

                updated_report = (
                    _create_categorical_report(
                        column_report=column_report,
                        pipeline=pipeline,
                        local_column_index=local_column_index,
                        output_names=output_names,
                        output_support=output_support,
                        training_values=(
                            X_train[:, column_index]
                        ),
                        test_values=(
                            X_test[:, column_index]
                        ),
                        configuration=configuration,
                    )
                )

            else:

                raise PreprocessingReportError(
                    f"Неподдържана група: „{group_name}“."
                )

            column_reports[column_index] = (
                updated_report
            )

            generated_features.extend(
                updated_report.features
            )

    # Запазваме оригиналния ред на входните колони.
    ordered_columns = tuple(
        column_reports[index]
        for index in range(expected_column_count)
    )

    ordered_features = tuple(
        generated_features
    )

    # Проверяваме пълното съответствие с трансформатора.
    report_generated_names = tuple(
        feature.technical_name
        for feature in ordered_features
    )

    if report_generated_names != generated_names:

        raise PreprocessingReportError(
            "Генерираните характеристики в отчета "
            "не съответстват на обучения трансформатор."
        )

    warning_messages = _build_warnings(
        columns=ordered_columns,
        configuration=configuration,
        excluded_row_count=prepared_data.excluded_rows,
    )

    report = PreprocessingReport(
        configuration=configuration,
        target_column=prepared_data.target_column,
        original_row_count=(
            len(prepared_data.X)
            + prepared_data.excluded_rows
        ),
        excluded_row_count=prepared_data.excluded_rows,
        training_row_count=X_train.shape[0],
        test_row_count=X_test.shape[0],
        random_state=data_split.random_state,
        requested_test_size=data_split.test_size,
        columns=ordered_columns,
        generated_features=ordered_features,
        removed_empty_columns=(
            selection_report.removed_empty_columns
        ),
        warnings=warning_messages,
    )

    # Финална проверка на размерността.
    if (
        report.final_feature_count
        != transformed_data.feature_count
    ):

        raise PreprocessingReportError(
            "Крайният брой характеристики в отчета "
            "не съответства на трансформираните данни."
        )

    if (
        report.final_technical_feature_names
        != transformed_data.feature_names
    ):

        raise PreprocessingReportError(
            "Крайните характеристики в отчета "
            "не съответстват на характеристиките на модела."
        )

    return report