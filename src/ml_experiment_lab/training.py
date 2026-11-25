"""Обучение на ML модели за експерименти."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from ml_experiment_lab.datasets import LoadedDataset

from ml_experiment_lab.preprocessing import (
    PreparedExperimentData,
    PreprocessingConfiguration,
    parse_numeric_value,
    prepare_experiment_data,
    with_training_feature_types,
)

from ml_experiment_lab.splitting import (
    ExperimentDataSplit,
    split_experiment_data,
)

from ml_experiment_lab.transformation import (
    TransformedExperimentSplit,
    transform_experiment_split,
)


TaskType = Literal["classification", "regression"]


class ModelTrainingError(ValueError):
    """Грешка при конфигуриране или обучение на ML модел."""


CLASSIFICATION_ALGORITHMS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree Classifier",
    "random_forest": "Random Forest Classifier",
}

REGRESSION_ALGORITHMS = {
    "linear_regression": "Linear Regression",
    "decision_tree": "Decision Tree Regressor",
    "random_forest": "Random Forest Regressor",
}


@dataclass(frozen=True)
class TrainedExperiment:
    """Резултат от обучението на ML модел."""

    estimator: BaseEstimator
    task_type: TaskType
    algorithm: str
    target_column: str
    feature_columns: tuple[str, ...]
    prepared_data: PreparedExperimentData
    data_split: ExperimentDataSplit
    transformed_data: TransformedExperimentSplit
    random_state: int

    @property
    def training_row_count(self) -> int:
        """Връща броя редове, използвани за обучение."""

        return self.transformed_data.train_row_count

    @property
    def test_row_count(self) -> int:
        """Връща броя редове, оставени за тестване."""

        return self.transformed_data.test_row_count

    @property
    def feature_count(self) -> int:
        """Връща броя характеристики след обработката."""

        return self.transformed_data.feature_count

    @property
    def preprocessing_configuration(
        self,
    ) -> PreprocessingConfiguration:
        """Връща използваната конфигурация за обработка."""

        return (
            self.prepared_data.preprocessing_configuration
        )


def get_available_algorithms(
    task_type: TaskType,
) -> tuple[tuple[str, str], ...]:
    """Връща наличните алгоритми за избрания тип задача."""

    if task_type == "classification":
        algorithms = CLASSIFICATION_ALGORITHMS

    elif task_type == "regression":
        algorithms = REGRESSION_ALGORITHMS

    else:
        raise ModelTrainingError(
            f"Неподдържан тип задача: {task_type}"
        )

    return tuple(algorithms.items())


def _create_estimator(
    task_type: TaskType,
    algorithm: str,
    random_state: int,
) -> BaseEstimator:
    """Създава ML модел според типа задача и алгоритъма."""

    if task_type == "classification":

        if algorithm == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            )

        if algorithm == "decision_tree":
            return DecisionTreeClassifier(
                random_state=random_state,
            )

        if algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            )

    elif task_type == "regression":

        if algorithm == "linear_regression":
            return LinearRegression()

        if algorithm == "decision_tree":
            return DecisionTreeRegressor(
                random_state=random_state,
            )

        if algorithm == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            )

    raise ModelTrainingError(
        f"Алгоритъмът „{algorithm}“ не е наличен "
        f"за задача „{task_type}“."
    )


def _parse_regression_target(
    values: np.ndarray,
) -> np.ndarray:
    """Преобразува целевите стойности при регресия.

    Поддържа цели числа, десетична точка,
    десетична запетая и научна нотация.

    Невалидните и безкрайните стойности
    не се допускат.
    """

    numeric_values = []

    for value in values:
        text = str(value).strip()

        number = parse_numeric_value(text)

        if number is None:

            try:
                candidate = float(
                    text.replace(",", ".")
                )

            except (ValueError, TypeError, OverflowError):
                candidate = None

            if (
                candidate is not None
                and not np.isfinite(candidate)
            ):
                raise ModelTrainingError(
                    "Целевата колона съдържа невалидни "
                    f"числови стойности: „{text}“."
                )

            raise ModelTrainingError(
                "При регресия целевата колона трябва "
                "да съдържа числови стойности. "
                f"Невалидна стойност: „{text}“."
            )

        numeric_values.append(number)

    return np.asarray(
        numeric_values,
        dtype=float,
    )


def _prepare_target_values(
    task_type: TaskType,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Проверява и подготвя целевите стойности."""

    # При класификация Target остава категориален.
    if task_type == "classification":

        unique_classes = np.unique(y_train)

        if len(unique_classes) < 2:
            raise ModelTrainingError(
                "За класификация са необходими поне "
                "два различни класа в тренировъчните данни."
            )

        return y_train, y_test

    # При регресия преобразуваме Target в числа.
    numeric_train = _parse_regression_target(
        y_train
    )

    numeric_test = _parse_regression_target(
        y_test
    )

    return numeric_train, numeric_test


def train_experiment_model(
    dataset: LoadedDataset,
    target_column: str,
    feature_columns: tuple[str, ...] | list[str],
    task_type: TaskType,
    algorithm: str,
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    preprocessing_configuration: (
        PreprocessingConfiguration | None
    ) = None,
) -> TrainedExperiment:
    """Подготвя, разделя, трансформира и обучава ML модел.

    Последователност:
        1. Подготвя необработените входни и целеви данни.
        2. Приема настройките за предварителна обработка.
        3. Разделя данните на Training и Test.
        4. Разпознава типовете само от Training.
        5. Обучава трансформатора само върху Training.
        6. Трансформира Training и Test.
        7. Проверява и преобразува целевите стойности.
        8. Създава ML модел.
        9. Обучава модела върху Training данните.

    Ако няма подадена конфигурация, използва
    настройките за обработка по подразбиране.

    Тестовите данни не участват в разпознаването
    на типовете или обучението на трансформатора
    и ML модела.
    """

    available_algorithms = dict(
        get_available_algorithms(task_type)
    )

    if algorithm not in available_algorithms:
        raise ModelTrainingError(
            f"Неподдържан алгоритъм „{algorithm}“ "
            f"за задача „{task_type}“."
        )

    # Подготвяме необработените данни и конфигурацията.
    prepared_data = prepare_experiment_data(
        dataset=dataset,
        target_column=target_column,
        feature_columns=feature_columns,
        defer_type_detection=True,
        preprocessing_configuration=(
            preprocessing_configuration
        ),
    )

    # Разделяме необработените данни.
    data_split = split_experiment_data(
        prepared_data=prepared_data,
        test_size=test_size,
        random_state=random_state,
    )

    # Записваме типовете само от Training.
    prepared_data = with_training_feature_types(
        prepared_data=prepared_data,
        X_train=data_split.X_train,
    )

    # Обучаваме трансформатора върху Training.
    # Прилагаме научените параметри върху Test.
    transformed_data = transform_experiment_split(
        prepared_data=prepared_data,
        data_split=data_split,
    )

    # Подготвяме целевите стойности според типа задача.
    y_train, y_test = _prepare_target_values(
        task_type=task_type,
        y_train=transformed_data.y_train,
        y_test=transformed_data.y_test,
    )

    if task_type == "regression":
        transformed_data = replace(
            transformed_data,
            y_train=y_train,
            y_test=y_test,
        )

    # Създаваме избрания ML модел.
    estimator = _create_estimator(
        task_type=task_type,
        algorithm=algorithm,
        random_state=random_state,
    )

    # Обучаваме модела само върху Training.
    try:
        estimator.fit(
            transformed_data.X_train,
            y_train,
        )

    except (ValueError, TypeError, OverflowError) as exc:
        raise ModelTrainingError(
            f"Обучението на модела не успя: {exc}"
        ) from exc

    return TrainedExperiment(
        estimator=estimator,
        task_type=task_type,
        algorithm=algorithm,
        target_column=target_column,
        feature_columns=prepared_data.feature_columns,
        prepared_data=prepared_data,
        data_split=data_split,
        transformed_data=transformed_data,
        random_state=random_state,
    )