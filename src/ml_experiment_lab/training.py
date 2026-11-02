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
    prepare_experiment_data,
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


def _prepare_target_values(
    task_type: TaskType,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Проверява и подготвя целевите стойности."""

    if task_type == "classification":
        unique_classes = np.unique(y_train)

        if len(unique_classes) < 2:
            raise ModelTrainingError(
                "За класификация са необходими поне "
                "два различни класа в тренировъчните данни."
            )

        return y_train, y_test

    try:
        numeric_train = np.asarray(y_train, dtype=float)
        numeric_test = np.asarray(y_test, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ModelTrainingError(
            "При регресия целевата колона трябва "
            "да съдържа числови стойности."
        ) from exc

    if (
        not np.isfinite(numeric_train).all()
        or not np.isfinite(numeric_test).all()
    ):
        raise ModelTrainingError(
            "Целевата колона съдържа невалидни числови стойности."
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
) -> TrainedExperiment:
    """Подготвя, разделя, трансформира и обучава ML модел.

    Последователност:
        1. Подготвя входните и целевите данни.
        2. Разделя ги на Training и Test.
        3. Обучава трансформатора само върху Training.
        4. Трансформира Training и Test.
        5. Създава ML модел.
        6. Обучава модела върху Training данните.

    Тестовите данни не участват в обучението на модела.
    Те ще се използват за последваща оценка.
    """

    available_algorithms = dict(
        get_available_algorithms(task_type)
    )

    if algorithm not in available_algorithms:
        raise ModelTrainingError(
            f"Неподдържан алгоритъм „{algorithm}“ "
            f"за задача „{task_type}“."
        )

    prepared_data = prepare_experiment_data(
        dataset=dataset,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    data_split = split_experiment_data(
        prepared_data=prepared_data,
        test_size=test_size,
        random_state=random_state,
    )

    transformed_data = transform_experiment_split(
        prepared_data=prepared_data,
        data_split=data_split,
    )

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

    estimator = _create_estimator(
        task_type=task_type,
        algorithm=algorithm,
        random_state=random_state,
    )

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