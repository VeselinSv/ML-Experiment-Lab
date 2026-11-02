"""Прилагане на предварителната обработка след разделяне на данните."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer

from ml_experiment_lab.preprocessing import PreparedExperimentData
from ml_experiment_lab.splitting import ExperimentDataSplit


class TransformationError(ValueError):
    """Грешка при трансформиране на тренировъчни и тестови данни."""


@dataclass(frozen=True)
class TransformedExperimentSplit:
    """Обработени Training и Test данни."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    fitted_preprocessor: ColumnTransformer
    feature_names: tuple[str, ...]

    @property
    def train_row_count(self) -> int:
        """Връща броя обработени тренировъчни редове."""
        return len(self.X_train)

    @property
    def test_row_count(self) -> int:
        """Връща броя обработени тестови редове."""
        return len(self.X_test)

    @property
    def feature_count(self) -> int:
        """Връща броя характеристики след обработката."""
        return self.X_train.shape[1]


def transform_experiment_split(
    prepared_data: PreparedExperimentData,
    data_split: ExperimentDataSplit,
) -> TransformedExperimentSplit:
    """Обучава трансформатора само върху Training данните.

    Training:
        fit_transform() научава параметрите за обработка
        и трансформира тренировъчните данни.

    Test:
        transform() прилага вече научените параметри,
        без да ги променя.

    Оригиналният трансформатор в prepared_data не се променя,
    защото използваме clone() за създаване на отделно копие.
    """

    X_train_raw = np.asarray(data_split.X_train, dtype=object)
    X_test_raw = np.asarray(data_split.X_test, dtype=object)

    y_train = np.asarray(data_split.y_train)
    y_test = np.asarray(data_split.y_test)

    expected_feature_count = len(prepared_data.feature_columns)

    if X_train_raw.ndim != 2 or X_test_raw.ndim != 2:
        raise TransformationError(
            "Входните тренировъчни и тестови данни трябва "
            "да бъдат двумерни."
        )

    if (
        X_train_raw.shape[1] != expected_feature_count
        or X_test_raw.shape[1] != expected_feature_count
    ):
        raise TransformationError(
            "Броят на входните колони не съвпада "
            "с конфигурацията на експеримента."
        )

    if len(X_train_raw) == 0 or len(X_test_raw) == 0:
        raise TransformationError(
            "Тренировъчната и тестовата извадка "
            "трябва да съдържат поне един ред."
        )

    if y_train.ndim != 1 or y_test.ndim != 1:
        raise TransformationError(
            "Целевите стойности трябва да бъдат едномерни."
        )

    if len(X_train_raw) != len(y_train):
        raise TransformationError(
            "Броят на тренировъчните входни редове "
            "не съвпада с броя на целевите стойности."
        )

    if len(X_test_raw) != len(y_test):
        raise TransformationError(
            "Броят на тестовите входни редове "
            "не съвпада с броя на целевите стойности."
        )

    # Създаваме независимо копие на необучения трансформатор.
    preprocessor = clone(prepared_data.preprocessor)

    try:
        # Научаваме параметрите само от тренировъчните данни.
        X_train = preprocessor.fit_transform(X_train_raw)

        # Прилагаме научените параметри върху тестовите данни.
        X_test = preprocessor.transform(X_test_raw)

        # Получаваме имената на характеристиките след обработката.
        feature_names = tuple(
            str(name)
            for name in preprocessor.get_feature_names_out()
        )

    except (ValueError, TypeError, AttributeError) as exc:
        raise TransformationError(
            f"Предварителната обработка не успя: {exc}"
        ) from exc

    # Преобразуваме резултатите в числови NumPy масиви.
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # Проверяваме дали обработката е създала валидни стойности.
    if not np.isfinite(X_train).all():
        raise TransformationError(
            "Тренировъчните данни съдържат невалидни "
            "числови стойности след обработката."
        )

    if not np.isfinite(X_test).all():
        raise TransformationError(
            "Тестовите данни съдържат невалидни "
            "числови стойности след обработката."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise TransformationError(
            "Броят на характеристиките в Training и Test "
            "извадките не съвпада."
        )

    return TransformedExperimentSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        fitted_preprocessor=preprocessor,
        feature_names=feature_names,
    )