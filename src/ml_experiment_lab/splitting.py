"""Разделяне на данните за ML експерименти."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from ml_experiment_lab.preprocessing import PreparedExperimentData


class DataSplitError(ValueError):
    """Грешка при разделяне на данните."""


@dataclass(frozen=True)
class ExperimentDataSplit:
    """Training и test части на подготвен ML експеримент."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    test_size: float
    random_state: int

    @property
    def train_row_count(self) -> int:
        """Връща броя редове в обучаващата извадка."""
        return len(self.X_train)

    @property
    def test_row_count(self) -> int:
        """Връща броя редове в тестовата извадка."""
        return len(self.X_test)


def split_experiment_data(
    prepared_data: PreparedExperimentData,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ExperimentDataSplit:
    """Разделя подготвените данни на training и test извадки."""

    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not math.isfinite(test_size)
        or not 0 < test_size < 1
    ):
        raise DataSplitError(
            "Размерът на тестовата извадка трябва да бъде "
            "число между 0 и 1."
        )

    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise DataSplitError(
            "Random state трябва да бъде цяло число."
        )

    X = np.asarray(prepared_data.X)
    y = np.asarray(prepared_data.y)

    if X.ndim != 2 or y.ndim != 1:
        raise DataSplitError(
            "Входните данни или целевите стойности имат "
            "невалидна структура."
        )

    if len(X) != len(y):
        raise DataSplitError(
            "Броят на входните редове не съвпада с броя "
            "на целевите стойности."
        )

    if len(X) < 2:
        raise DataSplitError(
            "Необходими са поне два реда за разделяне "
            "на обучаваща и тестова извадка."
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=random_state,
            shuffle=True,
        )
    except ValueError as exc:
        raise DataSplitError(
            f"Данните не могат да бъдат разделени: {exc}"
        ) from exc

    return ExperimentDataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_size=float(test_size),
        random_state=random_state,
    )