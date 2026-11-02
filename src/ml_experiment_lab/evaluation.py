"""Оценяване на обучени ML модели."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from ml_experiment_lab.training import TrainedExperiment


class ModelEvaluationError(ValueError):
    """Грешка при оценяването на ML модел."""


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Резултати от оценяването на обучен ML модел."""

    task_type: str
    metrics: dict[str, float | None]
    y_true: tuple[str | float, ...]
    y_pred: tuple[str | float, ...]
    class_labels: tuple[str, ...] = ()
    confusion_matrix: tuple[tuple[int, ...], ...] | None = None


def _validate_test_data(
    y_true: np.ndarray,
    X_test: np.ndarray,
) -> None:
    """Проверява структурата на тестовите данни."""

    if X_test.ndim != 2:
        raise ModelEvaluationError(
            "Тестовите входни данни трябва да бъдат двумерни."
        )

    if y_true.ndim != 1:
        raise ModelEvaluationError(
            "Тестовите целеви стойности трябва да бъдат едномерни."
        )

    if len(X_test) == 0 or len(y_true) == 0:
        raise ModelEvaluationError(
            "Няма налични тестови данни за оценяване."
        )

    if len(X_test) != len(y_true):
        raise ModelEvaluationError(
            "Броят на тестовите редове не съвпада "
            "с броя на целевите стойности."
        )

    if not np.isfinite(X_test).all():
        raise ModelEvaluationError(
            "Тестовите входни данни съдържат невалидни стойности."
        )


def _evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ModelEvaluationResult:
    """Изчислява метриките за класификация."""

    true_labels = np.asarray(
        [str(value) for value in y_true],
        dtype=str,
    )

    predicted_labels = np.asarray(
        [str(value) for value in y_pred],
        dtype=str,
    )

    labels = tuple(
        sorted(
            set(true_labels.tolist())
            | set(predicted_labels.tolist())
        )
    )

    try:
        accuracy = accuracy_score(
            true_labels,
            predicted_labels,
        )

        precision = precision_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )

        matrix = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=list(labels),
        )

    except (ValueError, TypeError) as exc:
        raise ModelEvaluationError(
            f"Оценяването на класификацията не успя: {exc}"
        ) from exc

    return ModelEvaluationResult(
        task_type="classification",
        metrics={
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
        },
        y_true=tuple(true_labels.tolist()),
        y_pred=tuple(predicted_labels.tolist()),
        class_labels=labels,
        confusion_matrix=tuple(
            tuple(int(value) for value in row)
            for row in matrix.tolist()
        ),
    )


def _evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ModelEvaluationResult:
    """Изчислява метриките за регресия."""

    try:
        true_values = np.asarray(
            y_true,
            dtype=float,
        )

        predicted_values = np.asarray(
            y_pred,
            dtype=float,
        )

    except (ValueError, TypeError) as exc:
        raise ModelEvaluationError(
            "Регресионните резултати трябва да бъдат числови."
        ) from exc

    if (
        not np.isfinite(true_values).all()
        or not np.isfinite(predicted_values).all()
    ):
        raise ModelEvaluationError(
            "Регресионните резултати съдържат невалидни стойности."
        )

    mae = mean_absolute_error(
        true_values,
        predicted_values,
    )

    mse = mean_squared_error(
        true_values,
        predicted_values,
    )

    rmse = float(np.sqrt(mse))

    # R² не е надеждно определен при по-малко от два тестови реда.
    if len(true_values) >= 2:
        r2 = float(
            r2_score(
                true_values,
                predicted_values,
            )
        )
    else:
        r2 = None

    return ModelEvaluationResult(
        task_type="regression",
        metrics={
            "mae": float(mae),
            "mse": float(mse),
            "rmse": rmse,
            "r2": r2,
        },
        y_true=tuple(float(value) for value in true_values),
        y_pred=tuple(float(value) for value in predicted_values),
    )


def evaluate_trained_experiment(
    experiment: TrainedExperiment,
) -> ModelEvaluationResult:
    """Оценява обучен ML модел върху тестовите данни.

    Моделът използва само X_test за генериране на прогнози.
    Получените прогнози се сравняват с y_test.

    Оценяването не променя обученото състояние на модела.
    """

    X_test = np.asarray(
        experiment.transformed_data.X_test,
        dtype=float,
    )

    y_true = np.asarray(
        experiment.transformed_data.y_test,
    )

    _validate_test_data(
        y_true=y_true,
        X_test=X_test,
    )

    try:
        y_pred = np.asarray(
            experiment.estimator.predict(X_test)
        )

    except (ValueError, TypeError, AttributeError) as exc:
        raise ModelEvaluationError(
            f"Моделът не успя да генерира прогнози: {exc}"
        ) from exc

    if y_pred.ndim != 1:
        raise ModelEvaluationError(
            "Моделът върна прогнози с невалидна структура."
        )

    if len(y_pred) != len(y_true):
        raise ModelEvaluationError(
            "Броят на прогнозите не съвпада "
            "с броя на действителните стойности."
        )

    if experiment.task_type == "classification":
        return _evaluate_classification(
            y_true=y_true,
            y_pred=y_pred,
        )

    if experiment.task_type == "regression":
        return _evaluate_regression(
            y_true=y_true,
            y_pred=y_pred,
        )

    raise ModelEvaluationError(
        f"Неподдържан тип задача: {experiment.task_type}"
    )