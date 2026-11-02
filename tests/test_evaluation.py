"""Тестове за оценяване на обучени ML модели."""

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.training import train_experiment_model
from ml_experiment_lab.evaluation import (
    ModelEvaluationError,
    evaluate_trained_experiment,
)


class ModelEvaluationTests(unittest.TestCase):
    """Тестове за метриките при класификация и регресия."""

    def setUp(self) -> None:
        classification_rows = tuple(
            (
                str(index),
                "червено" if index < 20 else "синьо",
                "A" if index < 20 else "B",
            )
            for index in range(40)
        )

        self.classification_dataset = LoadedDataset(
            path=Path("classification.csv"),
            columns=("Стойност", "Цвят", "Клас"),
            rows=classification_rows,
        )

        regression_rows = tuple(
            (
                str(index),
                "група_A" if index % 2 == 0 else "група_B",
                str(100 + index * 2),
            )
            for index in range(40)
        )

        self.regression_dataset = LoadedDataset(
            path=Path("regression.csv"),
            columns=("Стойност", "Група", "Цена"),
            rows=regression_rows,
        )

        self.classification_experiment = train_experiment_model(
            dataset=self.classification_dataset,
            target_column="Клас",
            feature_columns=("Стойност", "Цвят"),
            task_type="classification",
            algorithm="decision_tree",
            test_size=0.2,
            random_state=42,
        )

        self.regression_experiment = train_experiment_model(
            dataset=self.regression_dataset,
            target_column="Цена",
            feature_columns=("Стойност", "Група"),
            task_type="regression",
            algorithm="linear_regression",
            test_size=0.2,
            random_state=42,
        )

    def test_classification_returns_all_metrics(self) -> None:
        result = evaluate_trained_experiment(
            self.classification_experiment
        )

        expected_metrics = {
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        }

        self.assertEqual(
            set(result.metrics.keys()),
            expected_metrics,
        )

        self.assertEqual(result.task_type, "classification")

    def test_classification_metrics_are_between_zero_and_one(
        self,
    ) -> None:
        result = evaluate_trained_experiment(
            self.classification_experiment
        )

        for metric_name, metric_value in result.metrics.items():
            with self.subTest(metric=metric_name):
                self.assertIsNotNone(metric_value)
                self.assertGreaterEqual(metric_value, 0.0)
                self.assertLessEqual(metric_value, 1.0)

    def test_classification_returns_confusion_matrix(self) -> None:
        result = evaluate_trained_experiment(
            self.classification_experiment
        )

        self.assertIsNotNone(result.confusion_matrix)

        matrix = np.asarray(result.confusion_matrix)

        self.assertEqual(
            matrix.shape,
            (
                len(result.class_labels),
                len(result.class_labels),
            ),
        )

        self.assertEqual(
            int(matrix.sum()),
            len(result.y_true),
        )

    def test_classification_returns_actual_and_predicted_values(
        self,
    ) -> None:
        result = evaluate_trained_experiment(
            self.classification_experiment
        )

        self.assertEqual(
            len(result.y_true),
            self.classification_experiment.test_row_count,
        )

        self.assertEqual(
            len(result.y_pred),
            self.classification_experiment.test_row_count,
        )

    def test_regression_returns_all_metrics(self) -> None:
        result = evaluate_trained_experiment(
            self.regression_experiment
        )

        expected_metrics = {
            "mae",
            "mse",
            "rmse",
            "r2",
        }

        self.assertEqual(
            set(result.metrics.keys()),
            expected_metrics,
        )

        self.assertEqual(result.task_type, "regression")
        self.assertIsNone(result.confusion_matrix)

    def test_regression_errors_are_non_negative(self) -> None:
        result = evaluate_trained_experiment(
            self.regression_experiment
        )

        for metric_name in ("mae", "mse", "rmse"):
            with self.subTest(metric=metric_name):
                self.assertGreaterEqual(
                    result.metrics[metric_name],
                    0.0,
                )

    def test_regression_metrics_are_finite(self) -> None:
        result = evaluate_trained_experiment(
            self.regression_experiment
        )

        for metric_name, metric_value in result.metrics.items():
            with self.subTest(metric=metric_name):
                if metric_value is not None:
                    self.assertTrue(
                        np.isfinite(metric_value)
                    )

    def test_regression_returns_actual_and_predicted_values(
        self,
    ) -> None:
        result = evaluate_trained_experiment(
            self.regression_experiment
        )

        self.assertEqual(
            len(result.y_true),
            self.regression_experiment.test_row_count,
        )

        self.assertEqual(
            len(result.y_pred),
            self.regression_experiment.test_row_count,
        )

        self.assertTrue(
            all(isinstance(value, float) for value in result.y_true)
        )

        self.assertTrue(
            all(isinstance(value, float) for value in result.y_pred)
        )

    def test_r2_is_none_when_test_set_has_one_row(self) -> None:
        dataset = LoadedDataset(
            path=Path("small_regression.csv"),
            columns=("Стойност", "Цена"),
            rows=(
                ("1", "10"),
                ("2", "20"),
                ("3", "30"),
                ("4", "40"),
                ("5", "50"),
            ),
        )

        experiment = train_experiment_model(
            dataset=dataset,
            target_column="Цена",
            feature_columns=("Стойност",),
            task_type="regression",
            algorithm="linear_regression",
            test_size=0.2,
            random_state=42,
        )

        result = evaluate_trained_experiment(experiment)

        self.assertEqual(len(result.y_true), 1)
        self.assertIsNone(result.metrics["r2"])

    def test_rejects_empty_test_data(self) -> None:
        transformed = replace(
            self.classification_experiment.transformed_data,
            X_test=np.empty(
                (0, self.classification_experiment.feature_count)
            ),
            y_test=np.array([]),
        )

        invalid_experiment = replace(
            self.classification_experiment,
            transformed_data=transformed,
        )

        with self.assertRaises(ModelEvaluationError):
            evaluate_trained_experiment(invalid_experiment)

    def test_rejects_mismatched_test_lengths(self) -> None:
        transformed = replace(
            self.classification_experiment.transformed_data,
            y_test=self.classification_experiment.transformed_data.y_test[:-1],
        )

        invalid_experiment = replace(
            self.classification_experiment,
            transformed_data=transformed,
        )

        with self.assertRaises(ModelEvaluationError):
            evaluate_trained_experiment(invalid_experiment)

    def test_rejects_invalid_test_dimensions(self) -> None:
        transformed = replace(
            self.classification_experiment.transformed_data,
            X_test=np.array([1, 2, 3]),
        )

        invalid_experiment = replace(
            self.classification_experiment,
            transformed_data=transformed,
        )

        with self.assertRaises(ModelEvaluationError):
            evaluate_trained_experiment(invalid_experiment)

    def test_rejects_non_finite_test_features(self) -> None:
        modified_test = (
            self.classification_experiment.transformed_data.X_test.copy()
        )
        modified_test[0, 0] = np.nan

        transformed = replace(
            self.classification_experiment.transformed_data,
            X_test=modified_test,
        )

        invalid_experiment = replace(
            self.classification_experiment,
            transformed_data=transformed,
        )

        with self.assertRaises(ModelEvaluationError):
            evaluate_trained_experiment(invalid_experiment)

    def test_rejects_invalid_task_type(self) -> None:
        invalid_experiment = replace(
            self.classification_experiment,
            task_type="clustering",
        )

        with self.assertRaises(ModelEvaluationError):
            evaluate_trained_experiment(invalid_experiment)

    def test_handles_prediction_error(self) -> None:
        estimator = Mock()
        estimator.predict.side_effect = ValueError(
            "Test prediction error"
        )

        invalid_experiment = replace(
            self.classification_experiment,
            estimator=estimator,
        )

        with self.assertRaises(ModelEvaluationError):
            evaluate_trained_experiment(invalid_experiment)


if __name__ == "__main__":
    unittest.main()