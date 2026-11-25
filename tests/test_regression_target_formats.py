"""Тестове за числовите формати в целевата колона."""

import unittest
from pathlib import Path

import numpy as np

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.training import (
    ModelTrainingError,
    train_experiment_model,
)


class RegressionTargetFormatsTests(unittest.TestCase):
    """Проверява обработката на Target при ML експерименти."""

    def _create_dataset(
        self,
        target_values: tuple[str, ...],
    ) -> LoadedDataset:
        """Създава примерен набор от данни."""

        return LoadedDataset(
            path=Path("target_formats.xlsx"),
            columns=("Value", "Target"),
            rows=tuple(
                (str(index), target)
                for index, target in enumerate(
                    target_values,
                    start=1,
                )
            ),
        )

    def _train_regression(
        self,
        target_values: tuple[str, ...],
    ):
        """Обучава модел за линейна регресия."""

        dataset = self._create_dataset(
            target_values
        )

        return train_experiment_model(
            dataset=dataset,
            target_column="Target",
            feature_columns=("Value",),
            task_type="regression",
            algorithm="linear_regression",
            test_size=0.2,
            random_state=42,
        )

    def _assert_target_values(
        self,
        experiment,
        expected_values: list[float],
    ) -> None:
        """Проверява преобразуваните Training и Test стойности."""

        actual_values = np.concatenate(
            (
                experiment.transformed_data.y_train,
                experiment.transformed_data.y_test,
            )
        )

        self.assertTrue(
            np.issubdtype(
                actual_values.dtype,
                np.floating,
            )
        )

        self.assertTrue(
            np.isfinite(actual_values).all()
        )

        np.testing.assert_allclose(
            np.sort(actual_values),
            np.sort(expected_values),
        )

    def test_decimal_comma_target(self):
        """Проверява Target с десетична запетая."""

        experiment = self._train_regression(
            (
                "10,5",
                "20,5",
                "30,5",
                "40,5",
                "50,5",
            )
        )

        self._assert_target_values(
            experiment,
            [10.5, 20.5, 30.5, 40.5, 50.5],
        )

    def test_mixed_decimal_separators(self):
        """Проверява десетична точка и запетая."""

        experiment = self._train_regression(
            (
                "10,5",
                "20.5",
                "30",
                "40,75",
                "50.25",
            )
        )

        self._assert_target_values(
            experiment,
            [10.5, 20.5, 30.0, 40.75, 50.25],
        )

    def test_negative_target_values(self):
        """Проверява отрицателни числови стойности."""

        experiment = self._train_regression(
            (
                "-50,5",
                "-25.75",
                "-10",
                "0",
                "15,25",
            )
        )

        self._assert_target_values(
            experiment,
            [-50.5, -25.75, -10.0, 0.0, 15.25],
        )

    def test_scientific_notation_target(self):
        """Проверява научна нотация в Target."""

        experiment = self._train_regression(
            (
                "1e3",
                "1,5e3",
                "2.5e2",
                "3e-1",
                "-2,5e2",
            )
        )

        self._assert_target_values(
            experiment,
            [1000.0, 1500.0, 250.0, 0.3, -250.0],
        )

    def test_missing_target_rows_are_excluded(self):
        """Проверява изключването на липсващи Target стойности."""

        experiment = self._train_regression(
            (
                "10,5",
                "",
                "20,5",
                "   ",
                "30,5",
                "40,5",
                "50,5",
            )
        )

        self.assertEqual(
            experiment.prepared_data.excluded_rows,
            2,
        )

        self.assertEqual(
            experiment.training_row_count
            + experiment.test_row_count,
            5,
        )

        self._assert_target_values(
            experiment,
            [10.5, 20.5, 30.5, 40.5, 50.5],
        )

    def test_invalid_text_target_is_rejected(self):
        """Проверява отхвърлянето на текст при регресия."""

        with self.assertRaisesRegex(
            ModelTrainingError,
            "числови стойности",
        ):
            self._train_regression(
                (
                    "10,5",
                    "20,5",
                    "Неизвестен",
                    "40,5",
                    "50,5",
                )
            )

    def test_non_finite_target_is_rejected(self):
        """Проверява отхвърлянето на NaN и Infinity."""

        invalid_values = (
            "NaN",
            "Infinity",
            "-Infinity",
            "inf",
            "-inf",
            "1e999",
        )

        for invalid_value in invalid_values:
            with self.subTest(value=invalid_value):
                with self.assertRaisesRegex(
                    ModelTrainingError,
                    "невалидни числови стойности",
                ):
                    self._train_regression(
                        (
                            "10,5",
                            "20,5",
                            invalid_value,
                            "40,5",
                            "50,5",
                        )
                    )

    def test_classification_preserves_original_labels(self):
        """Проверява запазването на категориите."""

        target_values = (
            "1,5",
            "1.5",
        ) * 10

        dataset = self._create_dataset(
            target_values
        )

        experiment = train_experiment_model(
            dataset=dataset,
            target_column="Target",
            feature_columns=("Value",),
            task_type="classification",
            algorithm="decision_tree",
            test_size=0.2,
            random_state=42,
        )

        actual_values = np.concatenate(
            (
                experiment.transformed_data.y_train,
                experiment.transformed_data.y_test,
            )
        )

        self.assertEqual(
            set(actual_values.tolist()),
            {"1,5", "1.5"},
        )

        self.assertEqual(
            set(experiment.estimator.classes_),
            {"1,5", "1.5"},
        )


if __name__ == "__main__":
    unittest.main()