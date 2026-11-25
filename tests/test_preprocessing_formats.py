"""Тестове за разширеното разпознаване на типовете данни."""

import unittest
from pathlib import Path

import numpy as np

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.preprocessing import prepare_experiment_data


class PreprocessingFormatsTests(unittest.TestCase):
    """Проверява числовите формати и типовете колони."""

    def _create_dataset(
        self,
        columns: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> LoadedDataset:
        """Създава примерен набор от данни."""

        return LoadedDataset(
            path=Path("test_dataset.xlsx"),
            columns=columns,
            rows=rows,
        )

    def test_decimal_comma(self):
        """Проверява числа с десетична запетая."""

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=(
                ("25,50", "Да"),
                ("30,75", "Не"),
                ("40,25", "Да"),
            ),
        )

        result = prepare_experiment_data(
            dataset,
            target_column="Target",
            feature_columns=("Value",),
        )

        self.assertEqual(
            result.numeric_columns,
            ("Value",),
        )

        np.testing.assert_allclose(
            result.X[:, 0].astype(float),
            [25.50, 30.75, 40.25],
        )

    def test_negative_decimal_comma(self):
        """Проверява отрицателни десетични числа."""

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=(
                ("-15,75", "Да"),
                ("-20,50", "Не"),
                ("10,25", "Да"),
            ),
        )

        result = prepare_experiment_data(
            dataset,
            target_column="Target",
            feature_columns=("Value",),
        )

        self.assertEqual(
            result.numeric_columns,
            ("Value",),
        )

        np.testing.assert_allclose(
            result.X[:, 0].astype(float),
            [-15.75, -20.50, 10.25],
        )

    def test_mixed_decimal_separators(self):
        """Проверява десетична точка и запетая в една колона."""

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=(
                ("10.5", "Да"),
                ("20,5", "Не"),
                ("30", "Да"),
            ),
        )

        result = prepare_experiment_data(
            dataset,
            target_column="Target",
            feature_columns=("Value",),
        )

        self.assertEqual(
            result.numeric_columns,
            ("Value",),
        )

        self.assertEqual(result.mixed_columns, ())

        np.testing.assert_allclose(
            result.X[:, 0].astype(float),
            [10.5, 20.5, 30.0],
        )

    def test_scientific_notation(self):
        """Проверява числа в научна нотация."""

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=(
                ("1e3", "Да"),
                ("2.5e2", "Не"),
                ("1,5e2", "Да"),
            ),
        )

        result = prepare_experiment_data(
            dataset,
            target_column="Target",
            feature_columns=("Value",),
        )

        self.assertEqual(
            result.numeric_columns,
            ("Value",),
        )

        np.testing.assert_allclose(
            result.X[:, 0].astype(float),
            [1000.0, 250.0, 150.0],
        )

    def test_mixed_column_detection(self):
        """Проверява разпознаването на смесена колона."""

        dataset = self._create_dataset(
            columns=("Income", "Target"),
            rows=(
                ("2000", "Да"),
                ("Неизвестен", "Не"),
                ("3500", "Да"),
            ),
        )

        result = prepare_experiment_data(
            dataset,
            target_column="Target",
            feature_columns=("Income",),
        )

        self.assertEqual(
            result.mixed_columns,
            ("Income",),
        )

        self.assertEqual(
            result.categorical_columns,
            ("Income",),
        )

        self.assertEqual(result.numeric_columns, ())

    def test_empty_column_detection(self):
        """Проверява разпознаването на изцяло празна колона."""

        dataset = self._create_dataset(
            columns=("Notes", "Target"),
            rows=(
                ("", "Да"),
                ("", "Не"),
                ("   ", "Да"),
            ),
        )

        result = prepare_experiment_data(
            dataset,
            target_column="Target",
            feature_columns=("Notes",),
        )

        self.assertEqual(
            result.empty_columns,
            ("Notes",),
        )

        self.assertEqual(
            result.categorical_columns,
            ("Notes",),
        )

        self.assertEqual(result.mixed_columns, ())

    def test_categorical_column_detection(self):
        """Проверява разпознаването на текстова колона."""

        dataset = self._create_dataset(
            columns=("City", "Target"),
            rows=(
                ("София", "Да"),
                ("Варна", "Не"),
                ("Пловдив", "Да"),
            ),
        )

        result = prepare_experiment_data(
            dataset,
            target_column="Target",
            feature_columns=("City",),
        )

        self.assertEqual(
            result.categorical_columns,
            ("City",),
        )

        self.assertEqual(result.numeric_columns, ())
        self.assertEqual(result.mixed_columns, ())
        self.assertEqual(result.empty_columns, ())

    def test_missing_numeric_values(self):
        """Проверява липсващи стойности в числова колона."""

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=(
                ("25,50", "Да"),
                ("", "Не"),
                ("40,25", "Да"),
            ),
        )

        result = prepare_experiment_data(
            dataset,
            target_column="Target",
            feature_columns=("Value",),
        )

        self.assertEqual(
            result.numeric_columns,
            ("Value",),
        )

        self.assertTrue(np.isnan(result.X[1, 0]))

        self.assertEqual(result.excluded_rows, 0)

    def test_original_dataset_is_preserved(self):
        """Проверява дали оригиналните данни остават непроменени."""

        original_rows = (
            ("25,50", "Да"),
            ("", "Не"),
            ("40,25", "Да"),
        )

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=original_rows,
        )

        result = prepare_experiment_data(
            dataset,
            target_column="Target",
            feature_columns=("Value",),
        )

        result.preprocessor.fit_transform(result.X)

        self.assertEqual(dataset.rows, original_rows)

        self.assertEqual(dataset.rows[0][0], "25,50")
        self.assertEqual(dataset.rows[1][0], "")

    def test_preprocessor_transforms_new_formats(self):
        """Проверява успешната трансформация на новите формати."""

        dataset = self._create_dataset(
            columns=("Age", "City", "Target"),
            rows=(
                ("20,5", "София", "Да"),
                ("30.5", "Варна", "Не"),
                ("", "София", "Да"),
                ("40,5", "", "Не"),
            ),
        )

        result = prepare_experiment_data(
            dataset,
            target_column="Target",
            feature_columns=("Age", "City"),
        )

        transformed = result.preprocessor.fit_transform(
            result.X
        )

        self.assertEqual(transformed.shape, (4, 4))

        self.assertTrue(
            np.isfinite(transformed).all()
        )

        # Медианата на 20.5, 30.5 и 40.5 е 30.5.
        # След попълването и стандартизацията
        # липсващата стойност трябва да стане 0.
        self.assertAlmostEqual(
            transformed[2, 0],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()