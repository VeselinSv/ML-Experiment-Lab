"""Тестове за предварителната обработка на ML данни."""

import unittest
from pathlib import Path

import numpy as np

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.preprocessing import (
    PreprocessingError,
    prepare_experiment_data,
)


class PreprocessingTests(unittest.TestCase):
    """Тестове за подготовката на входни и целеви данни."""

    def setUp(self) -> None:
        self.dataset = LoadedDataset(
            path=Path("sample.csv"),
            columns=("Възраст", "Цвят", "Клас"),
            rows=(
                ("20", "червено", "да"),
                ("", "синьо", "не"),
                ("30", "", "да"),
                ("40", "зелено", ""),
            ),
        )

    def test_detects_numeric_and_categorical_columns(self) -> None:
        prepared = prepare_experiment_data(
            self.dataset,
            target_column="Клас",
            feature_columns=("Възраст", "Цвят"),
        )

        self.assertEqual(
            prepared.numeric_columns,
            ("Възраст",),
        )

        self.assertEqual(
            prepared.categorical_columns,
            ("Цвят",),
        )

    def test_excludes_rows_with_missing_target(self) -> None:
        prepared = prepare_experiment_data(
            self.dataset,
            target_column="Клас",
            feature_columns=("Възраст", "Цвят"),
        )

        self.assertEqual(prepared.excluded_rows, 1)
        self.assertEqual(prepared.X.shape, (3, 2))
        self.assertEqual(prepared.y.shape, (3,))

        self.assertEqual(
            prepared.y.tolist(),
            ["да", "не", "да"],
        )

    def test_pipeline_handles_missing_values(self) -> None:
        prepared = prepare_experiment_data(
            self.dataset,
            target_column="Клас",
            feature_columns=("Възраст", "Цвят"),
        )

        transformed = prepared.preprocessor.fit_transform(
            prepared.X
        )

        self.assertEqual(transformed.shape[0], 3)
        self.assertTrue(np.isfinite(transformed).all())

    def test_pipeline_handles_unknown_categories(self) -> None:
        prepared = prepare_experiment_data(
            self.dataset,
            target_column="Клас",
            feature_columns=("Възраст", "Цвят"),
        )

        training_data = prepared.X[:2]
        test_data = prepared.X[2:3].copy()

        test_data[0, 1] = "лилаво"

        prepared.preprocessor.fit(training_data)

        transformed = prepared.preprocessor.transform(test_data)

        self.assertEqual(transformed.shape[0], 1)
        self.assertEqual(
            transformed.shape[1],
            prepared.preprocessor.transform(training_data).shape[1],
        )

    def test_rejects_empty_feature_selection(self) -> None:
        with self.assertRaises(PreprocessingError):
            prepare_experiment_data(
                self.dataset,
                target_column="Клас",
                feature_columns=(),
            )

    def test_rejects_target_as_feature(self) -> None:
        with self.assertRaises(PreprocessingError):
            prepare_experiment_data(
                self.dataset,
                target_column="Клас",
                feature_columns=("Възраст", "Клас"),
            )

    def test_rejects_unknown_feature(self) -> None:
        with self.assertRaises(PreprocessingError):
            prepare_experiment_data(
                self.dataset,
                target_column="Клас",
                feature_columns=("Неизвестна колона",),
            )

    def test_rejects_duplicate_features(self) -> None:
        with self.assertRaises(PreprocessingError):
            prepare_experiment_data(
                self.dataset,
                target_column="Клас",
                feature_columns=("Възраст", "Възраст"),
            )

    def test_rejects_dataset_without_target_values(self) -> None:
        dataset = LoadedDataset(
            path=Path("empty_target.csv"),
            columns=("Възраст", "Клас"),
            rows=(
                ("20", ""),
                ("30", " "),
            ),
        )

        with self.assertRaises(PreprocessingError):
            prepare_experiment_data(
                dataset,
                target_column="Клас",
                feature_columns=("Възраст",),
            )


if __name__ == "__main__":
    unittest.main()