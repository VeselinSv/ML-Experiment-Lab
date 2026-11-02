"""Тестове за прилагане на предварителната обработка."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.preprocessing import prepare_experiment_data
from ml_experiment_lab.splitting import (
    ExperimentDataSplit,
    split_experiment_data,
)
from ml_experiment_lab.transformation import (
    TransformationError,
    transform_experiment_split,
)


class ExperimentTransformationTests(unittest.TestCase):
    """Тестове за трансформиране на Training и Test данни."""

    def setUp(self) -> None:
        dataset = LoadedDataset(
            path=Path("sample.csv"),
            columns=("Възраст", "Цвят", "Клас"),
            rows=(
                ("20", "червено", "A"),
                ("25", "синьо", "B"),
                ("", "зелено", "A"),
                ("40", "", "B"),
                ("35", "червено", "A"),
                ("30", "синьо", "B"),
                ("45", "зелено", "A"),
                ("50", "червено", "B"),
            ),
        )

        self.prepared_data = prepare_experiment_data(
            dataset=dataset,
            target_column="Клас",
            feature_columns=("Възраст", "Цвят"),
        )

        self.data_split = split_experiment_data(
            self.prepared_data,
            test_size=0.25,
            random_state=42,
        )

    def test_transforms_training_and_test_data(self) -> None:
        result = transform_experiment_split(
            self.prepared_data,
            self.data_split,
        )

        self.assertEqual(
            result.train_row_count,
            self.data_split.train_row_count,
        )

        self.assertEqual(
            result.test_row_count,
            self.data_split.test_row_count,
        )

        self.assertTrue(
            np.isfinite(result.X_train).all()
        )

        self.assertTrue(
            np.isfinite(result.X_test).all()
        )

    def test_transformed_data_has_numeric_values(self) -> None:
        result = transform_experiment_split(
            self.prepared_data,
            self.data_split,
        )

        self.assertTrue(
            np.issubdtype(result.X_train.dtype, np.number)
        )

        self.assertTrue(
            np.issubdtype(result.X_test.dtype, np.number)
        )

    def test_training_and_test_have_same_feature_count(self) -> None:
        result = transform_experiment_split(
            self.prepared_data,
            self.data_split,
        )

        self.assertEqual(
            result.X_train.shape[1],
            result.X_test.shape[1],
        )

        self.assertEqual(
            result.feature_count,
            len(result.feature_names),
        )

    def test_target_values_are_preserved(self) -> None:
        result = transform_experiment_split(
            self.prepared_data,
            self.data_split,
        )

        np.testing.assert_array_equal(
            result.y_train,
            self.data_split.y_train,
        )

        np.testing.assert_array_equal(
            result.y_test,
            self.data_split.y_test,
        )

    def test_returns_fitted_preprocessor(self) -> None:
        result = transform_experiment_split(
            self.prepared_data,
            self.data_split,
        )

        self.assertIsNotNone(
            result.fitted_preprocessor
        )

        self.assertGreater(
            result.feature_count,
            0,
        )

    def test_original_preprocessor_remains_unfitted(self) -> None:
        original_numeric_pipeline = next(
            transformer
            for name, transformer, _ in (
                self.prepared_data.preprocessor.transformers
            )
            if name == "numeric"
        )

        result = transform_experiment_split(
            self.prepared_data,
            self.data_split,
        )

        original_scaler = (
            original_numeric_pipeline.named_steps["scaler"]
        )

        fitted_scaler = (
            result.fitted_preprocessor
            .named_transformers_["numeric"]
            .named_steps["scaler"]
        )

        self.assertFalse(
            hasattr(original_scaler, "mean_")
        )

        self.assertTrue(
            hasattr(fitted_scaler, "mean_")
        )

    def test_test_data_does_not_influence_scaler(self) -> None:
        dataset = LoadedDataset(
            path=Path("numeric.csv"),
            columns=("Стойност", "Клас"),
            rows=(
                ("1", "A"),
                ("2", "B"),
                ("3", "A"),
            ),
        )

        prepared = prepare_experiment_data(
            dataset=dataset,
            target_column="Клас",
            feature_columns=("Стойност",),
        )

        split = ExperimentDataSplit(
            X_train=np.array([[1.0], [3.0]], dtype=object),
            X_test=np.array([[999.0]], dtype=object),
            y_train=np.array(["A", "B"]),
            y_test=np.array(["A"]),
            test_size=1 / 3,
            random_state=42,
        )

        result = transform_experiment_split(
            prepared,
            split,
        )

        scaler = (
            result.fitted_preprocessor
            .named_transformers_["numeric"]
            .named_steps["scaler"]
        )

        # Средната стойност трябва да бъде (1 + 3) / 2 = 2.
        # Стойността 999 от Test не трябва да участва.
        self.assertAlmostEqual(
            scaler.mean_[0],
            2.0,
        )

    def test_unknown_category_is_handled(self) -> None:
        modified_test = self.data_split.X_test.copy()
        modified_test[0, 1] = "непозната_категория"

        modified_split = replace(
            self.data_split,
            X_test=modified_test,
        )

        result = transform_experiment_split(
            self.prepared_data,
            modified_split,
        )

        self.assertTrue(
            np.isfinite(result.X_test).all()
        )

        self.assertEqual(
            result.X_train.shape[1],
            result.X_test.shape[1],
        )

    def test_rejects_incorrect_feature_count(self) -> None:
        invalid_split = replace(
            self.data_split,
            X_test=self.data_split.X_test[:, :1],
        )

        with self.assertRaises(TransformationError):
            transform_experiment_split(
                self.prepared_data,
                invalid_split,
            )

    def test_rejects_empty_training_data(self) -> None:
        invalid_split = replace(
            self.data_split,
            X_train=np.empty((0, 2)),
            y_train=np.array([]),
        )

        with self.assertRaises(TransformationError):
            transform_experiment_split(
                self.prepared_data,
                invalid_split,
            )

    def test_rejects_empty_test_data(self) -> None:
        invalid_split = replace(
            self.data_split,
            X_test=np.empty((0, 2)),
            y_test=np.array([]),
        )

        with self.assertRaises(TransformationError):
            transform_experiment_split(
                self.prepared_data,
                invalid_split,
            )

    def test_rejects_mismatched_target_lengths(self) -> None:
        invalid_split = replace(
            self.data_split,
            y_test=self.data_split.y_test[:-1],
        )

        with self.assertRaises(TransformationError):
            transform_experiment_split(
                self.prepared_data,
                invalid_split,
            )

    def test_rejects_invalid_input_dimensions(self) -> None:
        invalid_split = replace(
            self.data_split,
            X_train=np.array([1, 2, 3]),
        )

        with self.assertRaises(TransformationError):
            transform_experiment_split(
                self.prepared_data,
                invalid_split,
            )

    def test_rejects_invalid_target_dimensions(self) -> None:
        invalid_split = replace(
            self.data_split,
            y_train=self.data_split.y_train.reshape(-1, 1),
        )

        with self.assertRaises(TransformationError):
            transform_experiment_split(
                self.prepared_data,
                invalid_split,
            )


if __name__ == "__main__":
    unittest.main()