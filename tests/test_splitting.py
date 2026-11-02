"""Тестове за разделяне на ML данни."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.preprocessing import prepare_experiment_data
from ml_experiment_lab.splitting import (
    DataSplitError,
    split_experiment_data,
)


class ExperimentDataSplitTests(unittest.TestCase):
    """Тестове за Training/Test разделянето."""

    def setUp(self) -> None:
        rows = tuple(
            (
                str(index),
                "A" if index % 2 == 0 else "B",
            )
            for index in range(20)
        )

        dataset = LoadedDataset(
            path=Path("sample.csv"),
            columns=("Стойност", "Клас"),
            rows=rows,
        )

        self.prepared_data = prepare_experiment_data(
            dataset=dataset,
            target_column="Клас",
            feature_columns=("Стойност",),
        )

    def test_splits_data_into_training_and_test_sets(self) -> None:
        result = split_experiment_data(
            self.prepared_data,
            test_size=0.2,
            random_state=42,
        )

        self.assertEqual(result.train_row_count, 16)
        self.assertEqual(result.test_row_count, 4)
        self.assertEqual(len(result.y_train), 16)
        self.assertEqual(len(result.y_test), 4)

    def test_training_and_test_sets_are_not_empty(self) -> None:
        result = split_experiment_data(self.prepared_data)

        self.assertGreater(result.train_row_count, 0)
        self.assertGreater(result.test_row_count, 0)

    def test_same_random_state_produces_same_split(self) -> None:
        first = split_experiment_data(
            self.prepared_data,
            random_state=42,
        )

        second = split_experiment_data(
            self.prepared_data,
            random_state=42,
        )

        np.testing.assert_array_equal(
            first.X_train,
            second.X_train,
        )
        np.testing.assert_array_equal(
            first.X_test,
            second.X_test,
        )
        np.testing.assert_array_equal(
            first.y_train,
            second.y_train,
        )
        np.testing.assert_array_equal(
            first.y_test,
            second.y_test,
        )

    def test_different_test_size_changes_split_dimensions(self) -> None:
        result = split_experiment_data(
            self.prepared_data,
            test_size=0.3,
            random_state=42,
        )

        self.assertEqual(result.train_row_count, 14)
        self.assertEqual(result.test_row_count, 6)

    def test_split_preserves_all_rows(self) -> None:
        result = split_experiment_data(
            self.prepared_data,
            random_state=42,
        )

        combined_X = np.concatenate(
            (result.X_train, result.X_test),
            axis=0,
        )

        combined_y = np.concatenate(
            (result.y_train, result.y_test),
            axis=0,
        )

        self.assertEqual(len(combined_X), len(self.prepared_data.X))
        self.assertEqual(len(combined_y), len(self.prepared_data.y))

        self.assertEqual(
            set(combined_X[:, 0]),
            set(self.prepared_data.X[:, 0]),
        )

    def test_invalid_test_size_is_rejected(self) -> None:
        invalid_sizes = (
            0,
            1,
            -0.1,
            1.1,
            float("nan"),
            float("inf"),
            True,
            "0.2",
        )

        for invalid_size in invalid_sizes:
            with self.subTest(test_size=invalid_size):
                with self.assertRaises(DataSplitError):
                    split_experiment_data(
                        self.prepared_data,
                        test_size=invalid_size,
                    )

    def test_invalid_random_state_is_rejected(self) -> None:
        invalid_states = (True, 42.5, "42", None)

        for invalid_state in invalid_states:
            with self.subTest(random_state=invalid_state):
                with self.assertRaises(DataSplitError):
                    split_experiment_data(
                        self.prepared_data,
                        random_state=invalid_state,
                    )

    def test_dataset_with_one_row_is_rejected(self) -> None:
        dataset = LoadedDataset(
            path=Path("one_row.csv"),
            columns=("Стойност", "Клас"),
            rows=(("10", "A"),),
        )

        prepared = prepare_experiment_data(
            dataset=dataset,
            target_column="Клас",
            feature_columns=("Стойност",),
        )

        with self.assertRaises(DataSplitError):
            split_experiment_data(prepared)

    def test_mismatched_input_and_target_lengths_are_rejected(self) -> None:
        invalid_data = replace(
            self.prepared_data,
            y=self.prepared_data.y[:-1],
        )

        with self.assertRaises(DataSplitError):
            split_experiment_data(invalid_data)

    def test_invalid_input_dimensions_are_rejected(self) -> None:
        invalid_data = replace(
            self.prepared_data,
            X=np.array([1, 2, 3]),
        )

        with self.assertRaises(DataSplitError):
            split_experiment_data(invalid_data)


if __name__ == "__main__":
    unittest.main()