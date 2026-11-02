"""Тестове за обучението на ML модели."""

import unittest
from pathlib import Path

import numpy as np

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.training import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    ModelTrainingError,
    get_available_algorithms,
    train_experiment_model,
)


class ModelTrainingTests(unittest.TestCase):
    """Тестове за класификация и регресия."""

    def setUp(self) -> None:
        classification_rows = tuple(
            (
                str(18 + index),
                "група_A" if index % 2 == 0 else "група_B",
                "приет" if index % 2 == 0 else "отхвърлен",
            )
            for index in range(60)
        )

        self.classification_dataset = LoadedDataset(
            path=Path("classification.csv"),
            columns=("Възраст", "Група", "Резултат"),
            rows=classification_rows,
        )

        regression_rows = tuple(
            (
                str(index + 1),
                "група_A" if index % 2 == 0 else "група_B",
                str(100 + (index + 1) * 2),
            )
            for index in range(60)
        )

        self.regression_dataset = LoadedDataset(
            path=Path("regression.csv"),
            columns=("Стойност", "Група", "Цена"),
            rows=regression_rows,
        )

    def test_available_classification_algorithms(self) -> None:
        algorithms = dict(
            get_available_algorithms("classification")
        )

        self.assertEqual(
            algorithms,
            CLASSIFICATION_ALGORITHMS,
        )

    def test_available_regression_algorithms(self) -> None:
        algorithms = dict(
            get_available_algorithms("regression")
        )

        self.assertEqual(
            algorithms,
            REGRESSION_ALGORITHMS,
        )

    def test_unsupported_task_type_is_rejected(self) -> None:
        with self.assertRaises(ModelTrainingError):
            get_available_algorithms("clustering")

    def test_trains_classification_model(self) -> None:
        result = train_experiment_model(
            dataset=self.classification_dataset,
            target_column="Резултат",
            feature_columns=("Възраст", "Група"),
            task_type="classification",
            algorithm="logistic_regression",
            test_size=0.2,
            random_state=42,
        )

        self.assertEqual(result.task_type, "classification")
        self.assertEqual(
            result.algorithm,
            "logistic_regression",
        )
        self.assertEqual(result.target_column, "Резултат")

        self.assertEqual(
            result.training_row_count,
            48,
        )
        self.assertEqual(
            result.test_row_count,
            12,
        )

        predictions = result.estimator.predict(
            result.transformed_data.X_test
        )

        self.assertEqual(len(predictions), 12)

        self.assertTrue(
            set(predictions).issubset(
                {"приет", "отхвърлен"}
            )
        )

    def test_trains_regression_model(self) -> None:
        result = train_experiment_model(
            dataset=self.regression_dataset,
            target_column="Цена",
            feature_columns=("Стойност", "Група"),
            task_type="regression",
            algorithm="linear_regression",
            test_size=0.2,
            random_state=42,
        )

        self.assertEqual(result.task_type, "regression")
        self.assertEqual(
            result.algorithm,
            "linear_regression",
        )

        self.assertEqual(
            result.training_row_count,
            48,
        )
        self.assertEqual(
            result.test_row_count,
            12,
        )

        self.assertTrue(
            np.issubdtype(
                result.transformed_data.y_train.dtype,
                np.number,
            )
        )

        predictions = result.estimator.predict(
            result.transformed_data.X_test
        )

        self.assertEqual(len(predictions), 12)

        self.assertTrue(
            np.isfinite(predictions).all()
        )

    def test_all_classification_algorithms_can_be_trained(
        self,
    ) -> None:
        for algorithm in CLASSIFICATION_ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                result = train_experiment_model(
                    dataset=self.classification_dataset,
                    target_column="Резултат",
                    feature_columns=("Възраст", "Група"),
                    task_type="classification",
                    algorithm=algorithm,
                    test_size=0.2,
                    random_state=42,
                )

                self.assertIsNotNone(result.estimator)

                predictions = result.estimator.predict(
                    result.transformed_data.X_test
                )

                self.assertEqual(len(predictions), 12)

    def test_all_regression_algorithms_can_be_trained(
        self,
    ) -> None:
        for algorithm in REGRESSION_ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                result = train_experiment_model(
                    dataset=self.regression_dataset,
                    target_column="Цена",
                    feature_columns=("Стойност", "Група"),
                    task_type="regression",
                    algorithm=algorithm,
                    test_size=0.2,
                    random_state=42,
                )

                self.assertIsNotNone(result.estimator)

                predictions = result.estimator.predict(
                    result.transformed_data.X_test
                )

                self.assertEqual(len(predictions), 12)

                self.assertTrue(
                    np.isfinite(predictions).all()
                )

    def test_rejects_algorithm_for_wrong_task_type(self) -> None:
        with self.assertRaises(ModelTrainingError):
            train_experiment_model(
                dataset=self.classification_dataset,
                target_column="Резултат",
                feature_columns=("Възраст",),
                task_type="classification",
                algorithm="linear_regression",
            )

    def test_rejects_non_numeric_regression_target(self) -> None:
        with self.assertRaises(ModelTrainingError):
            train_experiment_model(
                dataset=self.classification_dataset,
                target_column="Резултат",
                feature_columns=("Възраст", "Група"),
                task_type="regression",
                algorithm="linear_regression",
            )

    def test_rejects_classification_with_one_class(self) -> None:
        rows = tuple(
            (str(index), "единствен_клас")
            for index in range(30)
        )

        dataset = LoadedDataset(
            path=Path("one_class.csv"),
            columns=("Стойност", "Клас"),
            rows=rows,
        )

        with self.assertRaises(ModelTrainingError):
            train_experiment_model(
                dataset=dataset,
                target_column="Клас",
                feature_columns=("Стойност",),
                task_type="classification",
                algorithm="decision_tree",
            )

    def test_rejects_unknown_algorithm(self) -> None:
        with self.assertRaises(ModelTrainingError):
            train_experiment_model(
                dataset=self.classification_dataset,
                target_column="Резултат",
                feature_columns=("Възраст",),
                task_type="classification",
                algorithm="unknown_algorithm",
            )

    def test_rejects_unsupported_task_type(self) -> None:
        with self.assertRaises(ModelTrainingError):
            train_experiment_model(
                dataset=self.classification_dataset,
                target_column="Резултат",
                feature_columns=("Възраст",),
                task_type="clustering",
                algorithm="decision_tree",
            )


if __name__ == "__main__":
    unittest.main()