"""Тестове за предотвратяване на Data Leakage."""

import unittest
from pathlib import Path

import numpy as np

from ml_experiment_lab.datasets import LoadedDataset

from ml_experiment_lab.preprocessing import (
    prepare_experiment_data,
    with_training_feature_types,
)

from ml_experiment_lab.splitting import (
    ExperimentDataSplit,
)

from ml_experiment_lab.transformation import (
    transform_experiment_split,
)

from ml_experiment_lab.training import (
    train_experiment_model,
)


class PreprocessingLeakageTests(unittest.TestCase):
    """Проверява независимостта на Training и Test."""

    def _create_dataset(
        self,
        columns: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> LoadedDataset:
        """Създава примерен набор от данни."""

        return LoadedDataset(
            path=Path("leakage_test.xlsx"),
            columns=columns,
            rows=rows,
        )

    def _prepare_controlled_split(
        self,
        dataset: LoadedDataset,
        feature_columns: tuple[str, ...],
        train_count: int,
    ):
        """Създава контролирано разделяне за тестовете."""

        prepared = prepare_experiment_data(
            dataset=dataset,
            target_column="Target",
            feature_columns=feature_columns,
            defer_type_detection=True,
        )

        data_split = ExperimentDataSplit(
            X_train=prepared.X[:train_count],
            X_test=prepared.X[train_count:],
            y_train=prepared.y[:train_count],
            y_test=prepared.y[train_count:],
            test_size=(
                len(prepared.X) - train_count
            ) / len(prepared.X),
            random_state=42,
        )

        prepared = with_training_feature_types(
            prepared_data=prepared,
            X_train=data_split.X_train,
        )

        transformed = transform_experiment_split(
            prepared_data=prepared,
            data_split=data_split,
        )

        return prepared, transformed

    def test_deferred_type_detection(self):
        """Типовете не се определят преди разделянето."""

        dataset = self._create_dataset(
            columns=("Age", "City", "Target"),
            rows=(
                ("20,5", "Варна", "Да"),
                ("30", "София", "Не"),
                ("40", "", "Да"),
            ),
        )

        prepared = prepare_experiment_data(
            dataset=dataset,
            target_column="Target",
            feature_columns=("Age", "City"),
            defer_type_detection=True,
        )

        self.assertEqual(
            prepared.numeric_columns,
            (),
        )

        self.assertEqual(
            prepared.categorical_columns,
            (),
        )

        self.assertEqual(
            prepared.mixed_columns,
            (),
        )

        self.assertEqual(
            prepared.empty_columns,
            (),
        )

        # Числовите стойности все още са текст.
        self.assertEqual(
            prepared.X[0, 0],
            "20,5",
        )

        # Празните клетки са NaN.
        self.assertTrue(
            np.isnan(prepared.X[2, 1])
        )

    def test_test_data_does_not_change_feature_type(self):
        """Текстова стойност в Test не променя числовия тип."""

        dataset = self._create_dataset(
            columns=("Income", "Target"),
            rows=(
                ("1000", "Да"),
                ("2000", "Не"),
                ("3000", "Да"),
                ("4000", "Не"),
                ("Неизвестен", "Да"),
                ("5000", "Не"),
            ),
        )

        prepared, transformed = (
            self._prepare_controlled_split(
                dataset=dataset,
                feature_columns=("Income",),
                train_count=4,
            )
        )

        self.assertEqual(
            prepared.numeric_columns,
            ("Income",),
        )

        self.assertEqual(
            prepared.categorical_columns,
            (),
        )

        self.assertEqual(
            prepared.mixed_columns,
            (),
        )

        self.assertEqual(
            transformed.feature_count,
            1,
        )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

    def test_numeric_imputer_uses_training_median(self):
        """Медианата се изчислява само от Training."""

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=(
                ("10", "Да"),
                ("20", "Не"),
                ("30", "Да"),
                ("", "Не"),
                ("10000", "Да"),
                ("", "Не"),
                ("Неизвестен", "Да"),
            ),
        )

        _, transformed = (
            self._prepare_controlled_split(
                dataset=dataset,
                feature_columns=("Value",),
                train_count=4,
            )
        )

        numeric_pipeline = (
            transformed.fitted_preprocessor
            .named_transformers_["numeric"]
        )

        imputer = numeric_pipeline.named_steps["imputer"]
        scaler = numeric_pipeline.named_steps["scaler"]

        # Медианата на 10, 20 и 30 е 20.
        self.assertAlmostEqual(
            imputer.statistics_[0],
            20.0,
        )

        # Средната стойност след попълването е 20.
        self.assertAlmostEqual(
            scaler.mean_[0],
            20.0,
        )

        # Празната стойност от Test.
        self.assertAlmostEqual(
            transformed.X_test[1, 0],
            0.0,
        )

        # Невалидната текстова стойност от Test.
        self.assertAlmostEqual(
            transformed.X_test[2, 0],
            0.0,
        )

    def test_unknown_test_category_is_not_learned(self):
        """Категория от Test не участва в OneHotEncoder."""

        dataset = self._create_dataset(
            columns=("City", "Target"),
            rows=(
                ("Варна", "Да"),
                ("София", "Не"),
                ("Варна", "Да"),
                ("София", "Не"),
                ("Пловдив", "Да"),
                ("Бургас", "Не"),
            ),
        )

        prepared, transformed = (
            self._prepare_controlled_split(
                dataset=dataset,
                feature_columns=("City",),
                train_count=4,
            )
        )

        self.assertEqual(
            prepared.categorical_columns,
            ("City",),
        )

        categorical_pipeline = (
            transformed.fitted_preprocessor
            .named_transformers_["categorical"]
        )

        encoder = (
            categorical_pipeline.named_steps["encoder"]
        )

        learned_categories = set(
            encoder.categories_[0]
        )

        self.assertEqual(
            learned_categories,
            {"Варна", "София"},
        )

        self.assertNotIn(
            "Пловдив",
            learned_categories,
        )

        self.assertNotIn(
            "Бургас",
            learned_categories,
        )

        # Непознатите категории се кодират с нули.
        np.testing.assert_array_equal(
            transformed.X_test,
            np.zeros((2, 2)),
        )

    def test_empty_training_column_detection(self):
        """Test стойностите не променят празния Training тип."""

        dataset = self._create_dataset(
            columns=("Notes", "Target"),
            rows=(
                ("", "Да"),
                ("", "Не"),
                ("", "Да"),
                ("", "Не"),
                ("100", "Да"),
                ("200", "Не"),
            ),
        )

        prepared, transformed = (
            self._prepare_controlled_split(
                dataset=dataset,
                feature_columns=("Notes",),
                train_count=4,
            )
        )

        self.assertEqual(
            prepared.empty_columns,
            ("Notes",),
        )

        self.assertEqual(
            prepared.numeric_columns,
            (),
        )

        self.assertEqual(
            prepared.categorical_columns,
            ("Notes",),
        )

        self.assertTrue(
            np.isfinite(transformed.X_train).all()
        )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

    def test_test_changes_do_not_affect_trained_model(self):
        """Промените само в Test не влияят на модела."""

        original_rows = tuple(
            (str(index * 10), str(index))
            for index in range(1, 21)
        )

        original_dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=original_rows,
        )

        original_experiment = train_experiment_model(
            dataset=original_dataset,
            target_column="Target",
            feature_columns=("Value",),
            task_type="regression",
            algorithm="linear_regression",
            test_size=0.2,
            random_state=42,
        )

        # Target е уникален за всеки ред.
        # Използваме го, за да установим кои редове
        # са попаднали в тестовата извадка.
        test_targets = set(
            original_experiment.data_split.y_test.tolist()
        )

        # Променяме само входните стойности в Test.
        modified_rows = tuple(
            (
                "Неизвестен"
                if target in test_targets
                else value,
                target,
            )
            for value, target in original_rows
        )

        modified_dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=modified_rows,
        )

        modified_experiment = train_experiment_model(
            dataset=modified_dataset,
            target_column="Target",
            feature_columns=("Value",),
            task_type="regression",
            algorithm="linear_regression",
            test_size=0.2,
            random_state=42,
        )

        # Разпознатите типове остават еднакви.
        self.assertEqual(
            original_experiment.prepared_data.numeric_columns,
            modified_experiment.prepared_data.numeric_columns,
        )

        self.assertEqual(
            modified_experiment.prepared_data.numeric_columns,
            ("Value",),
        )

        # Разделянето остава еднакво.
        np.testing.assert_array_equal(
            original_experiment.data_split.y_train,
            modified_experiment.data_split.y_train,
        )

        np.testing.assert_array_equal(
            original_experiment.data_split.y_test,
            modified_experiment.data_split.y_test,
        )

        # Обработените Training данни остават еднакви.
        np.testing.assert_allclose(
            original_experiment.transformed_data.X_train,
            modified_experiment.transformed_data.X_train,
        )

        # Имената на характеристиките остават еднакви.
        self.assertEqual(
            original_experiment.transformed_data.feature_names,
            modified_experiment.transformed_data.feature_names,
        )

        # Коефициентите на модела остават еднакви.
        np.testing.assert_allclose(
            original_experiment.estimator.coef_,
            modified_experiment.estimator.coef_,
        )

        self.assertAlmostEqual(
            original_experiment.estimator.intercept_,
            modified_experiment.estimator.intercept_,
        )


if __name__ == "__main__":
    unittest.main()