"""Тестове за конфигурируемата предварителна обработка."""

import unittest
from pathlib import Path

import numpy as np

from ml_experiment_lab.datasets import LoadedDataset

from ml_experiment_lab.preprocessing import (
    PreprocessingConfiguration,
    PreprocessingError,
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


class PreprocessingConfigurationTests(unittest.TestCase):
    """Проверява настройките за предварителна обработка."""

    def _create_dataset(
        self,
        columns: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> LoadedDataset:
        """Създава примерен набор от данни."""

        return LoadedDataset(
            path=Path("preprocessing_configuration.xlsx"),
            columns=columns,
            rows=rows,
        )

    def _transform_dataset(
        self,
        dataset: LoadedDataset,
        feature_columns: tuple[str, ...],
        train_count: int,
        configuration: PreprocessingConfiguration,
        *,
        defer_type_detection: bool = True,
    ):
        """Подготвя и трансформира контролирани извадки."""

        prepared = prepare_experiment_data(
            dataset=dataset,
            target_column="Target",
            feature_columns=feature_columns,
            defer_type_detection=defer_type_detection,
            preprocessing_configuration=configuration,
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

        if defer_type_detection:
            prepared = with_training_feature_types(
                prepared_data=prepared,
                X_train=data_split.X_train,
            )

        transformed = transform_experiment_split(
            prepared_data=prepared,
            data_split=data_split,
        )

        return prepared, transformed

    def _check_numeric_strategy(
        self,
        strategy: str,
        expected_value: float,
        fill_value: float = 0.0,
    ) -> None:
        """Проверява числова стратегия за попълване."""

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=(
                ("10", "Да"),
                ("20", "Не"),
                ("20", "Да"),
                ("100", "Не"),
                ("", "Да"),

                # Test данни.
                ("10000", "Не"),
                ("", "Да"),
                ("Неизвестен", "Не"),
            ),
        )

        configuration = PreprocessingConfiguration(
            numeric_imputation_strategy=strategy,
            numeric_fill_value=fill_value,
        )

        prepared, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Value",),
            train_count=5,
            configuration=configuration,
        )

        numeric_pipeline = (
            transformed.fitted_preprocessor
            .named_transformers_["numeric"]
        )

        imputer = numeric_pipeline.named_steps["imputer"]
        scaler = numeric_pipeline.named_steps["scaler"]

        self.assertEqual(
            imputer.strategy,
            strategy,
        )

        self.assertAlmostEqual(
            imputer.statistics_[0],
            expected_value,
        )

        # Възстановяваме стойностите преди стандартизация,
        # за да проверим действителното попълване.
        restored_train = scaler.inverse_transform(
            transformed.X_train
        )

        restored_test = scaler.inverse_transform(
            transformed.X_test
        )

        # Липсваща стойност в Training.
        self.assertAlmostEqual(
            restored_train[4, 0],
            expected_value,
        )

        # Липсваща стойност в Test.
        self.assertAlmostEqual(
            restored_test[1, 0],
            expected_value,
        )

        # Невалидна числова стойност в Test.
        self.assertAlmostEqual(
            restored_test[2, 0],
            expected_value,
        )

        # Голямата тестова стойност не трябва
        # да влияе върху статистиката за попълване.
        self.assertAlmostEqual(
            restored_test[0, 0],
            10000.0,
        )

        self.assertEqual(
            prepared.preprocessing_configuration,
            configuration,
        )

        self.assertTrue(
            np.isfinite(transformed.X_train).all()
        )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

    def test_default_configuration(self):
        """Проверява настройките по подразбиране."""

        configuration = PreprocessingConfiguration()

        self.assertEqual(
            configuration.numeric_imputation_strategy,
            "median",
        )

        self.assertEqual(
            configuration.categorical_imputation_strategy,
            "constant",
        )

        self.assertEqual(
            configuration.numeric_fill_value,
            0.0,
        )

        self.assertEqual(
            configuration.categorical_fill_value,
            "ЛИПСВАЩА_СТОЙНОСТ",
        )

    def test_numeric_mean_imputation(self):
        """Проверява попълване със средноаритметична стойност."""

        self._check_numeric_strategy(
            strategy="mean",
            expected_value=37.5,
        )

    def test_numeric_median_imputation(self):
        """Проверява попълване с медиана."""

        self._check_numeric_strategy(
            strategy="median",
            expected_value=20.0,
        )

    def test_numeric_most_frequent_imputation(self):
        """Проверява попълване с най-честото число."""

        self._check_numeric_strategy(
            strategy="most_frequent",
            expected_value=20.0,
        )

    def test_numeric_constant_imputation(self):
        """Проверява попълване с числова константа."""

        self._check_numeric_strategy(
            strategy="constant",
            expected_value=-99.0,
            fill_value=-99.0,
        )

    def test_categorical_most_frequent_imputation(self):
        """Проверява попълване с най-честата категория."""

        dataset = self._create_dataset(
            columns=("City", "Target"),
            rows=(
                ("Варна", "Да"),
                ("София", "Не"),
                ("Варна", "Да"),
                ("", "Не"),

                # Test данни.
                ("Пловдив", "Да"),
                ("", "Не"),
            ),
        )

        configuration = PreprocessingConfiguration(
            categorical_imputation_strategy="most_frequent",
        )

        _, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("City",),
            train_count=4,
            configuration=configuration,
        )

        categorical_pipeline = (
            transformed.fitted_preprocessor
            .named_transformers_["categorical"]
        )

        imputer = categorical_pipeline.named_steps[
            "imputer"
        ]

        encoder = categorical_pipeline.named_steps[
            "encoder"
        ]

        self.assertEqual(
            imputer.statistics_[0],
            "Варна",
        )

        self.assertEqual(
            set(encoder.categories_[0]),
            {"Варна", "София"},
        )

        # Проверяваме попълнената Training категория.
        restored_train = encoder.inverse_transform(
            transformed.X_train
        )

        self.assertEqual(
            restored_train[3, 0],
            "Варна",
        )

        # Проверяваме попълнената Test категория.
        restored_test = encoder.inverse_transform(
            transformed.X_test
        )

        self.assertEqual(
            restored_test[1, 0],
            "Варна",
        )

        # Непознатата Test категория се кодира с нули.
        np.testing.assert_array_equal(
            transformed.X_test[0],
            np.zeros(2),
        )

    def test_categorical_constant_imputation(self):
        """Проверява попълване с текстова константа."""

        dataset = self._create_dataset(
            columns=("City", "Target"),
            rows=(
                ("Варна", "Да"),
                ("София", "Не"),
                ("", "Да"),
                ("Варна", "Не"),

                # Test данни.
                ("", "Да"),
                ("Пловдив", "Не"),
            ),
        )

        configuration = PreprocessingConfiguration(
            categorical_imputation_strategy="constant",
            categorical_fill_value="Неизвестен",
        )

        _, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("City",),
            train_count=4,
            configuration=configuration,
        )

        categorical_pipeline = (
            transformed.fitted_preprocessor
            .named_transformers_["categorical"]
        )

        imputer = categorical_pipeline.named_steps[
            "imputer"
        ]

        encoder = categorical_pipeline.named_steps[
            "encoder"
        ]

        self.assertEqual(
            imputer.statistics_[0],
            "Неизвестен",
        )

        self.assertIn(
            "Неизвестен",
            encoder.categories_[0],
        )

        restored_train = encoder.inverse_transform(
            transformed.X_train
        )

        restored_test = encoder.inverse_transform(
            transformed.X_test
        )

        self.assertEqual(
            restored_train[2, 0],
            "Неизвестен",
        )

        self.assertEqual(
            restored_test[0, 0],
            "Неизвестен",
        )

    def test_empty_column_fallback(self):
        """Проверява резервната стратегия при празна колона."""

        dataset = self._create_dataset(
            columns=("City", "Notes", "Target"),
            rows=(
                ("Варна", "", "Да"),
                ("София", "", "Не"),
                ("Варна", "", "Да"),
                ("", "", "Не"),

                # Test данни.
                ("Пловдив", "Бележка", "Да"),
                ("", "", "Не"),
            ),
        )

        configuration = PreprocessingConfiguration(
            categorical_imputation_strategy="most_frequent",
            categorical_fill_value="Няма информация",
        )

        prepared, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("City", "Notes"),
            train_count=4,
            configuration=configuration,
        )

        self.assertEqual(
            prepared.empty_columns,
            ("Notes",),
        )

        transformers = (
            transformed.fitted_preprocessor
            .named_transformers_
        )

        normal_imputer = (
            transformers["categorical"]
            .named_steps["imputer"]
        )

        empty_imputer = (
            transformers["categorical_empty"]
            .named_steps["imputer"]
        )

        self.assertEqual(
            normal_imputer.strategy,
            "most_frequent",
        )

        self.assertEqual(
            normal_imputer.statistics_[0],
            "Варна",
        )

        self.assertEqual(
            empty_imputer.strategy,
            "constant",
        )

        self.assertEqual(
            empty_imputer.statistics_[0],
            "Няма информация",
        )

        self.assertEqual(
            transformed.feature_count,
            3,
        )

        self.assertTrue(
            np.isfinite(transformed.X_train).all()
        )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

        # Празната Test бележка се попълва
        # с научената резервна константа.
        self.assertEqual(
            transformed.X_test[1, -1],
            1.0,
        )

    def test_all_empty_training_features(self):
        """Проверява изцяло празна Training характеристика."""

        dataset = self._create_dataset(
            columns=("Notes", "Target"),
            rows=(
                ("", "Да"),
                ("", "Не"),
                ("", "Да"),
                ("", "Не"),

                # Test данни.
                ("Нова информация", "Да"),
                ("", "Не"),
            ),
        )

        configuration = PreprocessingConfiguration(
            categorical_imputation_strategy="most_frequent",
            categorical_fill_value="Липсва",
        )

        prepared, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Notes",),
            train_count=4,
            configuration=configuration,
        )

        self.assertEqual(
            prepared.empty_columns,
            ("Notes",),
        )

        self.assertEqual(
            transformed.feature_count,
            1,
        )

        np.testing.assert_array_equal(
            transformed.X_train,
            np.ones((4, 1)),
        )

        np.testing.assert_array_equal(
            transformed.X_test,
            np.array(
                [
                    [0.0],
                    [1.0],
                ]
            ),
        )

    def test_invalid_strategies(self):
        """Проверява отхвърляне на неподдържани стратегии."""

        invalid_configurations = (
            {
                "numeric_imputation_strategy": "unknown",
            },
            {
                "numeric_imputation_strategy": "mode",
            },
            {
                "categorical_imputation_strategy": "mean",
            },
            {
                "categorical_imputation_strategy": "median",
            },
        )

        for parameters in invalid_configurations:
            with self.subTest(parameters=parameters):
                with self.assertRaises(PreprocessingError):
                    PreprocessingConfiguration(
                        **parameters
                    )

    def test_invalid_fill_values(self):
        """Проверява отхвърляне на невалидни константи."""

        invalid_configurations = (
            {"numeric_fill_value": "abc"},
            {"numeric_fill_value": True},
            {"numeric_fill_value": None},
            {"numeric_fill_value": float("nan")},
            {"numeric_fill_value": float("inf")},
            {"numeric_fill_value": float("-inf")},
            {"categorical_fill_value": ""},
            {"categorical_fill_value": "   "},
            {"categorical_fill_value": 123},
            {"categorical_fill_value": None},
        )

        for parameters in invalid_configurations:
            with self.subTest(parameters=parameters):
                with self.assertRaises(PreprocessingError):
                    PreprocessingConfiguration(
                        **parameters
                    )

    def test_invalid_configuration_object(self):
        """Проверява отхвърляне на невалидна конфигурация."""

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=(
                ("10", "Да"),
                ("20", "Не"),
            ),
        )

        with self.assertRaises(PreprocessingError):
            prepare_experiment_data(
                dataset=dataset,
                target_column="Target",
                feature_columns=("Value",),
                preprocessing_configuration={
                    "numeric_imputation_strategy": "mean",
                },
            )

    def test_standard_preparation_mode(self):
        """Проверява съвместимостта със стандартния режим."""

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=(
                ("10,5", "Да"),
                ("20,5", "Не"),
                ("", "Да"),
                ("40,5", "Не"),
            ),
        )

        configuration = PreprocessingConfiguration(
            numeric_imputation_strategy="constant",
            numeric_fill_value=-1.0,
        )

        prepared, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Value",),
            train_count=3,
            configuration=configuration,
            defer_type_detection=False,
        )

        self.assertEqual(
            prepared.numeric_columns,
            ("Value",),
        )

        numeric_pipeline = (
            transformed.fitted_preprocessor
            .named_transformers_["numeric"]
        )

        imputer = numeric_pipeline.named_steps[
            "imputer"
        ]

        scaler = numeric_pipeline.named_steps[
            "scaler"
        ]

        self.assertAlmostEqual(
            imputer.statistics_[0],
            -1.0,
        )

        restored_train = scaler.inverse_transform(
            transformed.X_train
        )

        np.testing.assert_allclose(
            restored_train[:, 0],
            [10.5, 20.5, -1.0],
        )

    def test_training_uses_preprocessing_configuration(self):
        """Проверява конфигурацията при реално обучение."""

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=tuple(
                (
                    ""
                    if index % 3 == 0
                    else str(index * 10),
                    str(index * 100),
                )
                for index in range(1, 21)
            ),
        )

        configuration = PreprocessingConfiguration(
            numeric_imputation_strategy="constant",
            numeric_fill_value=-99.0,
            categorical_imputation_strategy="most_frequent",
        )

        experiment = train_experiment_model(
            dataset=dataset,
            target_column="Target",
            feature_columns=("Value",),
            task_type="regression",
            algorithm="linear_regression",
            test_size=0.2,
            random_state=42,
            preprocessing_configuration=configuration,
        )

        self.assertEqual(
            experiment.preprocessing_configuration,
            configuration,
        )

        self.assertEqual(
            experiment.training_row_count,
            16,
        )

        self.assertEqual(
            experiment.test_row_count,
            4,
        )

        numeric_pipeline = (
            experiment.transformed_data
            .fitted_preprocessor
            .named_transformers_["numeric"]
        )

        imputer = numeric_pipeline.named_steps[
            "imputer"
        ]

        self.assertEqual(
            imputer.strategy,
            "constant",
        )

        self.assertAlmostEqual(
            imputer.statistics_[0],
            -99.0,
        )

        predictions = experiment.estimator.predict(
            experiment.transformed_data.X_test
        )

        self.assertEqual(
            len(predictions),
            4,
        )

        self.assertTrue(
            np.isfinite(predictions).all()
        )


if __name__ == "__main__":
    unittest.main()