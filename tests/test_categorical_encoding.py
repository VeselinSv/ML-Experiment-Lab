"""Интеграционни тестове за категориално кодиране."""

import unittest
from pathlib import Path

import numpy as np

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
)

from ml_experiment_lab.datasets import LoadedDataset
from ml_experiment_lab.encoding import FrequencyEncoder

from ml_experiment_lab.preprocessing import (
    ENCODING_METHODS,
    PreprocessingConfiguration,
    PreprocessingError,
    get_feature_selection_report,
    prepare_experiment_data,
    with_training_feature_types,
)

from ml_experiment_lab.splitting import ExperimentDataSplit

from ml_experiment_lab.transformation import (
    transform_experiment_split,
)

from ml_experiment_lab.training import (
    train_experiment_model,
)


class CategoricalEncodingTests(unittest.TestCase):
    """Проверява методите за категориално кодиране."""

    def _create_dataset(
        self,
        columns: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> LoadedDataset:
        """Създава примерен набор от данни."""

        return LoadedDataset(
            path=Path("categorical_encoding.xlsx"),
            columns=columns,
            rows=rows,
        )

    def _create_categorical_dataset(
        self,
        values: tuple[str, ...] = (
            "Варна",
            "София",
            "Варна",
            "Пловдив",
            "Варна",
            "Бургас",
            "Варна",
        ),
    ) -> LoadedDataset:
        """Създава данни с една категориална колона."""

        return self._create_dataset(
            columns=("City", "Target"),
            rows=tuple(
                (value, str(index * 100))
                for index, value in enumerate(
                    values,
                    start=1,
                )
            ),
        )

    def _transform_dataset(
        self,
        dataset: LoadedDataset,
        feature_columns: tuple[str, ...],
        train_count: int,
        configuration: PreprocessingConfiguration | None = None,
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

    def _transform_categorical_data(
        self,
        method: str,
    ):
        """Трансформира категориални данни с избран метод."""

        dataset = self._create_categorical_dataset()

        configuration = PreprocessingConfiguration(
            encoding_method=method,
        )

        return self._transform_dataset(
            dataset=dataset,
            feature_columns=("City",),
            train_count=5,
            configuration=configuration,
        )

    def _get_categorical_pipeline(self, transformed):
        """Връща обучения категориален pipeline."""

        return (
            transformed.fitted_preprocessor
            .named_transformers_["categorical"]
        )

    def test_default_encoding_method(self):
        """Проверява метода за кодиране по подразбиране."""

        configuration = PreprocessingConfiguration()

        self.assertEqual(
            configuration.encoding_method,
            "onehot",
        )

        self.assertEqual(
            set(ENCODING_METHODS),
            {
                "onehot",
                "ordinal",
                "frequency",
            },
        )

        dataset = self._create_categorical_dataset()

        prepared, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("City",),
            train_count=5,
        )

        pipeline = self._get_categorical_pipeline(
            transformed
        )

        self.assertIsInstance(
            pipeline.named_steps["encoder"],
            OneHotEncoder,
        )

        self.assertEqual(
            prepared.preprocessing_configuration.encoding_method,
            "onehot",
        )

    def test_onehot_encoding(self):
        """Проверява One-Hot Encoding."""

        _, transformed = self._transform_categorical_data(
            "onehot"
        )

        pipeline = self._get_categorical_pipeline(
            transformed
        )

        encoder = pipeline.named_steps["encoder"]

        self.assertIsInstance(
            encoder,
            OneHotEncoder,
        )

        self.assertEqual(
            tuple(encoder.categories_[0]),
            ("Варна", "Пловдив", "София"),
        )

        self.assertEqual(
            transformed.feature_names,
            (
                "x0_Варна",
                "x0_Пловдив",
                "x0_София",
            ),
        )

        np.testing.assert_array_equal(
            transformed.X_train,
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )

        np.testing.assert_array_equal(
            transformed.X_test,
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )

        self.assertEqual(
            transformed.feature_count,
            3,
        )

    def test_ordinal_encoding(self):
        """Проверява Ordinal Encoding."""

        _, transformed = self._transform_categorical_data(
            "ordinal"
        )

        pipeline = self._get_categorical_pipeline(
            transformed
        )

        encoder = pipeline.named_steps["encoder"]

        self.assertIsInstance(
            encoder,
            OrdinalEncoder,
        )

        self.assertEqual(
            encoder.unknown_value,
            -1,
        )

        self.assertEqual(
            tuple(encoder.categories_[0]),
            ("Варна", "Пловдив", "София"),
        )

        np.testing.assert_array_equal(
            transformed.X_train[:, 0],
            [0.0, 2.0, 0.0, 1.0, 0.0],
        )

        np.testing.assert_array_equal(
            transformed.X_test[:, 0],
            [-1.0, 0.0],
        )

        self.assertEqual(
            transformed.feature_names,
            ("x0",),
        )

        self.assertEqual(
            transformed.feature_count,
            1,
        )

    def test_frequency_encoding(self):
        """Проверява Frequency Encoding."""

        _, transformed = self._transform_categorical_data(
            "frequency"
        )

        pipeline = self._get_categorical_pipeline(
            transformed
        )

        encoder = pipeline.named_steps["encoder"]

        self.assertIsInstance(
            encoder,
            FrequencyEncoder,
        )

        self.assertEqual(
            encoder.frequency_maps_[0],
            {
                "Варна": 0.6,
                "София": 0.2,
                "Пловдив": 0.2,
            },
        )

        np.testing.assert_allclose(
            transformed.X_train[:, 0],
            [0.6, 0.2, 0.6, 0.2, 0.6],
        )

        np.testing.assert_allclose(
            transformed.X_test[:, 0],
            [0.0, 0.6],
        )

        self.assertEqual(
            transformed.feature_names,
            ("x0",),
        )

        self.assertEqual(
            transformed.feature_count,
            1,
        )

    def test_invalid_encoding_methods(self):
        """Проверява отхвърлянето на невалидни методи."""

        invalid_methods = (
            "unknown",
            "label",
            "OneHotEncoder",
            "",
            None,
            True,
            123,
            [],
        )

        for method in invalid_methods:

            with self.subTest(method=method):

                with self.assertRaisesRegex(
                    PreprocessingError,
                    "Неподдържан метод за кодиране",
                ):
                    PreprocessingConfiguration(
                        encoding_method=method,
                    )

    def test_constant_imputation_before_encoding(self):
        """Проверява попълването с константа преди кодиране."""

        dataset = self._create_categorical_dataset(
            (
                "Варна",
                "",
                "София",
                "",
                "Варна",

                # Test.
                "",
                "Бургас",
            )
        )

        expected_values = {
            "onehot": (
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                ],
                [
                    [0, 1, 0],
                    [0, 0, 0],
                ],
            ),
            "ordinal": (
                [[0], [1], [2], [1], [0]],
                [[1], [-1]],
            ),
            "frequency": (
                [[0.4], [0.4], [0.2], [0.4], [0.4]],
                [[0.4], [0.0]],
            ),
        }

        for method in ENCODING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    encoding_method=method,
                    categorical_imputation_strategy="constant",
                    categorical_fill_value="Неизвестен",
                )

                _, transformed = self._transform_dataset(
                    dataset=dataset,
                    feature_columns=("City",),
                    train_count=5,
                    configuration=configuration,
                )

                pipeline = self._get_categorical_pipeline(
                    transformed
                )

                imputer = pipeline.named_steps["imputer"]

                self.assertEqual(
                    imputer.statistics_[0],
                    "Неизвестен",
                )

                expected_train, expected_test = (
                    expected_values[method]
                )

                np.testing.assert_allclose(
                    transformed.X_train,
                    expected_train,
                )

                np.testing.assert_allclose(
                    transformed.X_test,
                    expected_test,
                )

    def test_most_frequent_imputation_before_encoding(self):
        """Проверява попълването с най-честата категория."""

        dataset = self._create_categorical_dataset(
            (
                "Варна",
                "Варна",
                "София",
                "",
                "Варна",

                # Test.
                "",
                "Бургас",
            )
        )

        expected_values = {
            "onehot": (
                [
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [1, 0],
                ],
                [
                    [1, 0],
                    [0, 0],
                ],
            ),
            "ordinal": (
                [[0], [0], [1], [0], [0]],
                [[0], [-1]],
            ),
            "frequency": (
                [[0.8], [0.8], [0.2], [0.8], [0.8]],
                [[0.8], [0.0]],
            ),
        }

        for method in ENCODING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    encoding_method=method,
                    categorical_imputation_strategy="most_frequent",
                )

                _, transformed = self._transform_dataset(
                    dataset=dataset,
                    feature_columns=("City",),
                    train_count=5,
                    configuration=configuration,
                )

                pipeline = self._get_categorical_pipeline(
                    transformed
                )

                imputer = pipeline.named_steps["imputer"]

                self.assertEqual(
                    imputer.statistics_[0],
                    "Варна",
                )

                expected_train, expected_test = (
                    expected_values[method]
                )

                np.testing.assert_allclose(
                    transformed.X_train,
                    expected_train,
                )

                np.testing.assert_allclose(
                    transformed.X_test,
                    expected_test,
                )

    def test_multiple_categorical_columns(self):
        """Проверява едновременно кодиране на няколко колони."""

        dataset = self._create_dataset(
            columns=("City", "Country", "Target"),
            rows=(
                ("Варна", "България", "100"),
                ("София", "България", "200"),
                ("Варна", "Германия", "300"),
                ("Пловдив", "България", "400"),
                ("Варна", "България", "500"),

                # Test.
                ("Бургас", "Франция", "600"),
                ("Варна", "Германия", "700"),
            ),
        )

        expected_counts = {
            "onehot": 5,
            "ordinal": 2,
            "frequency": 2,
        }

        expected_test_values = {
            "onehot": [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
            ],
            "ordinal": [
                [-1, -1],
                [0, 1],
            ],
            "frequency": [
                [0.0, 0.0],
                [0.6, 0.2],
            ],
        }

        for method in ENCODING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    encoding_method=method,
                )

                prepared, transformed = self._transform_dataset(
                    dataset=dataset,
                    feature_columns=("City", "Country"),
                    train_count=5,
                    configuration=configuration,
                )

                self.assertEqual(
                    prepared.categorical_columns,
                    ("City", "Country"),
                )

                self.assertEqual(
                    transformed.feature_count,
                    expected_counts[method],
                )

                self.assertEqual(
                    transformed.X_train.shape,
                    (5, expected_counts[method]),
                )

                self.assertEqual(
                    transformed.X_test.shape,
                    (2, expected_counts[method]),
                )

                np.testing.assert_allclose(
                    transformed.X_test,
                    expected_test_values[method],
                )

                self.assertTrue(
                    np.isfinite(transformed.X_train).all()
                )

                self.assertTrue(
                    np.isfinite(transformed.X_test).all()
                )

    def test_empty_column_fallback(self):
        """Проверява резервната стратегия за празни колони."""

        dataset = self._create_dataset(
            columns=("Notes", "Target"),
            rows=(
                ("", "100"),
                ("", "200"),
                ("", "300"),
                ("", "400"),

                # Test.
                ("", "500"),
                ("Бележка", "600"),
            ),
        )

        expected_values = {
            "onehot": (
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0],
            ),
            "ordinal": (
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0],
            ),
            "frequency": (
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0],
            ),
        }

        for method in ENCODING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    encoding_method=method,
                    categorical_imputation_strategy="most_frequent",
                    categorical_fill_value="Неизвестен",
                    drop_empty_columns=False,
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

                pipeline = (
                    transformed.fitted_preprocessor
                    .named_transformers_["categorical_empty"]
                )

                imputer = pipeline.named_steps["imputer"]

                self.assertEqual(
                    imputer.strategy,
                    "constant",
                )

                self.assertEqual(
                    imputer.statistics_[0],
                    "Неизвестен",
                )

                self.assertEqual(
                    transformed.feature_count,
                    1,
                )

                expected_train, expected_test = (
                    expected_values[method]
                )

                np.testing.assert_allclose(
                    transformed.X_train[:, 0],
                    expected_train,
                )

                np.testing.assert_allclose(
                    transformed.X_test[:, 0],
                    expected_test,
                )

    def test_encoding_prevents_data_leakage(self):
        """Проверява независимостта на Encoding от Test."""

        training_values = (
            "Варна",
            "София",
            "Варна",
            "Пловдив",
            "Варна",
        )

        first_dataset = self._create_categorical_dataset(
            training_values + (
                "Варна",
                "Варна",
                "Варна",
            )
        )

        second_dataset = self._create_categorical_dataset(
            training_values + (
                "Атина",
                "Атина",
                "Бургас",
            )
        )

        for method in ENCODING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    encoding_method=method,
                )

                _, first_transformed = self._transform_dataset(
                    dataset=first_dataset,
                    feature_columns=("City",),
                    train_count=5,
                    configuration=configuration,
                )

                _, second_transformed = self._transform_dataset(
                    dataset=second_dataset,
                    feature_columns=("City",),
                    train_count=5,
                    configuration=configuration,
                )

                first_encoder = (
                    self._get_categorical_pipeline(
                        first_transformed
                    ).named_steps["encoder"]
                )

                second_encoder = (
                    self._get_categorical_pipeline(
                        second_transformed
                    ).named_steps["encoder"]
                )

                self.assertEqual(
                    first_transformed.feature_names,
                    second_transformed.feature_names,
                )

                np.testing.assert_allclose(
                    first_transformed.X_train,
                    second_transformed.X_train,
                )

                if method == "frequency":

                    self.assertEqual(
                        first_encoder.frequency_maps_,
                        second_encoder.frequency_maps_,
                    )

                    self.assertNotIn(
                        "Атина",
                        second_encoder.frequency_maps_[0],
                    )

                    self.assertNotIn(
                        "Бургас",
                        second_encoder.frequency_maps_[0],
                    )

                else:

                    np.testing.assert_array_equal(
                        first_encoder.categories_[0],
                        second_encoder.categories_[0],
                    )

                    self.assertNotIn(
                        "Атина",
                        second_encoder.categories_[0],
                    )

                    self.assertNotIn(
                        "Бургас",
                        second_encoder.categories_[0],
                    )

                self.assertFalse(
                    np.allclose(
                        first_transformed.X_test,
                        second_transformed.X_test,
                    )
                )

    def test_encoding_with_scaling_and_datetime(self):
        """Проверява комбинация от различни типове характеристики."""

        dataset = self._create_dataset(
            columns=(
                "Value",
                "City",
                "Date",
                "Target",
            ),
            rows=(
                ("10", "Варна", "2026-01-01", "100"),
                ("20", "София", "2026-02-01", "200"),
                ("30", "Варна", "2026-03-01", "300"),
                ("40", "Варна", "2026-04-01", "400"),

                # Test.
                ("50", "Бургас", "2026-05-01", "500"),
                ("60", "Варна", "2026-06-01", "600"),
            ),
        )

        expected_categorical_values = {
            "onehot": [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            "ordinal": [
                [0.0],
                [1.0],
                [0.0],
                [0.0],
            ],
            "frequency": [
                [0.75],
                [0.25],
                [0.75],
                [0.75],
            ],
        }

        for method in ENCODING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    encoding_method=method,
                    scaling_method="minmax",
                )

                prepared, transformed = self._transform_dataset(
                    dataset=dataset,
                    feature_columns=(
                        "Value",
                        "City",
                        "Date",
                    ),
                    train_count=4,
                    configuration=configuration,
                )

                self.assertEqual(
                    prepared.numeric_columns,
                    ("Value",),
                )

                self.assertEqual(
                    prepared.categorical_columns,
                    ("City",),
                )

                self.assertEqual(
                    prepared.datetime_columns,
                    ("Date",),
                )

                categorical_count = (
                    2 if method == "onehot" else 1
                )

                self.assertEqual(
                    transformed.feature_count,
                    1 + categorical_count + 9,
                )

                np.testing.assert_allclose(
                    transformed.X_train[:, 0],
                    [0.0, 1 / 3, 2 / 3, 1.0],
                    atol=1e-12,
                )

                np.testing.assert_allclose(
                    transformed.X_train[
                        :,
                        1:1 + categorical_count,
                    ],
                    expected_categorical_values[method],
                )

                categorical_pipeline = (
                    self._get_categorical_pipeline(
                        transformed
                    )
                )

                self.assertNotIn(
                    "scaler",
                    categorical_pipeline.named_steps,
                )

                self.assertTrue(
                    np.isfinite(transformed.X_train).all()
                )

                self.assertTrue(
                    np.isfinite(transformed.X_test).all()
                )

    def test_encoding_with_feature_selection(self):
        """Проверява Feature Selection след кодиране."""

        dataset = self._create_dataset(
            columns=(
                "Value",
                "City",
                "Country",
                "Notes",
                "Target",
            ),
            rows=(
                ("10", "Варна", "България", "", "100"),
                ("20", "София", "България", "", "200"),
                ("30", "Варна", "България", "", "300"),
                ("40", "София", "България", "", "400"),

                # Test.
                ("50", "Бургас", "Германия", "Текст", "500"),
                ("60", "Варна", "Франция", "", "600"),
            ),
        )

        expected_names = {
            "onehot": (
                "x0",
                "x1_Варна",
                "x1_София",
            ),
            "ordinal": (
                "x0",
                "x1",
            ),
            "frequency": (
                "x0",
            ),
        }

        expected_removed_features = {
            "onehot": (
                "x2_България",
            ),
            "ordinal": (
                "x2",
            ),
            "frequency": (
                "x1",
                "x2",
            ),
        }

        for method in ENCODING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    encoding_method=method,
                    drop_empty_columns=True,
                    drop_constant_features=True,
                )

                prepared, transformed = self._transform_dataset(
                    dataset=dataset,
                    feature_columns=(
                        "Value",
                        "City",
                        "Country",
                        "Notes",
                    ),
                    train_count=4,
                    configuration=configuration,
                )

                report = get_feature_selection_report(
                    prepared_data=prepared,
                    fitted_preprocessor=(
                        transformed.fitted_preprocessor
                    ),
                )

                self.assertEqual(
                    report.removed_empty_columns,
                    ("Notes",),
                )

                self.assertEqual(
                    report.removed_constant_features,
                    expected_removed_features[method],
                )

                self.assertEqual(
                    transformed.feature_names,
                    expected_names[method],
                )

                self.assertEqual(
                    transformed.feature_count,
                    len(expected_names[method]),
                )

                self.assertTrue(
                    np.isfinite(transformed.X_train).all()
                )

                self.assertTrue(
                    np.isfinite(transformed.X_test).all()
                )

    def test_standard_preparation_mode(self):
        """Проверява съвместимостта със стандартния режим."""

        dataset = self._create_categorical_dataset()

        expected_counts = {
            "onehot": 3,
            "ordinal": 1,
            "frequency": 1,
        }

        for method in ENCODING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    encoding_method=method,
                )

                prepared, transformed = self._transform_dataset(
                    dataset=dataset,
                    feature_columns=("City",),
                    train_count=5,
                    configuration=configuration,
                    defer_type_detection=False,
                )

                self.assertEqual(
                    prepared.categorical_columns,
                    ("City",),
                )

                self.assertEqual(
                    transformed.feature_count,
                    expected_counts[method],
                )

                self.assertTrue(
                    np.isfinite(transformed.X_train).all()
                )

                self.assertTrue(
                    np.isfinite(transformed.X_test).all()
                )

                encoder = (
                    self._get_categorical_pipeline(
                        transformed
                    ).named_steps["encoder"]
                )

                if method == "frequency":

                    self.assertNotIn(
                        "Бургас",
                        encoder.frequency_maps_[0],
                    )

                else:

                    self.assertNotIn(
                        "Бургас",
                        encoder.categories_[0],
                    )

    def test_training_with_encoding_methods(self):
        """Проверява реално обучение с трите метода."""

        cities = (
            "Варна",
            "София",
            "Пловдив",
        )

        dataset = self._create_dataset(
            columns=("Value", "City", "Target"),
            rows=tuple(
                (
                    str(index * 10),
                    cities[index % len(cities)],
                    str(index * 100),
                )
                for index in range(1, 31)
            ),
        )

        expected_counts = {
            "onehot": 4,
            "ordinal": 2,
            "frequency": 2,
        }

        for method in ENCODING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    encoding_method=method,
                )

                experiment = train_experiment_model(
                    dataset=dataset,
                    target_column="Target",
                    feature_columns=("Value", "City"),
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
                    24,
                )

                self.assertEqual(
                    experiment.test_row_count,
                    6,
                )

                self.assertEqual(
                    experiment.feature_count,
                    expected_counts[method],
                )

                predictions = experiment.estimator.predict(
                    experiment.transformed_data.X_test
                )

                self.assertEqual(
                    len(predictions),
                    6,
                )

                self.assertTrue(
                    np.isfinite(predictions).all()
                )

                np.testing.assert_allclose(
                    predictions,
                    experiment.transformed_data.y_test,
                    atol=1e-8,
                )


if __name__ == "__main__":
    unittest.main()