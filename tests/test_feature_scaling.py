"""Тестове за конфигурируемо мащабиране на характеристики."""

import unittest
from pathlib import Path

import numpy as np

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from ml_experiment_lab.datasets import LoadedDataset

from ml_experiment_lab.datetime_features import (
    extract_datetime_features,
)

from ml_experiment_lab.preprocessing import (
    SCALING_METHODS,
    PreprocessingConfiguration,
    PreprocessingError,
    get_feature_selection_report,
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


class FeatureScalingTests(unittest.TestCase):
    """Проверява методите за мащабиране на характеристики."""

    def _create_dataset(
        self,
        columns: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> LoadedDataset:
        """Създава примерен набор от данни."""

        return LoadedDataset(
            path=Path("feature_scaling.xlsx"),
            columns=columns,
            rows=rows,
        )

    def _create_numeric_dataset(
        self,
        values: tuple[str, ...] = (
            "10",
            "20",
            "30",
            "40",
            "100",
            "-10",
            "150",
        ),
    ) -> LoadedDataset:
        """Създава набор от данни с числова характеристика."""

        return self._create_dataset(
            columns=("Value", "Target"),
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

    def _transform_numeric_data(
        self,
        method: str,
    ):
        """Трансформира числови данни с избран метод."""

        dataset = self._create_numeric_dataset()

        configuration = PreprocessingConfiguration(
            scaling_method=method,
        )

        return self._transform_dataset(
            dataset=dataset,
            feature_columns=("Value",),
            train_count=5,
            configuration=configuration,
        )

    def _get_numeric_pipeline(self, transformed):
        """Връща обучения числов pipeline."""

        return (
            transformed.fitted_preprocessor
            .named_transformers_["numeric"]
        )

    def test_default_scaling_method(self):
        """Проверява метода за мащабиране по подразбиране."""

        configuration = PreprocessingConfiguration()

        self.assertEqual(
            configuration.scaling_method,
            "standard",
        )

        self.assertEqual(
            set(SCALING_METHODS),
            {
                "standard",
                "minmax",
                "robust",
                "maxabs",
                "none",
            },
        )

        dataset = self._create_numeric_dataset()

        prepared, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Value",),
            train_count=5,
        )

        pipeline = self._get_numeric_pipeline(
            transformed
        )

        self.assertIsInstance(
            pipeline.named_steps["scaler"],
            StandardScaler,
        )

        self.assertEqual(
            prepared.preprocessing_configuration.scaling_method,
            "standard",
        )

    def test_standard_scaler(self):
        """Проверява стандартизирането на числовите стойности."""

        _, transformed = self._transform_numeric_data(
            "standard"
        )

        pipeline = self._get_numeric_pipeline(
            transformed
        )

        scaler = pipeline.named_steps["scaler"]

        self.assertIsInstance(
            scaler,
            StandardScaler,
        )

        expected_mean = 40.0
        expected_scale = np.sqrt(1000.0)

        self.assertAlmostEqual(
            scaler.mean_[0],
            expected_mean,
        )

        self.assertAlmostEqual(
            scaler.scale_[0],
            expected_scale,
        )

        expected_train = (
            np.array([10, 20, 30, 40, 100])
            - expected_mean
        ) / expected_scale

        expected_test = (
            np.array([-10, 150])
            - expected_mean
        ) / expected_scale

        np.testing.assert_allclose(
            transformed.X_train[:, 0],
            expected_train,
        )

        np.testing.assert_allclose(
            transformed.X_test[:, 0],
            expected_test,
        )

        self.assertAlmostEqual(
            np.mean(transformed.X_train[:, 0]),
            0.0,
        )

        self.assertAlmostEqual(
            np.std(transformed.X_train[:, 0]),
            1.0,
        )

    def test_minmax_scaler(self):
        """Проверява нормализиране чрез MinMaxScaler."""

        _, transformed = self._transform_numeric_data(
            "minmax"
        )

        pipeline = self._get_numeric_pipeline(
            transformed
        )

        scaler = pipeline.named_steps["scaler"]

        self.assertIsInstance(
            scaler,
            MinMaxScaler,
        )

        self.assertAlmostEqual(
            scaler.data_min_[0],
            10.0,
        )

        self.assertAlmostEqual(
            scaler.data_max_[0],
            100.0,
        )

        expected_train = (
            np.array([10, 20, 30, 40, 100]) - 10
        ) / 90

        expected_test = (
            np.array([-10, 150]) - 10
        ) / 90

        np.testing.assert_allclose(
            transformed.X_train[:, 0],
            expected_train,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            transformed.X_test[:, 0],
            expected_test,
            atol=1e-12,
        )

        self.assertAlmostEqual(
            transformed.X_train.min(),
            0.0,
        )

        self.assertAlmostEqual(
            transformed.X_train.max(),
            1.0,
        )

        # Test стойностите могат да излизат
        # извън тренировъчния диапазон [0, 1].
        self.assertLess(
            transformed.X_test[0, 0],
            0.0,
        )

        self.assertGreater(
            transformed.X_test[1, 0],
            1.0,
        )

        self.assertFalse(
            scaler.clip
        )

    def test_robust_scaler(self):
        """Проверява мащабиране чрез медиана и IQR."""

        _, transformed = self._transform_numeric_data(
            "robust"
        )

        pipeline = self._get_numeric_pipeline(
            transformed
        )

        scaler = pipeline.named_steps["scaler"]

        self.assertIsInstance(
            scaler,
            RobustScaler,
        )

        expected_median = 30.0
        expected_iqr = 20.0

        self.assertAlmostEqual(
            scaler.center_[0],
            expected_median,
        )

        self.assertAlmostEqual(
            scaler.scale_[0],
            expected_iqr,
        )

        expected_train = (
            np.array([10, 20, 30, 40, 100])
            - expected_median
        ) / expected_iqr

        expected_test = (
            np.array([-10, 150])
            - expected_median
        ) / expected_iqr

        np.testing.assert_allclose(
            transformed.X_train[:, 0],
            expected_train,
        )

        np.testing.assert_allclose(
            transformed.X_test[:, 0],
            expected_test,
        )

    def test_maxabs_scaler(self):
        """Проверява мащабиране чрез абсолютен максимум."""

        _, transformed = self._transform_numeric_data(
            "maxabs"
        )

        pipeline = self._get_numeric_pipeline(
            transformed
        )

        scaler = pipeline.named_steps["scaler"]

        self.assertIsInstance(
            scaler,
            MaxAbsScaler,
        )

        self.assertAlmostEqual(
            scaler.max_abs_[0],
            100.0,
        )

        np.testing.assert_allclose(
            transformed.X_train[:, 0],
            [0.1, 0.2, 0.3, 0.4, 1.0],
        )

        np.testing.assert_allclose(
            transformed.X_test[:, 0],
            [-0.1, 1.5],
        )

        # Проверяваме и запазването на отрицателния знак.
        self.assertLess(
            transformed.X_test[0, 0],
            0.0,
        )

    def test_no_scaling(self):
        """Проверява запазването на числовия мащаб."""

        _, transformed = self._transform_numeric_data(
            "none"
        )

        pipeline = self._get_numeric_pipeline(
            transformed
        )

        self.assertEqual(
            pipeline.named_steps["scaler"],
            "passthrough",
        )

        np.testing.assert_allclose(
            transformed.X_train[:, 0],
            [10, 20, 30, 40, 100],
        )

        np.testing.assert_allclose(
            transformed.X_test[:, 0],
            [-10, 150],
        )

        self.assertEqual(
            transformed.feature_names,
            ("x0",),
        )

    def test_invalid_scaling_methods(self):
        """Проверява отхвърлянето на невалидни методи."""

        invalid_methods = (
            "unknown",
            "normalization",
            "StandardScaler",
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
                    "Неподдържан метод за мащабиране",
                ):
                    PreprocessingConfiguration(
                        scaling_method=method,
                    )

    def test_imputation_before_scaling(self):
        """Проверява попълването преди мащабиране."""

        dataset = self._create_numeric_dataset(
            (
                "10",
                "20",
                "30",
                "",

                # Test.
                "",
                "1000",
            )
        )

        configuration = PreprocessingConfiguration(
            numeric_imputation_strategy="constant",
            numeric_fill_value=100.0,
            scaling_method="standard",
        )

        _, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Value",),
            train_count=4,
            configuration=configuration,
        )

        pipeline = self._get_numeric_pipeline(
            transformed
        )

        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]

        self.assertAlmostEqual(
            float(imputer.statistics_[0]),
            100.0,
        )

        # След попълването Training съдържа:
        # 10, 20, 30 и 100.
        self.assertAlmostEqual(
            scaler.mean_[0],
            40.0,
        )

        restored_train = scaler.inverse_transform(
            transformed.X_train
        )

        restored_test = scaler.inverse_transform(
            transformed.X_test
        )

        np.testing.assert_allclose(
            restored_train[:, 0],
            [10, 20, 30, 100],
        )

        np.testing.assert_allclose(
            restored_test[:, 0],
            [100, 1000],
        )

    def test_scaling_datetime_features(self):
        """Проверява мащабирането на датни характеристики."""

        dates = (
            "2026-01-01",
            "2026-02-15",
            "2026-03-20",
            "2026-04-25",

            # Test.
            "2026-05-10",
            "2027-06-15 14:30:45",
        )

        dataset = self._create_dataset(
            columns=("Date", "Target"),
            rows=tuple(
                (value, str(index * 100))
                for index, value in enumerate(
                    dates,
                    start=1,
                )
            ),
        )

        raw_features = np.asarray(
            [
                extract_datetime_features(value)
                for value in dates
            ],
            dtype=float,
        )

        raw_train = raw_features[:4]
        raw_test = raw_features[4:]

        scaler_classes = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "maxabs": MaxAbsScaler,
            "none": None,
        }

        for method, scaler_class in scaler_classes.items():

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    scaling_method=method,
                )

                prepared, transformed = (
                    self._transform_dataset(
                        dataset=dataset,
                        feature_columns=("Date",),
                        train_count=4,
                        configuration=configuration,
                    )
                )

                self.assertEqual(
                    prepared.datetime_columns,
                    ("Date",),
                )

                self.assertEqual(
                    transformed.feature_count,
                    9,
                )

                pipeline = (
                    transformed.fitted_preprocessor
                    .named_transformers_["datetime"]
                )

                actual_scaler = pipeline.named_steps[
                    "scaler"
                ]

                if scaler_class is None:

                    self.assertEqual(
                        actual_scaler,
                        "passthrough",
                    )

                    expected_train = raw_train
                    expected_test = raw_test

                else:

                    self.assertIsInstance(
                        actual_scaler,
                        scaler_class,
                    )

                    reference_scaler = scaler_class()

                    expected_train = (
                        reference_scaler.fit_transform(
                            raw_train
                        )
                    )

                    expected_test = (
                        reference_scaler.transform(
                            raw_test
                        )
                    )

                np.testing.assert_allclose(
                    transformed.X_train,
                    expected_train,
                    atol=1e-10,
                )

                np.testing.assert_allclose(
                    transformed.X_test,
                    expected_test,
                    atol=1e-10,
                )

                self.assertTrue(
                    np.isfinite(transformed.X_train).all()
                )

                self.assertTrue(
                    np.isfinite(transformed.X_test).all()
                )

    def test_categorical_features_are_not_scaled(self):
        """Проверява запазването на One-Hot характеристиките."""

        dataset = self._create_dataset(
            columns=("Value", "City", "Target"),
            rows=(
                ("10", "Варна", "100"),
                ("20", "София", "200"),
                ("30", "Варна", "300"),
                ("40", "София", "400"),

                # Test.
                ("50", "Пловдив", "500"),
                ("60", "Варна", "600"),
            ),
        )

        expected_train = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        expected_test = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )

        for method in SCALING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    scaling_method=method,
                )

                _, transformed = self._transform_dataset(
                    dataset=dataset,
                    feature_columns=("Value", "City"),
                    train_count=4,
                    configuration=configuration,
                )

                categorical_pipeline = (
                    transformed.fitted_preprocessor
                    .named_transformers_["categorical"]
                )

                self.assertNotIn(
                    "scaler",
                    categorical_pipeline.named_steps,
                )

                self.assertEqual(
                    transformed.feature_count,
                    3,
                )

                # Първата характеристика е числова.
                # Останалите две са One-Hot характеристики.
                np.testing.assert_array_equal(
                    transformed.X_train[:, 1:],
                    expected_train,
                )

                np.testing.assert_array_equal(
                    transformed.X_test[:, 1:],
                    expected_test,
                )

    def test_scaling_prevents_data_leakage(self):
        """Проверява независимостта на Training от Test."""

        training_values = (
            "10",
            "20",
            "30",
            "40",
            "100",
        )

        first_dataset = self._create_numeric_dataset(
            training_values + (
                "50",
                "60",
            )
        )

        second_dataset = self._create_numeric_dataset(
            training_values + (
                "-100000",
                "100000",
            )
        )

        scaler_attributes = {
            "standard": (
                "mean_",
                "scale_",
                "var_",
            ),
            "minmax": (
                "data_min_",
                "data_max_",
                "scale_",
            ),
            "robust": (
                "center_",
                "scale_",
            ),
            "maxabs": (
                "max_abs_",
                "scale_",
            ),
            "none": (),
        }

        for method in SCALING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    scaling_method=method,
                )

                first_prepared, first_transformed = (
                    self._transform_dataset(
                        dataset=first_dataset,
                        feature_columns=("Value",),
                        train_count=5,
                        configuration=configuration,
                    )
                )

                second_prepared, second_transformed = (
                    self._transform_dataset(
                        dataset=second_dataset,
                        feature_columns=("Value",),
                        train_count=5,
                        configuration=configuration,
                    )
                )

                self.assertEqual(
                    first_prepared.numeric_columns,
                    second_prepared.numeric_columns,
                )

                self.assertEqual(
                    first_transformed.feature_names,
                    second_transformed.feature_names,
                )

                np.testing.assert_allclose(
                    first_transformed.X_train,
                    second_transformed.X_train,
                )

                first_pipeline = self._get_numeric_pipeline(
                    first_transformed
                )

                second_pipeline = self._get_numeric_pipeline(
                    second_transformed
                )

                first_scaler = first_pipeline.named_steps[
                    "scaler"
                ]

                second_scaler = second_pipeline.named_steps[
                    "scaler"
                ]

                for attribute in scaler_attributes[method]:

                    np.testing.assert_allclose(
                        getattr(
                            first_scaler,
                            attribute,
                        ),
                        getattr(
                            second_scaler,
                            attribute,
                        ),
                    )

                # Test матриците трябва да бъдат различни,
                # но научените параметри остават еднакви.
                self.assertFalse(
                    np.allclose(
                        first_transformed.X_test,
                        second_transformed.X_test,
                    )
                )

    def test_scaling_with_feature_selection(self):
        """Проверява съвместимостта с Feature Selection."""

        dataset = self._create_dataset(
            columns=(
                "Value",
                "Constant",
                "Notes",
                "Target",
            ),
            rows=(
                ("10", "5", "", "100"),
                ("20", "5", "", "200"),
                ("30", "5", "", "300"),
                ("40", "5", "", "400"),

                # Test.
                ("50", "100", "Бележка", "500"),
                ("60", "200", "", "600"),
            ),
        )

        for method in SCALING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    scaling_method=method,
                    drop_empty_columns=True,
                    drop_constant_features=True,
                )

                prepared, transformed = (
                    self._transform_dataset(
                        dataset=dataset,
                        feature_columns=(
                            "Value",
                            "Constant",
                            "Notes",
                        ),
                        train_count=4,
                        configuration=configuration,
                    )
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
                    ("x1",),
                )

                self.assertEqual(
                    transformed.feature_count,
                    1,
                )

                self.assertEqual(
                    transformed.feature_names,
                    ("x0",),
                )

                self.assertEqual(
                    transformed.X_train.shape,
                    (4, 1),
                )

                self.assertEqual(
                    transformed.X_test.shape,
                    (2, 1),
                )

                self.assertTrue(
                    np.isfinite(transformed.X_train).all()
                )

                self.assertTrue(
                    np.isfinite(transformed.X_test).all()
                )

    def test_standard_preparation_mode(self):
        """Проверява съвместимостта със стандартния режим."""

        dataset = self._create_numeric_dataset(
            (
                "10,5",
                "20,5",
                "",

                # Test.
                "30,5",
            )
        )

        configuration = PreprocessingConfiguration(
            scaling_method="minmax",
            numeric_imputation_strategy="median",
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

        pipeline = self._get_numeric_pipeline(
            transformed
        )

        scaler = pipeline.named_steps["scaler"]

        self.assertIsInstance(
            scaler,
            MinMaxScaler,
        )

        np.testing.assert_allclose(
            transformed.X_train[:, 0],
            [0.0, 1.0, 0.5],
            atol=1e-12,
        )

        np.testing.assert_allclose(
            transformed.X_test[:, 0],
            [2.0],
            atol=1e-12,
        )

    def test_training_with_scaling_methods(self):
        """Проверява реално обучение с всички методи."""

        dataset = self._create_dataset(
            columns=("Value", "Target"),
            rows=tuple(
                (
                    str(index * 10),
                    str(index * 100),
                )
                for index in range(1, 21)
            ),
        )

        for method in SCALING_METHODS:

            with self.subTest(method=method):

                configuration = PreprocessingConfiguration(
                    scaling_method=method,
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

                self.assertEqual(
                    experiment.feature_count,
                    1,
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

                np.testing.assert_allclose(
                    predictions,
                    experiment.transformed_data.y_test,
                    atol=1e-8,
                )

                # Target не трябва да се мащабира.
                original_y_train = np.asarray(
                    experiment.data_split.y_train,
                    dtype=float,
                )

                original_y_test = np.asarray(
                    experiment.data_split.y_test,
                    dtype=float,
                )

                np.testing.assert_array_equal(
                    experiment.transformed_data.y_train,
                    original_y_train,
                )

                np.testing.assert_array_equal(
                    experiment.transformed_data.y_test,
                    original_y_test,
                )


if __name__ == "__main__":
    unittest.main()