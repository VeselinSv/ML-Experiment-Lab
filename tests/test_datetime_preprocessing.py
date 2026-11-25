"""Интеграционни тестове за обработката на датни колони."""

import unittest
from pathlib import Path

import numpy as np

from ml_experiment_lab.datasets import LoadedDataset

from ml_experiment_lab.datetime_features import (
    DATETIME_FEATURE_NAMES,
    extract_datetime_features,
)

from ml_experiment_lab.preprocessing import (
    PreprocessingConfiguration,
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


class DatetimePreprocessingTests(unittest.TestCase):
    """Проверява интеграцията на датите в Preprocessing."""

    def _create_dataset(
        self,
        columns: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> LoadedDataset:
        """Създава примерен набор от данни."""

        return LoadedDataset(
            path=Path("datetime_preprocessing.xlsx"),
            columns=columns,
            rows=rows,
        )

    def _transform_dataset(
        self,
        dataset: LoadedDataset,
        feature_columns: tuple[str, ...],
        train_count: int,
        configuration: PreprocessingConfiguration | None = None,
    ):
        """Подготвя и трансформира контролирани извадки."""

        prepared = prepare_experiment_data(
            dataset=dataset,
            target_column="Target",
            feature_columns=feature_columns,
            defer_type_detection=True,
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

        prepared = with_training_feature_types(
            prepared_data=prepared,
            X_train=data_split.X_train,
        )

        transformed = transform_experiment_split(
            prepared_data=prepared,
            data_split=data_split,
        )

        return prepared, transformed

    def _create_date_dataset(
        self,
        date_values: tuple[str, ...],
    ) -> LoadedDataset:
        """Създава набор от данни с една датна колона."""

        return self._create_dataset(
            columns=("Date", "Target"),
            rows=tuple(
                (value, str(index * 100))
                for index, value in enumerate(
                    date_values,
                    start=1,
                )
            ),
        )

    def _get_datetime_pipeline(self, transformed):
        """Връща обучения pipeline за дати."""

        return (
            transformed.fitted_preprocessor
            .named_transformers_["datetime"]
        )

    def test_datetime_column_detection(self):
        """Проверява разпознаването на датни колони."""

        dataset = self._create_date_dataset(
            (
                "2026-01-01",
                "15.02.2026",
                "2026-03-20",
                "",
                "2026-05-10",
            )
        )

        prepared = prepare_experiment_data(
            dataset=dataset,
            target_column="Target",
            feature_columns=("Date",),
        )

        self.assertEqual(
            prepared.datetime_columns,
            ("Date",),
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

    def test_datetime_detection_uses_training_only(self):
        """Проверява разпознаването само от Training."""

        dataset = self._create_date_dataset(
            (
                "2026-01-01",
                "2026-02-01",
                "2026-03-01",
                "",

                # Test.
                "Неизвестен",
                "31.02.2026",
            )
        )

        initial_prepared = prepare_experiment_data(
            dataset=dataset,
            target_column="Target",
            feature_columns=("Date",),
            defer_type_detection=True,
        )

        self.assertEqual(
            initial_prepared.datetime_columns,
            (),
        )

        prepared, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Date",),
            train_count=4,
        )

        self.assertEqual(
            prepared.datetime_columns,
            ("Date",),
        )

        self.assertEqual(
            prepared.mixed_columns,
            (),
        )

        self.assertEqual(
            prepared.categorical_columns,
            (),
        )

        self.assertEqual(
            transformed.feature_count,
            9,
        )

        # Дати само в Test не превръщат
        # категориалната Training колона в датна.
        categorical_dataset = self._create_date_dataset(
            (
                "Варна",
                "София",
                "Варна",
                "София",
                "2026-01-01",
                "2026-02-01",
            )
        )

        categorical_prepared, _ = self._transform_dataset(
            dataset=categorical_dataset,
            feature_columns=("Date",),
            train_count=4,
        )

        self.assertEqual(
            categorical_prepared.datetime_columns,
            (),
        )

        self.assertEqual(
            categorical_prepared.categorical_columns,
            ("Date",),
        )

    def test_datetime_feature_expansion(self):
        """Проверява извличането на девет характеристики."""

        dataset = self._create_date_dataset(
            (
                "2026-01-01",
                "2026-03-15",
                "2026-06-20",
                "2026-11-25 14:30:45",

                # Test.
                "2026-12-31",
                "2027-01-01",
            )
        )

        _, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Date",),
            train_count=4,
        )

        self.assertEqual(
            transformed.feature_count,
            len(DATETIME_FEATURE_NAMES),
        )

        self.assertEqual(
            transformed.X_train.shape,
            (4, 9),
        )

        self.assertEqual(
            transformed.X_test.shape,
            (2, 9),
        )

        pipeline = self._get_datetime_pipeline(
            transformed
        )

        scaler = pipeline.named_steps["scaler"]

        restored_train = scaler.inverse_transform(
            transformed.X_train
        )

        np.testing.assert_allclose(
            restored_train[3],
            extract_datetime_features(
                "2026-11-25 14:30:45"
            ),
        )

        self.assertTrue(
            np.isfinite(transformed.X_train).all()
        )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

    def test_datetime_feature_names(self):
        """Проверява имената на извлечените характеристики."""

        dataset = self._create_date_dataset(
            (
                "2026-01-01",
                "2026-02-01",
                "2026-03-01",
                "2026-04-01",
                "2026-05-01",
            )
        )

        _, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Date",),
            train_count=4,
        )

        expected_names = tuple(
            f"x0_{feature_name}"
            for feature_name in DATETIME_FEATURE_NAMES
        )

        self.assertEqual(
            transformed.feature_names,
            expected_names,
        )

        self.assertEqual(
            len(set(transformed.feature_names)),
            9,
        )

    def test_missing_datetime_imputation(self):
        """Проверява попълването на липсващи дати."""

        dataset = self._create_date_dataset(
            (
                "2026-01-01",
                "2026-01-03",
                "",

                # Test.
                "",
                "2026-01-05",
            )
        )

        configuration = PreprocessingConfiguration(
            numeric_imputation_strategy="median",
        )

        _, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Date",),
            train_count=3,
            configuration=configuration,
        )

        pipeline = self._get_datetime_pipeline(
            transformed
        )

        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]

        expected_median = np.array(
            [
                2026.0,
                1.0,
                2.0,
                4.0,
                1.0,
                2.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        np.testing.assert_allclose(
            imputer.statistics_,
            expected_median,
        )

        restored_train = scaler.inverse_transform(
            transformed.X_train
        )

        restored_test = scaler.inverse_transform(
            transformed.X_test
        )

        # Липсваща дата в Training.
        np.testing.assert_allclose(
            restored_train[2],
            expected_median,
        )

        # Липсваща дата в Test.
        np.testing.assert_allclose(
            restored_test[0],
            expected_median,
        )

    def test_invalid_datetime_in_test(self):
        """Проверява невалидни дати в тестовата извадка."""

        dataset = self._create_date_dataset(
            (
                "2026-01-01",
                "2026-01-03",
                "2026-01-05",

                # Test.
                "31.02.2026",
                "Неизвестен",
            )
        )

        prepared, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Date",),
            train_count=3,
        )

        self.assertEqual(
            prepared.datetime_columns,
            ("Date",),
        )

        pipeline = self._get_datetime_pipeline(
            transformed
        )

        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]

        restored_test = scaler.inverse_transform(
            transformed.X_test
        )

        for row in restored_test:
            np.testing.assert_allclose(
                row,
                imputer.statistics_,
            )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

    def test_datetime_constant_imputation(self):
        """Проверява използването на числова константа."""

        dataset = self._create_date_dataset(
            (
                "2026-01-01",
                "2026-02-01",
                "",

                # Test.
                "",
                "Неизвестен",
            )
        )

        configuration = PreprocessingConfiguration(
            numeric_imputation_strategy="constant",
            numeric_fill_value=-1.0,
        )

        _, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Date",),
            train_count=3,
            configuration=configuration,
        )

        pipeline = self._get_datetime_pipeline(
            transformed
        )

        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]

        expected = np.full(
            9,
            -1.0,
            dtype=float,
        )

        self.assertEqual(
            imputer.strategy,
            "constant",
        )

        # Преобразуваме статистиките в числов масив,
        # за да осигурим съвместимост с assert_allclose.
        np.testing.assert_allclose(
            np.asarray(
                imputer.statistics_,
                dtype=float,
            ),
            expected,
        )

        restored_train = scaler.inverse_transform(
            transformed.X_train
        )

        restored_test = scaler.inverse_transform(
            transformed.X_test
        )

        np.testing.assert_allclose(
            restored_train[2],
            expected,
        )

        for row in restored_test:
            np.testing.assert_allclose(
                row,
                expected,
            )

    def test_multiple_datetime_columns(self):
        """Проверява обработката на няколко датни колони."""

        dataset = self._create_dataset(
            columns=(
                "StartDate",
                "EndDate",
                "Target",
            ),
            rows=(
                ("2026-01-01", "2026-01-10", "100"),
                ("2026-02-01", "2026-02-15", "200"),
                ("2026-03-01", "2026-03-20", "300"),
                ("2026-04-01", "2026-04-25", "400"),

                # Test.
                ("2026-05-01", "2026-05-30", "500"),
                ("", "2026-06-15", "600"),
            ),
        )

        prepared, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=(
                "StartDate",
                "EndDate",
            ),
            train_count=4,
        )

        self.assertEqual(
            prepared.datetime_columns,
            ("StartDate", "EndDate"),
        )

        self.assertEqual(
            transformed.feature_count,
            18,
        )

        self.assertEqual(
            transformed.X_train.shape,
            (4, 18),
        )

        self.assertEqual(
            transformed.X_test.shape,
            (2, 18),
        )

        expected_names = tuple(
            f"x{index}_{feature_name}"
            for index in range(2)
            for feature_name in DATETIME_FEATURE_NAMES
        )

        self.assertEqual(
            transformed.feature_names,
            expected_names,
        )

        pipeline = self._get_datetime_pipeline(
            transformed
        )

        scaler = pipeline.named_steps["scaler"]

        restored_train = scaler.inverse_transform(
            transformed.X_train
        )

        expected_first_row = (
            extract_datetime_features("2026-01-01")
            + extract_datetime_features("2026-01-10")
        )

        np.testing.assert_allclose(
            restored_train[0],
            expected_first_row,
        )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

    def test_combined_feature_types(self):
        """Проверява комбинация от трите основни типа."""

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
                ("40", "София", "2026-04-01", "400"),

                # Test.
                ("50", "Пловдив", "2026-05-01", "500"),
                ("60", "Варна", "", "600"),
            ),
        )

        prepared, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=(
                "Value",
                "City",
                "Date",
            ),
            train_count=4,
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

        # 1 числова + 2 категориални + 9 датни.
        self.assertEqual(
            transformed.feature_count,
            12,
        )

        self.assertEqual(
            transformed.X_train.shape,
            (4, 12),
        )

        self.assertEqual(
            transformed.X_test.shape,
            (2, 12),
        )

        self.assertTrue(
            np.isfinite(transformed.X_train).all()
        )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

    def test_mixed_datetime_and_text_column(self):
        """Проверява колона със смесени дати и текст."""

        dataset = self._create_date_dataset(
            (
                "2026-01-01",
                "Неизвестен",
                "2026-03-01",
                "Няма информация",

                # Test.
                "2026-05-01",
                "",
            )
        )

        prepared, transformed = self._transform_dataset(
            dataset=dataset,
            feature_columns=("Date",),
            train_count=4,
        )

        self.assertEqual(
            prepared.datetime_columns,
            (),
        )

        self.assertEqual(
            prepared.mixed_columns,
            ("Date",),
        )

        self.assertEqual(
            prepared.categorical_columns,
            ("Date",),
        )

        categorical_pipeline = (
            transformed.fitted_preprocessor
            .named_transformers_["categorical"]
        )

        encoder = categorical_pipeline.named_steps[
            "encoder"
        ]

        self.assertEqual(
            set(encoder.categories_[0]),
            {
                "2026-01-01",
                "Неизвестен",
                "2026-03-01",
                "Няма информация",
            },
        )

        self.assertEqual(
            transformed.feature_count,
            4,
        )

    def test_datetime_prevents_data_leakage(self):
        """Проверява независимостта на Training от Test."""

        training_dates = (
            "2026-01-01",
            "2026-02-01",
            "2026-03-01",
            "",
        )

        first_dataset = self._create_date_dataset(
            training_dates
            + (
                "2026-04-01",
                "2026-05-01",
            )
        )

        second_dataset = self._create_date_dataset(
            training_dates
            + (
                "2099-12-31 23:59:59",
                "Неизвестен",
            )
        )

        configuration = PreprocessingConfiguration(
            numeric_imputation_strategy="mean",
        )

        first_prepared, first_transformed = (
            self._transform_dataset(
                dataset=first_dataset,
                feature_columns=("Date",),
                train_count=4,
                configuration=configuration,
            )
        )

        second_prepared, second_transformed = (
            self._transform_dataset(
                dataset=second_dataset,
                feature_columns=("Date",),
                train_count=4,
                configuration=configuration,
            )
        )

        self.assertEqual(
            first_prepared.datetime_columns,
            second_prepared.datetime_columns,
        )

        self.assertEqual(
            first_transformed.feature_names,
            second_transformed.feature_names,
        )

        np.testing.assert_allclose(
            first_transformed.X_train,
            second_transformed.X_train,
        )

        first_pipeline = self._get_datetime_pipeline(
            first_transformed
        )

        second_pipeline = self._get_datetime_pipeline(
            second_transformed
        )

        first_imputer = first_pipeline.named_steps[
            "imputer"
        ]

        second_imputer = second_pipeline.named_steps[
            "imputer"
        ]

        np.testing.assert_allclose(
            first_imputer.statistics_,
            second_imputer.statistics_,
        )

        first_scaler = first_pipeline.named_steps[
            "scaler"
        ]

        second_scaler = second_pipeline.named_steps[
            "scaler"
        ]

        np.testing.assert_allclose(
            first_scaler.mean_,
            second_scaler.mean_,
        )

        np.testing.assert_allclose(
            first_scaler.scale_,
            second_scaler.scale_,
        )

    def test_training_with_datetime_features(self):
        """Проверява реално обучение с датни характеристики."""

        dataset = self._create_dataset(
            columns=(
                "Date",
                "Value",
                "Target",
            ),
            rows=tuple(
                (
                    f"2026-01-{index:02d}",
                    str(index * 10),
                    str(index * 100),
                )
                for index in range(1, 21)
            ),
        )

        experiment = train_experiment_model(
            dataset=dataset,
            target_column="Target",
            feature_columns=("Date", "Value"),
            task_type="regression",
            algorithm="linear_regression",
            test_size=0.2,
            random_state=42,
        )

        self.assertEqual(
            experiment.prepared_data.datetime_columns,
            ("Date",),
        )

        self.assertEqual(
            experiment.prepared_data.numeric_columns,
            ("Value",),
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
            10,
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


if __name__ == "__main__":
    unittest.main()