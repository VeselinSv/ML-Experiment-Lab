"""Тестове за премахване на празни и постоянни характеристики."""

import unittest
from pathlib import Path

import numpy as np

from ml_experiment_lab.datasets import LoadedDataset

from ml_experiment_lab.preprocessing import (
    FeatureSelectionReport,
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
    TransformationError,
    transform_experiment_split,
)

from ml_experiment_lab.training import (
    train_experiment_model,
)


class FeatureSelectionTests(unittest.TestCase):
    """Проверява конфигурируемото филтриране на характеристики."""

    def _create_dataset(
        self,
        columns: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> LoadedDataset:
        """Създава примерен набор от данни."""

        return LoadedDataset(
            path=Path("feature_selection.xlsx"),
            columns=columns,
            rows=rows,
        )

    def _create_example_dataset(self) -> LoadedDataset:
        """Създава данни с празна и постоянна колона."""

        return self._create_dataset(
            columns=(
                "Value",
                "Country",
                "Notes",
                "Target",
            ),
            rows=(
                ("10", "България", "", "100"),
                ("20", "България", "", "200"),
                ("30", "България", "", "300"),
                ("40", "България", "", "400"),

                # Test данни.
                ("50", "Германия", "Бележка", "500"),
                ("60", "Франция", "", "600"),
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

        report = get_feature_selection_report(
            prepared_data=prepared,
            fitted_preprocessor=(
                transformed.fitted_preprocessor
            ),
        )

        return prepared, transformed, report

    def test_default_configuration_preserves_features(self):
        """Проверява запазването на характеристиките."""

        configuration = PreprocessingConfiguration()

        self.assertFalse(
            configuration.drop_empty_columns
        )

        self.assertFalse(
            configuration.drop_constant_features
        )

        dataset = self._create_example_dataset()

        prepared, transformed, report = (
            self._transform_dataset(
                dataset=dataset,
                feature_columns=(
                    "Value",
                    "Country",
                    "Notes",
                ),
                train_count=4,
                configuration=configuration,
            )
        )

        self.assertEqual(
            prepared.empty_columns,
            ("Notes",),
        )

        self.assertEqual(
            transformed.feature_count,
            3,
        )

        self.assertEqual(
            report,
            FeatureSelectionReport(),
        )

        self.assertEqual(
            report.removed_empty_count,
            0,
        )

        self.assertEqual(
            report.removed_constant_count,
            0,
        )

    def test_drop_empty_columns(self):
        """Проверява премахването на празни колони."""

        dataset = self._create_example_dataset()

        original_rows = dataset.rows

        configuration = PreprocessingConfiguration(
            drop_empty_columns=True,
            drop_constant_features=False,
        )

        prepared, transformed, report = (
            self._transform_dataset(
                dataset=dataset,
                feature_columns=(
                    "Value",
                    "Country",
                    "Notes",
                ),
                train_count=4,
                configuration=configuration,
            )
        )

        self.assertEqual(
            prepared.empty_columns,
            ("Notes",),
        )

        self.assertEqual(
            report.removed_empty_columns,
            ("Notes",),
        )

        self.assertEqual(
            report.removed_empty_count,
            1,
        )

        self.assertEqual(
            report.removed_constant_features,
            (),
        )

        self.assertEqual(
            transformed.feature_count,
            2,
        )

        self.assertEqual(
            transformed.X_train.shape,
            (4, 2),
        )

        self.assertEqual(
            transformed.X_test.shape,
            (2, 2),
        )

        # Оригиналният набор от данни не се променя.
        self.assertEqual(
            dataset.rows,
            original_rows,
        )

        self.assertTrue(
            np.isfinite(transformed.X_train).all()
        )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

    def test_drop_constant_features(self):
        """Проверява премахването на постоянни характеристики."""

        dataset = self._create_example_dataset()

        configuration = PreprocessingConfiguration(
            drop_empty_columns=False,
            drop_constant_features=True,
        )

        _, transformed, report = (
            self._transform_dataset(
                dataset=dataset,
                feature_columns=(
                    "Value",
                    "Country",
                    "Notes",
                ),
                train_count=4,
                configuration=configuration,
            )
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
            report.removed_empty_columns,
            (),
        )

        self.assertEqual(
            report.removed_constant_count,
            2,
        )

        self.assertEqual(
            set(report.removed_constant_features),
            {
                "x1_България",
                "x2_ЛИПСВАЩА_СТОЙНОСТ",
            },
        )

        variance_selector = (
            transformed.fitted_preprocessor
            .named_steps["variance"]
        )

        np.testing.assert_array_equal(
            variance_selector.get_support(),
            [True, False, False],
        )

    def test_drop_empty_and_constant_features(self):
        """Проверява комбинираното филтриране."""

        dataset = self._create_example_dataset()

        configuration = PreprocessingConfiguration(
            drop_empty_columns=True,
            drop_constant_features=True,
        )

        _, transformed, report = (
            self._transform_dataset(
                dataset=dataset,
                feature_columns=(
                    "Value",
                    "Country",
                    "Notes",
                ),
                train_count=4,
                configuration=configuration,
            )
        )

        self.assertEqual(
            report.removed_empty_columns,
            ("Notes",),
        )

        self.assertEqual(
            report.removed_constant_features,
            ("x1_България",),
        )

        self.assertEqual(
            report.removed_empty_count,
            1,
        )

        self.assertEqual(
            report.removed_constant_count,
            1,
        )

        self.assertEqual(
            transformed.feature_count,
            1,
        )

        self.assertEqual(
            transformed.X_train.shape,
            (4, 1),
        )

        self.assertEqual(
            transformed.X_test.shape,
            (2, 1),
        )

        self.assertEqual(
            transformed.feature_names,
            ("x0",),
        )

    def test_drop_constant_datetime_features(self):
        """Проверява премахването на постоянни датни характеристики."""

        dataset = self._create_dataset(
            columns=("Date", "Target"),
            rows=(
                ("2026-01-01", "100"),
                ("2026-02-15", "200"),
                ("2026-03-20", "300"),
                ("2026-04-25", "400"),

                # Test.
                ("2026-05-10", "500"),
                ("2027-06-15 14:30:45", "600"),
            ),
        )

        configuration = PreprocessingConfiguration(
            drop_constant_features=True,
        )

        prepared, transformed, report = (
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
            report.removed_constant_count,
            4,
        )

        self.assertEqual(
            set(report.removed_constant_features),
            {
                "x0_year",
                "x0_hour",
                "x0_minute",
                "x0_second",
            },
        )

        self.assertEqual(
            transformed.feature_count,
            5,
        )

        self.assertEqual(
            transformed.feature_names,
            (
                "x0_month",
                "x0_day",
                "x0_day_of_week",
                "x0_quarter",
                "x0_day_of_year",
            ),
        )

        self.assertEqual(
            transformed.X_train.shape,
            (4, 5),
        )

        self.assertEqual(
            transformed.X_test.shape,
            (2, 5),
        )

        self.assertTrue(
            np.isfinite(transformed.X_train).all()
        )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

    def test_all_empty_columns_are_rejected(self):
        """Проверява случая без оставащи входни колони."""

        dataset = self._create_dataset(
            columns=("Notes", "Description", "Target"),
            rows=(
                ("", "", "100"),
                ("", "", "200"),
                ("", "", "300"),
                ("", "", "400"),

                # Test.
                ("Бележка", "", "500"),
                ("", "Описание", "600"),
            ),
        )

        configuration = PreprocessingConfiguration(
            drop_empty_columns=True,
        )

        with self.assertRaisesRegex(
            TransformationError,
            "не остава нито една входна характеристика",
        ):
            self._transform_dataset(
                dataset=dataset,
                feature_columns=(
                    "Notes",
                    "Description",
                ),
                train_count=4,
                configuration=configuration,
            )

    def test_all_constant_features_are_rejected(self):
        """Проверява случая без променливи характеристики."""

        dataset = self._create_dataset(
            columns=("Value", "Country", "Target"),
            rows=(
                ("10", "България", "100"),
                ("10", "България", "200"),
                ("10", "България", "300"),
                ("10", "България", "400"),

                # Test.
                ("20", "Германия", "500"),
                ("30", "Франция", "600"),
            ),
        )

        configuration = PreprocessingConfiguration(
            drop_constant_features=True,
        )

        with self.assertRaises(TransformationError):
            self._transform_dataset(
                dataset=dataset,
                feature_columns=(
                    "Value",
                    "Country",
                ),
                train_count=4,
                configuration=configuration,
            )

    def test_invalid_feature_selection_configuration(self):
        """Проверява отхвърлянето на невалидни настройки."""

        invalid_configurations = (
            {"drop_empty_columns": "True"},
            {"drop_empty_columns": 1},
            {"drop_empty_columns": None},
            {"drop_empty_columns": []},
            {"drop_constant_features": "False"},
            {"drop_constant_features": 0},
            {"drop_constant_features": None},
            {"drop_constant_features": []},
        )

        for parameters in invalid_configurations:

            with self.subTest(parameters=parameters):

                with self.assertRaises(PreprocessingError):

                    PreprocessingConfiguration(
                        **parameters
                    )

    def test_feature_selection_prevents_data_leakage(self):
        """Проверява независимостта на Training от Test."""

        columns = (
            "Value",
            "Constant",
            "Notes",
            "Target",
        )

        training_rows = (
            ("10", "5", "", "100"),
            ("20", "5", "", "200"),
            ("30", "5", "", "300"),
            ("40", "5", "", "400"),
        )

        first_dataset = self._create_dataset(
            columns=columns,
            rows=training_rows + (
                ("50", "5", "", "500"),
                ("60", "5", "", "600"),
            ),
        )

        second_dataset = self._create_dataset(
            columns=columns,
            rows=training_rows + (
                ("50000", "1000", "Бележка", "500"),
                ("60000", "2000", "Описание", "600"),
            ),
        )

        configuration = PreprocessingConfiguration(
            drop_empty_columns=True,
            drop_constant_features=True,
        )

        feature_columns = (
            "Value",
            "Constant",
            "Notes",
        )

        first_prepared, first_transformed, first_report = (
            self._transform_dataset(
                dataset=first_dataset,
                feature_columns=feature_columns,
                train_count=4,
                configuration=configuration,
            )
        )

        second_prepared, second_transformed, second_report = (
            self._transform_dataset(
                dataset=second_dataset,
                feature_columns=feature_columns,
                train_count=4,
                configuration=configuration,
            )
        )

        self.assertEqual(
            first_prepared.empty_columns,
            second_prepared.empty_columns,
        )

        self.assertEqual(
            first_report,
            second_report,
        )

        self.assertEqual(
            second_report.removed_empty_columns,
            ("Notes",),
        )

        self.assertEqual(
            second_report.removed_constant_features,
            ("x1",),
        )

        self.assertEqual(
            first_transformed.feature_names,
            second_transformed.feature_names,
        )

        np.testing.assert_allclose(
            first_transformed.X_train,
            second_transformed.X_train,
        )

        first_variance = (
            first_transformed.fitted_preprocessor
            .named_steps["variance"]
        )

        second_variance = (
            second_transformed.fitted_preprocessor
            .named_steps["variance"]
        )

        np.testing.assert_array_equal(
            first_variance.get_support(),
            second_variance.get_support(),
        )

        np.testing.assert_allclose(
            first_variance.variances_,
            second_variance.variances_,
        )

        self.assertEqual(
            first_transformed.feature_count,
            1,
        )

        self.assertEqual(
            second_transformed.feature_count,
            1,
        )

    def test_feature_selection_with_most_frequent(self):
        """Проверява съвместимостта с Most Frequent Imputation."""

        dataset = self._create_dataset(
            columns=(
                "Value",
                "City",
                "Notes",
                "Target",
            ),
            rows=(
                ("10", "Варна", "", "100"),
                ("20", "София", "", "200"),
                ("30", "Варна", "", "300"),
                ("40", "", "", "400"),

                # Test.
                ("50", "", "Бележка", "500"),
                ("60", "Пловдив", "", "600"),
            ),
        )

        configuration = PreprocessingConfiguration(
            categorical_imputation_strategy="most_frequent",
            drop_empty_columns=True,
            drop_constant_features=True,
        )

        _, transformed, report = (
            self._transform_dataset(
                dataset=dataset,
                feature_columns=(
                    "Value",
                    "City",
                    "Notes",
                ),
                train_count=4,
                configuration=configuration,
            )
        )

        self.assertEqual(
            report.removed_empty_columns,
            ("Notes",),
        )

        self.assertEqual(
            report.removed_constant_features,
            (),
        )

        self.assertEqual(
            transformed.feature_count,
            3,
        )

        column_transformer = (
            transformed.fitted_preprocessor
            .named_steps["columns"]
        )

        self.assertNotIn(
            "categorical_empty",
            column_transformer.named_transformers_,
        )

        categorical_pipeline = (
            column_transformer
            .named_transformers_["categorical"]
        )

        imputer = categorical_pipeline.named_steps[
            "imputer"
        ]

        self.assertEqual(
            imputer.statistics_[0],
            "Варна",
        )

        encoder = categorical_pipeline.named_steps[
            "encoder"
        ]

        self.assertEqual(
            set(encoder.categories_[0]),
            {"Варна", "София"},
        )

        self.assertTrue(
            np.isfinite(transformed.X_train).all()
        )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

    def test_standard_preparation_mode(self):
        """Проверява съвместимостта със стандартния режим."""

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
                ("50", "10", "", "500"),
                ("60", "15", "", "600"),
            ),
        )

        configuration = PreprocessingConfiguration(
            drop_empty_columns=True,
            drop_constant_features=True,
        )

        prepared, transformed, report = (
            self._transform_dataset(
                dataset=dataset,
                feature_columns=(
                    "Value",
                    "Constant",
                    "Notes",
                ),
                train_count=4,
                configuration=configuration,
                defer_type_detection=False,
            )
        )

        self.assertEqual(
            prepared.numeric_columns,
            ("Value", "Constant"),
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
            transformed.feature_names,
            ("x0",),
        )

        self.assertEqual(
            transformed.feature_count,
            1,
        )

        self.assertTrue(
            np.isfinite(transformed.X_train).all()
        )

        self.assertTrue(
            np.isfinite(transformed.X_test).all()
        )

    def test_training_with_feature_selection(self):
        """Проверява реално обучение след филтриране."""

        dataset = self._create_dataset(
            columns=(
                "Value",
                "Constant",
                "Notes",
                "Target",
            ),
            rows=tuple(
                (
                    str(index * 10),
                    "5",
                    "",
                    str(index * 100),
                )
                for index in range(1, 21)
            ),
        )

        configuration = PreprocessingConfiguration(
            drop_empty_columns=True,
            drop_constant_features=True,
        )

        experiment = train_experiment_model(
            dataset=dataset,
            target_column="Target",
            feature_columns=(
                "Value",
                "Constant",
                "Notes",
            ),
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

        report = get_feature_selection_report(
            prepared_data=experiment.prepared_data,
            fitted_preprocessor=(
                experiment.transformed_data.fitted_preprocessor
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