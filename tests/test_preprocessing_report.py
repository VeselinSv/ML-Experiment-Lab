"""Тестове за подробния отчет за предварителна обработка."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sklearn.model_selection import train_test_split

from ml_experiment_lab.datasets import LoadedDataset

from ml_experiment_lab.preprocessing import (
    PreprocessingConfiguration,
)

from ml_experiment_lab.preprocessing_report import (
    CategoryReport,
    ColumnPreprocessingReport,
    FeatureReport,
    PreprocessingReport,
    PreprocessingReportError,
    build_preprocessing_report,
)

from ml_experiment_lab.training import (
    train_experiment_model,
)


class PreprocessingReportTests(unittest.TestCase):
    """Проверява съдържанието на Preprocessing отчета."""

    def _create_experiment(
        self,
        columns: tuple[str, ...],
        training_rows: tuple[tuple[str, ...], ...],
        test_rows: tuple[tuple[str, ...], ...],
        configuration: PreprocessingConfiguration | None = None,
        excluded_rows: tuple[tuple[str, ...], ...] = (),
    ):
        """Създава реален експеримент с контролирани извадки."""

        random_state = 42

        row_count = (
            len(training_rows)
            + len(test_rows)
        )

        test_size = len(test_rows) / row_count

        # Определяме позициите, които ще попаднат
        # в тренировъчната и тестовата извадка.
        training_indexes, test_indexes = train_test_split(
            np.arange(row_count),
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        self.assertEqual(
            len(training_indexes),
            len(training_rows),
        )

        self.assertEqual(
            len(test_indexes),
            len(test_rows),
        )

        ordered_rows = [None] * row_count

        for index, row in zip(
            training_indexes,
            training_rows,
        ):
            ordered_rows[int(index)] = row

        for index, row in zip(
            test_indexes,
            test_rows,
        ):
            ordered_rows[int(index)] = row

        dataset = LoadedDataset(
            path=Path("preprocessing_report.xlsx"),
            columns=columns,
            rows=(
                tuple(ordered_rows)
                + excluded_rows
            ),
        )

        feature_columns = tuple(
            column
            for column in columns
            if column != "Target"
        )

        experiment = train_experiment_model(
            dataset=dataset,
            target_column="Target",
            feature_columns=feature_columns,
            task_type="regression",
            algorithm="linear_regression",
            test_size=test_size,
            random_state=random_state,
            preprocessing_configuration=configuration,
        )

        return experiment

    def _create_numeric_experiment(
        self,
        configuration: PreprocessingConfiguration | None = None,
    ):
        """Създава експеримент с числова характеристика."""

        return self._create_experiment(
            columns=("Value", "Target"),
            training_rows=(
                ("10", "100"),
                ("20", "200"),
                ("30", "300"),
                ("40", "400"),
            ),
            test_rows=(
                ("50", "500"),
                ("60", "600"),
            ),
            configuration=configuration,
        )

    def _create_categorical_experiment(
        self,
        configuration: PreprocessingConfiguration | None = None,
    ):
        """Създава експеримент с категориална характеристика."""

        return self._create_experiment(
            columns=("City", "Target"),
            training_rows=(
                ("Варна", "100"),
                ("София", "200"),
                ("Варна", "300"),
                ("Пловдив", "400"),
                ("Варна", "500"),
            ),
            test_rows=(
                ("Бургас", "600"),
                ("Варна", "700"),
            ),
            configuration=configuration,
        )

    def test_report_summary(self):
        """Проверява обобщението на наблюденията."""

        experiment = self._create_experiment(
            columns=("Value", "Target"),
            training_rows=(
                ("10", "100"),
                ("20", "200"),
                ("30", "300"),
                ("40", "400"),
            ),
            test_rows=(
                ("50", "500"),
                ("60", "600"),
            ),
            excluded_rows=(
                ("70", ""),
                ("80", ""),
            ),
        )

        report = build_preprocessing_report(
            experiment
        )

        self.assertIsInstance(
            report,
            PreprocessingReport,
        )

        self.assertEqual(
            report.target_column,
            "Target",
        )

        self.assertEqual(
            report.original_row_count,
            8,
        )

        self.assertEqual(
            report.excluded_row_count,
            2,
        )

        self.assertEqual(
            report.valid_row_count,
            6,
        )

        self.assertEqual(
            report.training_row_count,
            4,
        )

        self.assertEqual(
            report.test_row_count,
            2,
        )

        self.assertEqual(
            report.random_state,
            42,
        )

        self.assertAlmostEqual(
            report.requested_test_size,
            2 / 6,
        )

        self.assertAlmostEqual(
            report.actual_test_size,
            2 / 6,
        )

        self.assertEqual(
            report.input_feature_count,
            1,
        )

        self.assertEqual(
            report.generated_feature_count,
            1,
        )

        self.assertEqual(
            report.final_feature_count,
            1,
        )

        self.assertEqual(
            report.configuration,
            experiment.preprocessing_configuration,
        )

    def test_column_types(self):
        """Проверява типовете на входните колони."""

        experiment = self._create_experiment(
            columns=(
                "Value",
                "City",
                "Date",
                "Mixed",
                "Notes",
                "Target",
            ),
            training_rows=(
                (
                    "10", "Варна", "2026-01-01",
                    "100", "", "100",
                ),
                (
                    "20", "София", "2026-02-01",
                    "Текст", "", "200",
                ),
                (
                    "30", "Варна", "2026-03-01",
                    "200", "", "300",
                ),
                (
                    "40", "София", "2026-04-01",
                    "Друг текст", "", "400",
                ),
            ),
            test_rows=(
                (
                    "50", "Бургас", "2026-05-01",
                    "300", "Бележка", "500",
                ),
                (
                    "60", "Варна", "2026-06-01",
                    "Текст", "", "600",
                ),
            ),
        )

        report = build_preprocessing_report(
            experiment
        )

        self.assertEqual(
            report.input_feature_count,
            5,
        )

        self.assertEqual(
            report.numeric_column_count,
            1,
        )

        self.assertEqual(
            report.categorical_column_count,
            2,
        )

        self.assertEqual(
            report.datetime_column_count,
            1,
        )

        self.assertEqual(
            report.empty_column_count,
            1,
        )

        self.assertEqual(
            report.mixed_column_count,
            1,
        )

        expected_types = {
            "Value": "numeric",
            "City": "categorical",
            "Date": "datetime",
            "Mixed": "categorical",
            "Notes": "empty",
        }

        for column_name, expected_type in (
            expected_types.items()
        ):
            with self.subTest(column=column_name):

                column = report.get_column_report(
                    column_name
                )

                self.assertIsInstance(
                    column,
                    ColumnPreprocessingReport,
                )

                self.assertEqual(
                    column.column_type,
                    expected_type,
                )

        self.assertTrue(
            report.get_column_report("Mixed").is_mixed
        )

        self.assertTrue(
            report.get_column_report("Notes").is_empty
        )

    def test_numeric_missing_and_invalid_values(self):
        """Проверява липсващи и невалидни числови стойности."""

        experiment = self._create_experiment(
            columns=("Value", "Target"),
            training_rows=(
                ("10", "100"),
                ("20", "200"),
                ("", "300"),
                ("30", "400"),
            ),
            test_rows=(
                ("", "500"),
                ("Неизвестен", "600"),
            ),
        )

        report = build_preprocessing_report(
            experiment
        )

        column = report.get_column_report(
            "Value"
        )

        self.assertEqual(
            column.training_missing_count,
            1,
        )

        self.assertEqual(
            column.test_missing_count,
            1,
        )

        self.assertEqual(
            column.training_invalid_count,
            0,
        )

        self.assertEqual(
            column.test_invalid_count,
            1,
        )

        self.assertEqual(
            column.total_missing_count,
            2,
        )

        self.assertEqual(
            column.total_invalid_count,
            1,
        )

        self.assertEqual(
            report.total_missing_count,
            2,
        )

        self.assertEqual(
            report.total_invalid_count,
            1,
        )

        self.assertEqual(
            column.imputation_strategy,
            "median",
        )

        self.assertEqual(
            column.generated_feature_count,
            1,
        )

        feature = column.features[0]

        self.assertIsInstance(
            feature,
            FeatureReport,
        )

        self.assertAlmostEqual(
            feature.imputation_value,
            20.0,
        )

        self.assertEqual(
            feature.display_name,
            "Value",
        )

        self.assertTrue(
            feature.is_retained
        )

    def test_datetime_missing_and_invalid_values(self):
        """Проверява липсващи и невалидни дати."""

        experiment = self._create_experiment(
            columns=("Date", "Target"),
            training_rows=(
                ("2026-01-01", "100"),
                ("2026-01-03", "200"),
                ("", "300"),
                ("2026-01-05", "400"),
            ),
            test_rows=(
                ("", "500"),
                ("31.02.2026", "600"),
            ),
        )

        report = build_preprocessing_report(
            experiment
        )

        column = report.get_column_report(
            "Date"
        )

        self.assertEqual(
            column.column_type,
            "datetime",
        )

        self.assertEqual(
            column.training_missing_count,
            1,
        )

        self.assertEqual(
            column.test_missing_count,
            1,
        )

        self.assertEqual(
            column.training_invalid_count,
            0,
        )

        self.assertEqual(
            column.test_invalid_count,
            1,
        )

        self.assertEqual(
            column.total_missing_count,
            2,
        )

        self.assertEqual(
            column.total_invalid_count,
            1,
        )

        self.assertEqual(
            column.imputation_strategy,
            "median",
        )

        self.assertEqual(
            column.generated_feature_count,
            9,
        )

        expected_imputation_values = {
            "x0_year": 2026.0,
            "x0_month": 1.0,
            "x0_day": 3.0,
            "x0_day_of_week": 3.0,
            "x0_quarter": 1.0,
            "x0_day_of_year": 3.0,
            "x0_hour": 0.0,
            "x0_minute": 0.0,
            "x0_second": 0.0,
        }

        for feature in column.features:

            with self.subTest(
                feature=feature.technical_name
            ):

                self.assertAlmostEqual(
                    feature.imputation_value,
                    expected_imputation_values[
                        feature.technical_name
                    ],
                )

                self.assertEqual(
                    feature.source_column,
                    "Date",
                )

    def test_scaling_parameters(self):
        """Проверява параметрите на всички Scaling методи."""

        expected_parameters = {
            "standard": {
                "mean": 25.0,
                "variance": 125.0,
                "scale": np.sqrt(125.0),
            },
            "minmax": {
                "minimum": 10.0,
                "maximum": 40.0,
                "range": 30.0,
                "scale": 1 / 30,
                "offset": -1 / 3,
            },
            "robust": {
                "center": 25.0,
                "scale": 15.0,
            },
            "maxabs": {
                "max_abs": 40.0,
                "scale": 40.0,
            },
            "none": {},
        }

        for method, expected in (
            expected_parameters.items()
        ):

            with self.subTest(method=method):

                configuration = (
                    PreprocessingConfiguration(
                        scaling_method=method,
                    )
                )

                experiment = (
                    self._create_numeric_experiment(
                        configuration
                    )
                )

                report = build_preprocessing_report(
                    experiment
                )

                column = report.get_column_report(
                    "Value"
                )

                self.assertEqual(
                    column.scaling_method,
                    method,
                )

                actual = dict(
                    column.features[0].scaling_parameters
                )

                self.assertEqual(
                    set(actual),
                    set(expected),
                )

                for parameter_name, expected_value in (
                    expected.items()
                ):

                    self.assertAlmostEqual(
                        actual[parameter_name],
                        expected_value,
                    )

    def test_onehot_encoding_report(self):
        """Проверява отчета за One-Hot Encoding."""

        configuration = PreprocessingConfiguration(
            encoding_method="onehot",
        )

        experiment = (
            self._create_categorical_experiment(
                configuration
            )
        )

        report = build_preprocessing_report(
            experiment
        )

        column = report.get_column_report(
            "City"
        )

        self.assertEqual(
            column.encoding_method,
            "onehot",
        )

        self.assertEqual(
            column.learned_category_count,
            3,
        )

        self.assertEqual(
            column.generated_feature_count,
            3,
        )

        self.assertEqual(
            column.categories,
            (
                CategoryReport(
                    category="Варна",
                    feature_name="x0_Варна",
                ),
                CategoryReport(
                    category="Пловдив",
                    feature_name="x0_Пловдив",
                ),
                CategoryReport(
                    category="София",
                    feature_name="x0_София",
                ),
            ),
        )

        self.assertEqual(
            report.final_feature_names,
            (
                "City — Варна",
                "City — Пловдив",
                "City — София",
            ),
        )

        self.assertEqual(
            column.training_unknown_count,
            0,
        )

        self.assertEqual(
            column.test_unknown_count,
            1,
        )

        self.assertEqual(
            report.total_unknown_count,
            1,
        )

        self.assertIsNone(
            column.scaling_method
        )

        for feature in column.features:

            self.assertEqual(
                feature.scaling_parameters,
                (),
            )

    def test_ordinal_encoding_report(self):
        """Проверява отчета за Ordinal Encoding."""

        configuration = PreprocessingConfiguration(
            encoding_method="ordinal",
        )

        experiment = (
            self._create_categorical_experiment(
                configuration
            )
        )

        report = build_preprocessing_report(
            experiment
        )

        column = report.get_column_report(
            "City"
        )

        self.assertEqual(
            column.encoding_method,
            "ordinal",
        )

        self.assertEqual(
            column.generated_feature_count,
            1,
        )

        self.assertEqual(
            column.learned_category_count,
            3,
        )

        actual_codes = {
            category.category: category.encoded_value
            for category in column.categories
        }

        self.assertEqual(
            actual_codes,
            {
                "Варна": 0.0,
                "Пловдив": 1.0,
                "София": 2.0,
            },
        )

        self.assertEqual(
            report.final_feature_names,
            ("City",),
        )

        self.assertEqual(
            report.final_technical_feature_names,
            ("x0",),
        )

        self.assertEqual(
            column.test_unknown_count,
            1,
        )

        self.assertTrue(
            any(
                "изкуствена" in warning
                for warning in report.warnings
            )
        )

    def test_frequency_encoding_report(self):
        """Проверява отчета за Frequency Encoding."""

        configuration = PreprocessingConfiguration(
            encoding_method="frequency",
        )

        experiment = (
            self._create_categorical_experiment(
                configuration
            )
        )

        report = build_preprocessing_report(
            experiment
        )

        column = report.get_column_report(
            "City"
        )

        self.assertEqual(
            column.encoding_method,
            "frequency",
        )

        self.assertEqual(
            column.learned_category_count,
            3,
        )

        self.assertEqual(
            column.generated_feature_count,
            1,
        )

        frequencies = {
            category.category: category.encoded_value
            for category in column.categories
        }

        self.assertEqual(
            frequencies,
            {
                "Варна": 0.6,
                "Пловдив": 0.2,
                "София": 0.2,
            },
        )

        self.assertAlmostEqual(
            sum(frequencies.values()),
            1.0,
        )

        self.assertNotIn(
            "Бургас",
            frequencies,
        )

        self.assertEqual(
            column.test_unknown_count,
            1,
        )

        self.assertEqual(
            report.final_feature_names,
            ("City",),
        )

        self.assertTrue(
            any(
                "еднаква числова стойност" in warning
                for warning in report.warnings
            )
        )

    def test_unknown_categories_after_imputation(self):
        """Проверява непознати категории след Imputation."""

        for method in (
            "onehot",
            "ordinal",
            "frequency",
        ):

            with self.subTest(method=method):

                configuration = (
                    PreprocessingConfiguration(
                        categorical_imputation_strategy="constant",
                        categorical_fill_value="Неизвестен",
                        encoding_method=method,
                    )
                )

                experiment = self._create_experiment(
                    columns=("City", "Target"),
                    training_rows=(
                        ("Варна", "100"),
                        ("София", "200"),
                        ("Варна", "300"),
                        ("София", "400"),
                    ),
                    test_rows=(
                        ("", "500"),
                        ("Бургас", "600"),
                    ),
                    configuration=configuration,
                )

                report = build_preprocessing_report(
                    experiment
                )

                column = report.get_column_report(
                    "City"
                )

                self.assertEqual(
                    column.imputation_strategy,
                    "constant",
                )

                self.assertEqual(
                    column.categorical_imputation_value,
                    "Неизвестен",
                )

                self.assertEqual(
                    column.training_missing_count,
                    0,
                )

                self.assertEqual(
                    column.test_missing_count,
                    1,
                )

                self.assertEqual(
                    column.training_unknown_count,
                    0,
                )

                # Празната стойност става "Неизвестен".
                # Тази категория и Бургас не са в Training.
                self.assertEqual(
                    column.test_unknown_count,
                    2,
                )

                self.assertEqual(
                    column.total_unknown_count,
                    2,
                )

                self.assertEqual(
                    report.total_unknown_count,
                    2,
                )

                learned_categories = {
                    category.category
                    for category in column.categories
                }

                self.assertNotIn(
                    "Неизвестен",
                    learned_categories,
                )

                self.assertNotIn(
                    "Бургас",
                    learned_categories,
                )

    def test_empty_column_fallback_report(self):
        """Проверява резервната обработка на празни колони."""

        configuration = PreprocessingConfiguration(
            categorical_imputation_strategy="most_frequent",
            categorical_fill_value="Неизвестен",
            encoding_method="frequency",
            drop_empty_columns=False,
        )

        experiment = self._create_experiment(
            columns=("Value", "Notes", "Target"),
            training_rows=(
                ("10", "", "100"),
                ("20", "", "200"),
                ("30", "", "300"),
                ("40", "", "400"),
            ),
            test_rows=(
                ("50", "", "500"),
                ("60", "Бележка", "600"),
            ),
            configuration=configuration,
        )

        report = build_preprocessing_report(
            experiment
        )

        column = report.get_column_report(
            "Notes"
        )

        self.assertTrue(
            column.is_empty
        )

        self.assertFalse(
            column.removed_as_empty
        )

        self.assertEqual(
            column.column_type,
            "empty",
        )

        self.assertEqual(
            column.training_missing_count,
            4,
        )

        self.assertEqual(
            column.test_missing_count,
            1,
        )

        # Избраната стратегия е most_frequent,
        # но за празната колона е използвана constant.
        self.assertEqual(
            report.configuration
            .categorical_imputation_strategy,
            "most_frequent",
        )

        self.assertEqual(
            column.imputation_strategy,
            "constant",
        )

        self.assertEqual(
            column.categorical_imputation_value,
            "Неизвестен",
        )

        self.assertEqual(
            column.learned_category_count,
            1,
        )

        self.assertEqual(
            column.categories[0].category,
            "Неизвестен",
        )

        self.assertAlmostEqual(
            column.categories[0].encoded_value,
            1.0,
        )

        self.assertEqual(
            column.test_unknown_count,
            1,
        )

        self.assertEqual(
            column.retained_feature_count,
            1,
        )

    def test_removed_features_report(self):
        """Проверява премахнатите празни и постоянни колони."""

        configuration = PreprocessingConfiguration(
            drop_empty_columns=True,
            drop_constant_features=True,
            encoding_method="onehot",
        )

        experiment = self._create_experiment(
            columns=(
                "Value",
                "Country",
                "Notes",
                "Target",
            ),
            training_rows=(
                ("10", "България", "", "100"),
                ("20", "България", "", "200"),
                ("30", "България", "", "300"),
                ("40", "България", "", "400"),
            ),
            test_rows=(
                ("50", "Германия", "Бележка", "500"),
                ("60", "Франция", "", "600"),
            ),
            configuration=configuration,
        )

        report = build_preprocessing_report(
            experiment
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
            ("Country — България",),
        )

        self.assertEqual(
            report.removed_constant_technical_names,
            ("x1_България",),
        )

        self.assertEqual(
            report.removed_constant_count,
            1,
        )

        self.assertEqual(
            report.input_feature_count,
            3,
        )

        self.assertEqual(
            report.generated_feature_count,
            2,
        )

        self.assertEqual(
            report.final_feature_count,
            1,
        )

        self.assertEqual(
            report.final_feature_names,
            ("Value",),
        )

        empty_column = report.get_column_report(
            "Notes"
        )

        self.assertTrue(
            empty_column.removed_as_empty
        )

        self.assertEqual(
            empty_column.generated_feature_count,
            0,
        )

        self.assertIsNone(
            empty_column.imputation_strategy
        )

        self.assertIsNone(
            empty_column.encoding_method
        )

        constant_column = report.get_column_report(
            "Country"
        )

        self.assertFalse(
            constant_column.removed_as_empty
        )

        self.assertEqual(
            constant_column.removed_constant_count,
            1,
        )

        self.assertEqual(
            constant_column.retained_feature_count,
            0,
        )

        self.assertTrue(
            constant_column.features[0].removed_as_constant
        )

        self.assertFalse(
            constant_column.features[0].is_retained
        )

    def test_removed_datetime_features_report(self):
        """Проверява премахнатите датни характеристики."""

        configuration = PreprocessingConfiguration(
            drop_constant_features=True,
        )

        experiment = self._create_experiment(
            columns=("Date", "Target"),
            training_rows=(
                ("2026-01-01", "100"),
                ("2026-02-15", "200"),
                ("2026-03-20", "300"),
                ("2026-04-25", "400"),
            ),
            test_rows=(
                ("2026-05-10", "500"),
                ("2027-06-15 14:30:45", "600"),
            ),
            configuration=configuration,
        )

        report = build_preprocessing_report(
            experiment
        )

        column = report.get_column_report(
            "Date"
        )

        self.assertEqual(
            column.generated_feature_count,
            9,
        )

        self.assertEqual(
            column.retained_feature_count,
            5,
        )

        self.assertEqual(
            column.removed_constant_count,
            4,
        )

        self.assertEqual(
            report.removed_constant_features,
            (
                "Date — година",
                "Date — час",
                "Date — минута",
                "Date — секунда",
            ),
        )

        self.assertEqual(
            report.final_feature_names,
            (
                "Date — месец",
                "Date — ден",
                "Date — ден от седмицата",
                "Date — тримесечие",
                "Date — ден от годината",
            ),
        )

        self.assertEqual(
            report.generated_feature_count,
            9,
        )

        self.assertEqual(
            report.final_feature_count,
            5,
        )

    def test_feature_names_and_order(self):
        """Проверява имената и реда на характеристиките."""

        experiment = self._create_experiment(
            columns=(
                "Date",
                "City",
                "Value",
                "Country",
                "Target",
            ),
            training_rows=(
                (
                    "2026-01-01", "Варна",
                    "10", "България", "100",
                ),
                (
                    "2026-02-01", "София",
                    "20", "Германия", "200",
                ),
                (
                    "2026-03-01", "Варна",
                    "30", "България", "300",
                ),
                (
                    "2026-04-01", "София",
                    "40", "Германия", "400",
                ),
            ),
            test_rows=(
                (
                    "2026-05-01", "Бургас",
                    "50", "Франция", "500",
                ),
                (
                    "2026-06-01", "Варна",
                    "60", "България", "600",
                ),
            ),
        )

        report = build_preprocessing_report(
            experiment
        )

        # Входните колони запазват оригиналния си ред.
        self.assertEqual(
            tuple(
                column.column_name
                for column in report.columns
            ),
            (
                "Date",
                "City",
                "Value",
                "Country",
            ),
        )

        # Изходните характеристики следват реда
        # на действително обучения ColumnTransformer.
        expected_technical_names = (
            "x2",
            "x1_Варна",
            "x1_София",
            "x3_България",
            "x3_Германия",
            "x0_year",
            "x0_month",
            "x0_day",
            "x0_day_of_week",
            "x0_quarter",
            "x0_day_of_year",
            "x0_hour",
            "x0_minute",
            "x0_second",
        )

        expected_display_names = (
            "Value",
            "City — Варна",
            "City — София",
            "Country — България",
            "Country — Германия",
            "Date — година",
            "Date — месец",
            "Date — ден",
            "Date — ден от седмицата",
            "Date — тримесечие",
            "Date — ден от годината",
            "Date — час",
            "Date — минута",
            "Date — секунда",
        )

        self.assertEqual(
            report.final_technical_feature_names,
            expected_technical_names,
        )

        self.assertEqual(
            report.final_feature_names,
            expected_display_names,
        )

        self.assertEqual(
            report.final_technical_feature_names,
            experiment.transformed_data.feature_names,
        )

        self.assertEqual(
            report.final_feature_count,
            experiment.feature_count,
        )

        self.assertEqual(
            report.final_feature_count,
            14,
        )

    def test_report_warnings(self):
        """Проверява диагностичните предупреждения."""

        configuration = PreprocessingConfiguration(
            encoding_method="frequency",
            drop_empty_columns=True,
            drop_constant_features=True,
        )

        experiment = self._create_experiment(
            columns=(
                "Value",
                "Mixed",
                "City",
                "Notes",
                "Target",
            ),
            training_rows=(
                ("10", "100", "Варна", "", "100"),
                ("20", "Текст", "София", "", "200"),
                ("30", "100", "Варна", "", "300"),
                ("40", "Текст", "София", "", "400"),
            ),
            test_rows=(
                (
                    "Неизвестен", "200",
                    "Бургас", "Бележка", "500",
                ),
                (
                    "60", "Текст",
                    "Варна", "", "600",
                ),
            ),
            excluded_rows=(
                ("70", "100", "София", "", ""),
            ),
            configuration=configuration,
        )

        report = build_preprocessing_report(
            experiment
        )

        expected_messages = (
            "липсваща целева стойност",
            "смесени типове",
            "изцяло празна в Training",
            "невалидни стойности",
            "непознати категориални стойности",
            "постоянни характеристики",
            "еднаква числова стойност",
        )

        combined_warnings = " ".join(
            report.warnings
        )

        for expected_message in expected_messages:

            with self.subTest(
                message=expected_message
            ):

                self.assertIn(
                    expected_message,
                    combined_warnings,
                )

        self.assertGreaterEqual(
            len(report.warnings),
            7,
        )

    def test_get_column_report(self):
        """Проверява достъпа до отчет за конкретна колона."""

        experiment = (
            self._create_numeric_experiment()
        )

        report = build_preprocessing_report(
            experiment
        )

        column = report.get_column_report(
            "Value"
        )

        self.assertIsInstance(
            column,
            ColumnPreprocessingReport,
        )

        self.assertEqual(
            column.column_name,
            "Value",
        )

        self.assertEqual(
            column.column_type,
            "numeric",
        )

        self.assertEqual(
            column,
            report.columns[0],
        )

        with self.assertRaisesRegex(
            PreprocessingReportError,
            "не присъства в отчета",
        ):
            report.get_column_report(
                "UnknownColumn"
            )

        # Target не е входна характеристика.
        with self.assertRaises(
            PreprocessingReportError
        ):
            report.get_column_report(
                "Target"
            )

    def test_report_does_not_modify_experiment(self):
        """Проверява създаването на отчет без промени."""

        configuration = PreprocessingConfiguration(
            encoding_method="frequency",
            scaling_method="robust",
        )

        experiment = self._create_experiment(
            columns=("Value", "City", "Target"),
            training_rows=(
                ("10", "Варна", "100"),
                ("20", "София", "200"),
                ("30", "Варна", "300"),
                ("40", "Варна", "400"),
            ),
            test_rows=(
                ("50", "Бургас", "500"),
                ("60", "Варна", "600"),
            ),
            configuration=configuration,
        )

        preprocessor = (
            experiment.transformed_data.fitted_preprocessor
        )

        estimator = experiment.estimator

        original_raw_train = (
            experiment.data_split.X_train.copy()
        )

        original_raw_test = (
            experiment.data_split.X_test.copy()
        )

        original_transformed_train = (
            experiment.transformed_data.X_train.copy()
        )

        original_transformed_test = (
            experiment.transformed_data.X_test.copy()
        )

        original_coefficients = (
            estimator.coef_.copy()
        )

        original_intercept = (
            estimator.intercept_
        )

        numeric_pipeline = (
            preprocessor.named_transformers_["numeric"]
        )

        categorical_pipeline = (
            preprocessor.named_transformers_["categorical"]
        )

        scaler = numeric_pipeline.named_steps[
            "scaler"
        ]

        imputer = numeric_pipeline.named_steps[
            "imputer"
        ]

        encoder = categorical_pipeline.named_steps[
            "encoder"
        ]

        original_center = scaler.center_.copy()

        original_scale = scaler.scale_.copy()

        original_statistics = (
            imputer.statistics_.copy()
        )

        original_frequencies = tuple(
            frequency_map.copy()
            for frequency_map in encoder.frequency_maps_
        )

        # Отчетът не трябва да извиква обучение,
        # трансформация или прогнозиране.
        with (
            patch.object(
                preprocessor,
                "fit",
                side_effect=AssertionError(
                    "Неочаквано повторно обучение."
                ),
            ) as fit_mock,
            patch.object(
                preprocessor,
                "fit_transform",
                side_effect=AssertionError(
                    "Неочакван fit_transform."
                ),
            ) as fit_transform_mock,
            patch.object(
                preprocessor,
                "transform",
                side_effect=AssertionError(
                    "Неочаквана трансформация."
                ),
            ) as transform_mock,
            patch.object(
                estimator,
                "fit",
                side_effect=AssertionError(
                    "Неочаквано обучение на модела."
                ),
            ) as model_fit_mock,
            patch.object(
                estimator,
                "predict",
                side_effect=AssertionError(
                    "Неочаквано прогнозиране."
                ),
            ) as predict_mock,
        ):

            first_report = (
                build_preprocessing_report(
                    experiment
                )
            )

            second_report = (
                build_preprocessing_report(
                    experiment
                )
            )

            fit_mock.assert_not_called()

            fit_transform_mock.assert_not_called()

            transform_mock.assert_not_called()

            model_fit_mock.assert_not_called()

            predict_mock.assert_not_called()

        # Повторното създаване на отчета
        # трябва да дава еднакъв резултат.
        self.assertEqual(
            first_report,
            second_report,
        )

        np.testing.assert_array_equal(
            experiment.data_split.X_train,
            original_raw_train,
        )

        np.testing.assert_array_equal(
            experiment.data_split.X_test,
            original_raw_test,
        )

        np.testing.assert_array_equal(
            experiment.transformed_data.X_train,
            original_transformed_train,
        )

        np.testing.assert_array_equal(
            experiment.transformed_data.X_test,
            original_transformed_test,
        )

        np.testing.assert_array_equal(
            estimator.coef_,
            original_coefficients,
        )

        self.assertEqual(
            estimator.intercept_,
            original_intercept,
        )

        np.testing.assert_array_equal(
            scaler.center_,
            original_center,
        )

        np.testing.assert_array_equal(
            scaler.scale_,
            original_scale,
        )

        np.testing.assert_array_equal(
            imputer.statistics_,
            original_statistics,
        )

        self.assertEqual(
            encoder.frequency_maps_,
            original_frequencies,
        )


if __name__ == "__main__":
    unittest.main()