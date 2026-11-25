"""Интеграционни тестове за Preprocessing настройките в ModelsPage."""

import os
import unittest
from pathlib import Path

os.environ.setdefault(
    "QT_QPA_PLATFORM",
    "offscreen",
)

import numpy as np

from PySide6.QtCore import Qt

from PySide6.QtWidgets import (
    QApplication,
    QScrollArea,
    QStackedWidget,
    QWidget,
)

from ml_experiment_lab.datasets import LoadedDataset

from ml_experiment_lab.preprocessing import (
    CATEGORICAL_IMPUTATION_STRATEGIES,
    ENCODING_METHODS,
    NUMERIC_IMPUTATION_STRATEGIES,
    SCALING_METHODS,
    PreprocessingConfiguration,
)

from ml_experiment_lab.preprocessing_report import (
    build_preprocessing_report,
)

from ml_experiment_lab.training import (
    train_experiment_model,
)

from ml_experiment_lab.ui.pages.experiments_page import (
    ExperimentConfiguration,
)

from ml_experiment_lab.ui.pages.models_page import (
    ModelsPage,
)

from ml_experiment_lab.ui.widgets.preprocessing_settings_widget import (
    PreprocessingSettingsWidget,
)


class FakeExperimentsPage:
    """Замества страницата за експерименти при тестовете."""

    def __init__(
        self,
        configuration: ExperimentConfiguration | None = None,
    ) -> None:

        self._configuration = configuration

    @property
    def experiment_configuration(
        self,
    ) -> ExperimentConfiguration | None:

        return self._configuration


class ModelsPagePreprocessingTests(unittest.TestCase):
    """Проверява интеграцията на Preprocessing в ModelsPage."""

    @classmethod
    def setUpClass(cls) -> None:
        """Създава QApplication за всички тестове."""

        cls.app = QApplication.instance()

        if cls.app is None:

            cls.app = QApplication([])

    def setUp(self) -> None:
        """Създава нова страница и тестов експеримент."""

        dataset = LoadedDataset(
            path=Path("preprocessing_integration.csv"),
            columns=(
                "Value",
                "City",
                "Country",
                "Notes",
                "Target",
            ),
            rows=tuple(
                (
                    str(index * 10),
                    "Варна" if index <= 13 else "София",
                    "България",
                    "",
                    str(index * 100),
                )
                for index in range(1, 21)
            ),
        )

        self.experiment_configuration = (
            ExperimentConfiguration(
                dataset=dataset,
                target_column="Target",
                feature_columns=(
                    "Value",
                    "City",
                    "Country",
                    "Notes",
                ),
            )
        )

        self.experiments_page = FakeExperimentsPage(
            self.experiment_configuration
        )

        self.page = ModelsPage(
            experiments_page=self.experiments_page,
        )

        self.settings = (
            self.page._preprocessing_settings_widget
        )

        # Използваме QStackedWidget, за да симулираме
        # навигацията между страниците на приложението.
        self.stack = QStackedWidget()

        self.other_page = QWidget()

        self.stack.addWidget(
            self.other_page
        )

        self.stack.addWidget(
            self.page
        )

    def tearDown(self) -> None:
        """Освобождава графичните компоненти."""

        self.stack.close()

        self.stack.deleteLater()

        self.app.processEvents()

    def _select_combo_value(
        self,
        combo_box,
        value: str,
    ) -> None:
        """Избира елемент чрез неговия идентификатор."""

        index = combo_box.findData(value)

        self.assertGreaterEqual(
            index,
            0,
            f"Липсва настройка: {value}",
        )

        combo_box.setCurrentIndex(index)

    def _create_custom_configuration(
        self,
    ) -> PreprocessingConfiguration:
        """Създава персонализирана конфигурация."""

        return PreprocessingConfiguration(
            numeric_imputation_strategy="constant",
            categorical_imputation_strategy="most_frequent",
            numeric_fill_value=10.5,
            categorical_fill_value="Неизвестен",
            drop_empty_columns=True,
            drop_constant_features=True,
            scaling_method="robust",
            encoding_method="frequency",
        )

    def test_preprocessing_widget_is_integrated(self):
        """Проверява интегрирането на графичния компонент."""

        self.assertIsInstance(
            self.settings,
            PreprocessingSettingsWidget,
        )

        found_widget = self.page.findChild(
            PreprocessingSettingsWidget
        )

        self.assertIs(
            found_widget,
            self.settings,
        )

        self.assertTrue(
            self.page.isAncestorOf(
                self.settings
            )
        )

    def test_default_preprocessing_configuration(self):
        """Проверява стандартната конфигурация на страницата."""

        expected = PreprocessingConfiguration()

        self.assertEqual(
            self.page.preprocessing_configuration,
            expected,
        )

        self.assertTrue(
            self.page.is_preprocessing_valid
        )

        self.assertTrue(
            self.page.is_configuration_valid
        )

        self.assertEqual(
            self.page.selected_task_type,
            "classification",
        )

        self.assertEqual(
            self.page.selected_algorithm,
            "logistic_regression",
        )

        self.assertAlmostEqual(
            self.page.test_size,
            0.2,
        )

        self.assertEqual(
            self.page.random_state,
            42,
        )

        self.assertIn(
            "Конфигурацията е валидна",
            self.page._configuration_status_label.text(),
        )

    def test_scroll_area_configuration(self):
        """Проверява настройките на scroll областта."""

        scroll_area = self.page._scroll_area

        self.assertIsInstance(
            scroll_area,
            QScrollArea,
        )

        self.assertTrue(
            scroll_area.widgetResizable()
        )

        self.assertIsNotNone(
            scroll_area.widget()
        )

        self.assertEqual(
            scroll_area.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )

        self.assertEqual(
            scroll_area.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )

        self.assertTrue(
            scroll_area.widget().isAncestorOf(
                self.settings
            )
        )

    def test_custom_preprocessing_configuration(self):
        """Проверява предаването на персонализирани настройки."""

        expected = (
            self._create_custom_configuration()
        )

        self.settings.set_configuration(
            expected
        )

        self.assertEqual(
            self.page.preprocessing_configuration,
            expected,
        )

        self.assertTrue(
            self.page.is_preprocessing_valid
        )

        self.assertTrue(
            self.page.is_configuration_valid
        )

        actual = (
            self.page.preprocessing_configuration
        )

        self.assertEqual(
            actual.numeric_fill_value,
            10.5,
        )

        self.assertEqual(
            actual.categorical_fill_value,
            "Неизвестен",
        )

        self.assertEqual(
            actual.scaling_method,
            "robust",
        )

        self.assertEqual(
            actual.encoding_method,
            "frequency",
        )

        self.assertTrue(
            actual.drop_empty_columns
        )

        self.assertTrue(
            actual.drop_constant_features
        )

    def test_preprocessing_summary_updates(self):
        """Проверява динамичното обобщение на Preprocessing."""

        self.settings.set_configuration(
            self._create_custom_configuration()
        )

        summary = (
            self.page._preprocessing_summary_label.text()
        )

        expected_messages = (
            NUMERIC_IMPUTATION_STRATEGIES["constant"],
            CATEGORICAL_IMPUTATION_STRATEGIES["most_frequent"],
            SCALING_METHODS["robust"],
            ENCODING_METHODS["frequency"],
            "Числова константа: 10.5",
            "Премахване на изцяло празни колони: Да",
            "Премахване на постоянни характеристики: Да",
            "Автоматично извличане на датни характеристики: Да",
        )

        for expected_message in expected_messages:

            with self.subTest(
                message=expected_message
            ):

                self.assertIn(
                    expected_message,
                    summary,
                )

        # Резервната константа не е необходима,
        # когато празните колони се премахват.
        self.assertNotIn(
            "Резервна категориална константа:",
            summary,
        )

        self.settings.drop_empty_checkbox.setChecked(
            False
        )

        updated_summary = (
            self.page._preprocessing_summary_label.text()
        )

        self.assertIn(
            "Резервна категориална константа: Неизвестен",
            updated_summary,
        )

        self.assertIn(
            "Премахване на изцяло празни колони: Не",
            updated_summary,
        )

        # Проверяваме промяна на метода за мащабиране.
        self._select_combo_value(
            self.settings.scaling_combo,
            "minmax",
        )

        self.assertIn(
            SCALING_METHODS["minmax"],
            self.page._preprocessing_summary_label.text(),
        )

        self.assertNotIn(
            SCALING_METHODS["robust"],
            self.page._preprocessing_summary_label.text(),
        )

    def test_model_summary_updates(self):
        """Проверява динамичното обобщение на ML модела."""

        self._select_combo_value(
            self.page._task_combo,
            "regression",
        )

        self._select_combo_value(
            self.page._algorithm_combo,
            "random_forest",
        )

        self.page._test_size_spin.setValue(
            30
        )

        self.page._random_state_spin.setValue(
            123
        )

        summary = (
            self.page._model_summary_label.text()
        )

        expected_messages = (
            "Тип задача: Регресия",
            "Алгоритъм: Random Forest Regressor",
            "Тренировъчна извадка: 70%",
            "Тестова извадка: 30%",
            "Начално състояние (Random State): 123",
        )

        for expected_message in expected_messages:

            with self.subTest(
                message=expected_message
            ):

                self.assertIn(
                    expected_message,
                    summary,
                )

        self.assertEqual(
            self.page.selected_task_type,
            "regression",
        )

        self.assertEqual(
            self.page.selected_algorithm,
            "random_forest",
        )

        self.assertAlmostEqual(
            self.page.test_size,
            0.3,
        )

        self.assertEqual(
            self.page.random_state,
            123,
        )

        self.assertTrue(
            self.page.is_configuration_valid
        )

    def test_invalid_numeric_configuration(self):
        """Проверява невалидните числови константи."""

        self._select_combo_value(
            self.settings.numeric_imputation_combo,
            "constant",
        )

        for invalid_value in (
            "",
            "abc",
            "NaN",
            "Infinity",
            "1e309",
        ):

            with self.subTest(
                value=invalid_value
            ):

                self.settings.numeric_fill_input.setText(
                    invalid_value
                )

                self.assertIsNone(
                    self.page.preprocessing_configuration
                )

                self.assertFalse(
                    self.page.is_preprocessing_valid
                )

                self.assertFalse(
                    self.page.is_configuration_valid
                )

                self.assertIn(
                    "Невалидни настройки",
                    self.page._preprocessing_summary_label.text(),
                )

                self.assertIn(
                    "Конфигурацията не е готова",
                    self.page._configuration_status_label.text(),
                )

        # Поправяме числовата константа.
        self.settings.numeric_fill_input.setText(
            "10,5"
        )

        self.assertTrue(
            self.page.is_preprocessing_valid
        )

        self.assertTrue(
            self.page.is_configuration_valid
        )

        self.assertAlmostEqual(
            self.page.preprocessing_configuration
            .numeric_fill_value,
            10.5,
        )

        self.assertIn(
            "Числова константа: 10.5",
            self.page._preprocessing_summary_label.text(),
        )

    def test_invalid_categorical_configuration(self):
        """Проверява невалидните категориални константи."""

        for strategy in (
            "constant",
            "most_frequent",
        ):

            self._select_combo_value(
                self.settings.categorical_imputation_combo,
                strategy,
            )

            for invalid_value in (
                "",
                " ",
                "   ",
            ):

                with self.subTest(
                    strategy=strategy,
                    value=invalid_value,
                ):

                    self.settings.categorical_fill_input.setText(
                        invalid_value
                    )

                    self.assertIsNone(
                        self.page.preprocessing_configuration
                    )

                    self.assertFalse(
                        self.page.is_preprocessing_valid
                    )

                    self.assertFalse(
                        self.page.is_configuration_valid
                    )

                    self.assertIn(
                        "Конфигурацията не е готова",
                        self.page._configuration_status_label.text(),
                    )

            # Възстановяваме валидна стойност.
            self.settings.categorical_fill_input.setText(
                "Неизвестен"
            )

            self.assertTrue(
                self.page.is_preprocessing_valid
            )

            self.assertTrue(
                self.page.is_configuration_valid
            )

            self.assertEqual(
                self.page.preprocessing_configuration
                .categorical_fill_value,
                "Неизвестен",
            )

    def test_missing_experiment_configuration(self):
        """Проверява липсващ експеримент при валиден Preprocessing."""

        self.experiments_page._configuration = None

        self.page.refresh_experiment_configuration()

        self.assertTrue(
            self.page.is_preprocessing_valid
        )

        self.assertFalse(
            self.page.is_configuration_valid
        )

        self.assertEqual(
            self.page.preprocessing_configuration,
            PreprocessingConfiguration(),
        )

        self.assertIn(
            "Няма завършена конфигурация",
            self.page._configuration_label.text(),
        )

        self.assertIn(
            "Конфигурацията не е готова",
            self.page._configuration_status_label.text(),
        )

        # Възстановяваме експеримента.
        self.experiments_page._configuration = (
            self.experiment_configuration
        )

        self.page.refresh_experiment_configuration()

        self.assertTrue(
            self.page.is_configuration_valid
        )

        self.assertIn(
            "preprocessing_integration.csv",
            self.page._configuration_label.text(),
        )

        self.assertIn(
            "Конфигурацията е валидна",
            self.page._configuration_status_label.text(),
        )

    def test_settings_are_preserved_during_navigation(self):
        """Проверява запазването на настройките при навигация."""

        expected = (
            self._create_custom_configuration()
        )

        self.stack.resize(
            1000,
            700,
        )

        self.stack.show()

        self.stack.setCurrentWidget(
            self.page
        )

        self.app.processEvents()

        self.settings.set_configuration(
            expected
        )

        self._select_combo_value(
            self.page._task_combo,
            "regression",
        )

        self._select_combo_value(
            self.page._algorithm_combo,
            "random_forest",
        )

        self.page._test_size_spin.setValue(
            25
        )

        self.page._random_state_spin.setValue(
            123
        )

        # Преминаваме към друга страница.
        self.stack.setCurrentWidget(
            self.other_page
        )

        self.app.processEvents()

        # Връщаме се в ModelsPage.
        self.stack.setCurrentWidget(
            self.page
        )

        self.app.processEvents()

        self.assertIs(
            self.stack.currentWidget(),
            self.page,
        )

        self.assertEqual(
            self.page.preprocessing_configuration,
            expected,
        )

        self.assertEqual(
            self.page.selected_task_type,
            "regression",
        )

        self.assertEqual(
            self.page.selected_algorithm,
            "random_forest",
        )

        self.assertAlmostEqual(
            self.page.test_size,
            0.25,
        )

        self.assertEqual(
            self.page.random_state,
            123,
        )

        self.assertTrue(
            self.page.is_configuration_valid
        )

    def test_reset_preserves_model_settings(self):
        """Проверява reset само на Preprocessing настройките."""

        self.settings.set_configuration(
            self._create_custom_configuration()
        )

        self._select_combo_value(
            self.page._task_combo,
            "regression",
        )

        self._select_combo_value(
            self.page._algorithm_combo,
            "decision_tree",
        )

        self.page._test_size_spin.setValue(
            35
        )

        self.page._random_state_spin.setValue(
            123
        )

        # Създаваме невалидна Preprocessing конфигурация.
        self.settings.numeric_fill_input.setText(
            "invalid"
        )

        self.assertFalse(
            self.page.is_configuration_valid
        )

        # Натискаме бутона за възстановяване.
        self.settings.reset_button.click()

        self.assertEqual(
            self.page.preprocessing_configuration,
            PreprocessingConfiguration(),
        )

        self.assertTrue(
            self.page.is_preprocessing_valid
        )

        self.assertTrue(
            self.page.is_configuration_valid
        )

        # ML настройките не трябва да бъдат променени.
        self.assertEqual(
            self.page.selected_task_type,
            "regression",
        )

        self.assertEqual(
            self.page.selected_algorithm,
            "decision_tree",
        )

        self.assertAlmostEqual(
            self.page.test_size,
            0.35,
        )

        self.assertEqual(
            self.page.random_state,
            123,
        )

        self.assertIn(
            SCALING_METHODS["standard"],
            self.page._preprocessing_summary_label.text(),
        )

        self.assertIn(
            ENCODING_METHODS["onehot"],
            self.page._preprocessing_summary_label.text(),
        )

        self.assertIn(
            "Премахване на изцяло празни колони: Не",
            self.page._preprocessing_summary_label.text(),
        )

        self.assertIn(
            "Премахване на постоянни характеристики: Не",
            self.page._preprocessing_summary_label.text(),
        )

    def test_gui_configuration_is_used_for_training(self):
        """Проверява предаването на GUI настройките към backend-а."""

        expected = (
            self._create_custom_configuration()
        )

        self.settings.set_configuration(
            expected
        )

        self._select_combo_value(
            self.page._task_combo,
            "regression",
        )

        self._select_combo_value(
            self.page._algorithm_combo,
            "linear_regression",
        )

        self.page._test_size_spin.setValue(
            25
        )

        self.page._random_state_spin.setValue(
            123
        )

        self.assertTrue(
            self.page.is_configuration_valid
        )

        configuration = (
            self.experiments_page.experiment_configuration
        )

        # Използваме настройките директно от ModelsPage.
        experiment = train_experiment_model(
            dataset=configuration.dataset,
            target_column=configuration.target_column,
            feature_columns=configuration.feature_columns,
            task_type=self.page.selected_task_type,
            algorithm=self.page.selected_algorithm,
            test_size=self.page.test_size,
            random_state=self.page.random_state,
            preprocessing_configuration=(
                self.page.preprocessing_configuration
            ),
        )

        self.assertEqual(
            experiment.preprocessing_configuration,
            expected,
        )

        self.assertEqual(
            experiment.task_type,
            "regression",
        )

        self.assertEqual(
            experiment.algorithm,
            "linear_regression",
        )

        self.assertEqual(
            experiment.random_state,
            123,
        )

        self.assertEqual(
            experiment.training_row_count,
            15,
        )

        self.assertEqual(
            experiment.test_row_count,
            5,
        )

        # Остават числовата характеристика Value
        # и честотно кодираната характеристика City.
        self.assertEqual(
            experiment.feature_count,
            2,
        )

        # Създаваме подробен Preprocessing отчет.
        report = build_preprocessing_report(
            experiment
        )

        self.assertEqual(
            report.configuration,
            expected,
        )

        self.assertEqual(
            report.original_row_count,
            20,
        )

        self.assertEqual(
            report.input_feature_count,
            4,
        )

        self.assertEqual(
            report.generated_feature_count,
            3,
        )

        self.assertEqual(
            report.final_feature_count,
            2,
        )

        self.assertEqual(
            report.removed_empty_columns,
            ("Notes",),
        )

        self.assertEqual(
            report.removed_constant_features,
            ("Country",),
        )

        self.assertEqual(
            report.final_feature_names,
            ("Value", "City"),
        )

        numeric_column = report.get_column_report(
            "Value"
        )

        self.assertEqual(
            numeric_column.imputation_strategy,
            "constant",
        )

        self.assertEqual(
            numeric_column.scaling_method,
            "robust",
        )

        categorical_column = report.get_column_report(
            "City"
        )

        self.assertEqual(
            categorical_column.encoding_method,
            "frequency",
        )

        self.assertEqual(
            categorical_column.imputation_strategy,
            "most_frequent",
        )

        # Проверяваме дали обученият модел
        # може да генерира валидни прогнози.
        predictions = experiment.estimator.predict(
            experiment.transformed_data.X_test
        )

        self.assertEqual(
            len(predictions),
            5,
        )

        self.assertTrue(
            np.isfinite(predictions).all()
        )

        np.testing.assert_allclose(
            predictions,
            experiment.transformed_data.y_test,
            atol=1e-8,
        )

        # Обучението не трябва да променя GUI настройките.
        self.assertEqual(
            self.page.preprocessing_configuration,
            expected,
        )


if __name__ == "__main__":
    unittest.main()