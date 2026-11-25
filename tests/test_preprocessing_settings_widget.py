"""Тестове за графичните настройки на предварителната обработка."""

import os
import unittest

os.environ.setdefault(
    "QT_QPA_PLATFORM",
    "offscreen",
)

from PySide6.QtWidgets import QApplication

from ml_experiment_lab.preprocessing import (
    CATEGORICAL_IMPUTATION_STRATEGIES,
    ENCODING_METHODS,
    NUMERIC_IMPUTATION_STRATEGIES,
    SCALING_METHODS,
    PreprocessingConfiguration,
    PreprocessingError,
)

from ml_experiment_lab.ui.widgets.preprocessing_settings_widget import (
    CATEGORICAL_IMPUTATION_DESCRIPTIONS,
    ENCODING_DESCRIPTIONS,
    NUMERIC_IMPUTATION_DESCRIPTIONS,
    SCALING_DESCRIPTIONS,
    PreprocessingSettingsWidget,
)


class PreprocessingSettingsWidgetTests(unittest.TestCase):
    """Проверява графичния компонент за Preprocessing настройки."""

    @classmethod
    def setUpClass(cls):
        """Създава QApplication за всички тестове."""

        cls.app = QApplication.instance()

        if cls.app is None:
            cls.app = QApplication([])

    def setUp(self):
        """Създава нов графичен компонент преди всеки тест."""

        self.widget = PreprocessingSettingsWidget()

    def tearDown(self):
        """Освобождава графичния компонент след всеки тест."""

        self.widget.close()

        self.widget.deleteLater()

        self.app.processEvents()

    def _select_combo_value(
        self,
        combo_box,
        value: str,
    ) -> None:
        """Избира елемент чрез неговия вътрешен идентификатор."""

        index = combo_box.findData(value)

        self.assertGreaterEqual(
            index,
            0,
            f"Липсва настройка: {value}",
        )

        combo_box.setCurrentIndex(index)

    def _assert_combo_options(
        self,
        combo_box,
        expected_options: dict[str, str],
    ) -> None:
        """Проверява съдържанието на падащо меню."""

        actual_options = {
            combo_box.itemData(index):
            combo_box.itemText(index)
            for index in range(combo_box.count())
        }

        self.assertEqual(
            combo_box.count(),
            len(expected_options),
        )

        self.assertEqual(
            actual_options,
            expected_options,
        )

    def test_default_configuration(self):
        """Проверява първоначалната конфигурация."""

        expected = PreprocessingConfiguration()

        actual = self.widget.get_configuration()

        self.assertEqual(
            actual,
            expected,
        )

        self.assertEqual(
            self.widget.preprocessing_configuration,
            expected,
        )

        self.assertTrue(
            self.widget.is_valid
        )

        self.assertEqual(
            self.widget.numeric_imputation_combo.currentData(),
            "median",
        )

        self.assertEqual(
            self.widget.categorical_imputation_combo.currentData(),
            "constant",
        )

        self.assertEqual(
            self.widget.scaling_combo.currentData(),
            "standard",
        )

        self.assertEqual(
            self.widget.encoding_combo.currentData(),
            "onehot",
        )

        self.assertFalse(
            self.widget.drop_empty_checkbox.isChecked()
        )

        self.assertFalse(
            self.widget.drop_constant_checkbox.isChecked()
        )

        self.assertTrue(
            self.widget.validation_label.isHidden()
        )

        self.assertEqual(
            self.widget.validation_label.text(),
            "",
        )

    def test_imputation_options(self):
        """Проверява стратегиите за попълване."""

        self._assert_combo_options(
            self.widget.numeric_imputation_combo,
            NUMERIC_IMPUTATION_STRATEGIES,
        )

        self._assert_combo_options(
            self.widget.categorical_imputation_combo,
            CATEGORICAL_IMPUTATION_STRATEGIES,
        )

        self.assertEqual(
            self.widget.numeric_imputation_combo.count(),
            4,
        )

        self.assertEqual(
            self.widget.categorical_imputation_combo.count(),
            2,
        )

    def test_transformation_options(self):
        """Проверява методите за Scaling и Encoding."""

        self._assert_combo_options(
            self.widget.scaling_combo,
            SCALING_METHODS,
        )

        self._assert_combo_options(
            self.widget.encoding_combo,
            ENCODING_METHODS,
        )

        self.assertEqual(
            self.widget.scaling_combo.count(),
            5,
        )

        self.assertEqual(
            self.widget.encoding_combo.count(),
            3,
        )

        for method in SCALING_METHODS:

            with self.subTest(scaling=method):

                self._select_combo_value(
                    self.widget.scaling_combo,
                    method,
                )

                configuration = (
                    self.widget.get_configuration()
                )

                self.assertEqual(
                    configuration.scaling_method,
                    method,
                )

        for method in ENCODING_METHODS:

            with self.subTest(encoding=method):

                self._select_combo_value(
                    self.widget.encoding_combo,
                    method,
                )

                configuration = (
                    self.widget.get_configuration()
                )

                self.assertEqual(
                    configuration.encoding_method,
                    method,
                )

    def test_numeric_fill_enabled_only_for_constant(self):
        """Проверява активирането на числовата константа."""

        for method in NUMERIC_IMPUTATION_STRATEGIES:

            with self.subTest(method=method):

                self._select_combo_value(
                    self.widget.numeric_imputation_combo,
                    method,
                )

                expected_enabled = (
                    method == "constant"
                )

                self.assertEqual(
                    self.widget.numeric_fill_input.isEnabled(),
                    expected_enabled,
                )

                self.assertEqual(
                    self.widget.numeric_fill_label.isEnabled(),
                    expected_enabled,
                )

    def test_categorical_fill_is_always_enabled(self):
        """Проверява достъпността на резервната константа."""

        for method in CATEGORICAL_IMPUTATION_STRATEGIES:

            with self.subTest(method=method):

                self._select_combo_value(
                    self.widget.categorical_imputation_combo,
                    method,
                )

                self.assertTrue(
                    self.widget.categorical_fill_input.isEnabled()
                )

                self.widget.categorical_fill_input.setText(
                    "Неизвестен"
                )

                configuration = (
                    self.widget.get_configuration()
                )

                self.assertEqual(
                    configuration.categorical_imputation_strategy,
                    method,
                )

                self.assertEqual(
                    configuration.categorical_fill_value,
                    "Неизвестен",
                )

    def test_valid_numeric_fill_values(self):
        """Проверява поддържаните числови формати."""

        self._select_combo_value(
            self.widget.numeric_imputation_combo,
            "constant",
        )

        valid_values = (
            ("0", 0.0),
            ("10", 10.0),
            ("-5", -5.0),
            ("+15", 15.0),
            ("10.5", 10.5),
            ("10,5", 10.5),
            (".5", 0.5),
            (",5", 0.5),
            ("1e3", 1000.0),
            ("1e-3", 0.001),
            ("1,5e2", 150.0),
            (" 25,5 ", 25.5),
        )

        for text, expected in valid_values:

            with self.subTest(value=text):

                self.widget.numeric_fill_input.setText(
                    text
                )

                configuration = (
                    self.widget.get_configuration()
                )

                self.assertAlmostEqual(
                    configuration.numeric_fill_value,
                    expected,
                )

                self.assertEqual(
                    configuration.numeric_imputation_strategy,
                    "constant",
                )

                self.assertTrue(
                    self.widget.is_valid
                )

                self.assertTrue(
                    self.widget.validation_label.isHidden()
                )

    def test_invalid_numeric_fill_values(self):
        """Проверява отхвърлянето на невалидни числа."""

        self._select_combo_value(
            self.widget.numeric_imputation_combo,
            "constant",
        )

        invalid_values = (
            "",
            "   ",
            "abc",
            "NaN",
            "Infinity",
            "-Infinity",
            "inf",
            "1e309",
            "10,5,2",
            "10.5.2",
            "1_000",
            "12 34",
        )

        for value in invalid_values:

            with self.subTest(value=value):

                self.widget.numeric_fill_input.setText(
                    value
                )

                self.assertFalse(
                    self.widget.is_valid
                )

                self.assertIsNone(
                    self.widget.preprocessing_configuration
                )

                self.assertFalse(
                    self.widget.validation_label.isHidden()
                )

                self.assertIn(
                    "числова константа",
                    self.widget.validation_label.text(),
                )

                with self.assertRaises(PreprocessingError):

                    self.widget.get_configuration()

        # Проверяваме възстановяването след грешка.
        self.widget.numeric_fill_input.setText(
            "10,5"
        )

        self.assertTrue(
            self.widget.is_valid
        )

        self.assertTrue(
            self.widget.validation_label.isHidden()
        )

        self.assertEqual(
            self.widget.validation_label.text(),
            "",
        )

    def test_unused_invalid_numeric_fill_value(self):
        """Проверява игнорирането на неизползвана константа."""

        self._select_combo_value(
            self.widget.numeric_imputation_combo,
            "constant",
        )

        self.widget.numeric_fill_input.setText(
            "невалидна стойност"
        )

        self.assertFalse(
            self.widget.is_valid
        )

        for method in (
            "mean",
            "median",
            "most_frequent",
        ):

            with self.subTest(method=method):

                self._select_combo_value(
                    self.widget.numeric_imputation_combo,
                    method,
                )

                configuration = (
                    self.widget.get_configuration()
                )

                self.assertTrue(
                    self.widget.is_valid
                )

                self.assertFalse(
                    self.widget.numeric_fill_input.isEnabled()
                )

                self.assertEqual(
                    configuration.numeric_imputation_strategy,
                    method,
                )

                self.assertEqual(
                    configuration.numeric_fill_value,
                    PreprocessingConfiguration().numeric_fill_value,
                )

        # При повторен избор на constant невалидната
        # стойност отново трябва да бъде отчетена.
        self._select_combo_value(
            self.widget.numeric_imputation_combo,
            "constant",
        )

        self.assertFalse(
            self.widget.is_valid
        )

    def test_invalid_categorical_fill_values(self):
        """Проверява празните категориални константи."""

        for method in CATEGORICAL_IMPUTATION_STRATEGIES:

            self._select_combo_value(
                self.widget.categorical_imputation_combo,
                method,
            )

            for value in (
                "",
                " ",
                "     ",
            ):

                with self.subTest(
                    method=method,
                    value=value,
                ):

                    self.widget.categorical_fill_input.setText(
                        value
                    )

                    self.assertFalse(
                        self.widget.is_valid
                    )

                    self.assertIsNone(
                        self.widget.preprocessing_configuration
                    )

                    self.assertFalse(
                        self.widget.validation_label.isHidden()
                    )

                    with self.assertRaises(
                        PreprocessingError
                    ):

                        self.widget.get_configuration()

            self.widget.categorical_fill_input.setText(
                "Липсваща стойност"
            )

            self.assertTrue(
                self.widget.is_valid
            )

            self.assertEqual(
                self.widget.get_configuration()
                .categorical_fill_value,
                "Липсваща стойност",
            )

    def test_feature_selection_configuration(self):
        """Проверява настройките за филтриране."""

        combinations = (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        )

        for drop_empty, drop_constant in combinations:

            with self.subTest(
                drop_empty=drop_empty,
                drop_constant=drop_constant,
            ):

                self.widget.drop_empty_checkbox.setChecked(
                    drop_empty
                )

                self.widget.drop_constant_checkbox.setChecked(
                    drop_constant
                )

                configuration = (
                    self.widget.get_configuration()
                )

                self.assertEqual(
                    configuration.drop_empty_columns,
                    drop_empty,
                )

                self.assertEqual(
                    configuration.drop_constant_features,
                    drop_constant,
                )

                self.assertTrue(
                    self.widget.is_valid
                )

    def test_dynamic_descriptions(self):
        """Проверява динамичното обновяване на поясненията."""

        controls = (
            (
                self.widget.numeric_imputation_combo,
                self.widget.numeric_imputation_description,
                NUMERIC_IMPUTATION_DESCRIPTIONS,
            ),
            (
                self.widget.categorical_imputation_combo,
                self.widget.categorical_imputation_description,
                CATEGORICAL_IMPUTATION_DESCRIPTIONS,
            ),
            (
                self.widget.scaling_combo,
                self.widget.scaling_description,
                SCALING_DESCRIPTIONS,
            ),
            (
                self.widget.encoding_combo,
                self.widget.encoding_description,
                ENCODING_DESCRIPTIONS,
            ),
        )

        for combo_box, label, descriptions in controls:

            for method, expected_description in (
                descriptions.items()
            ):

                with self.subTest(method=method):

                    self._select_combo_value(
                        combo_box,
                        method,
                    )

                    self.assertEqual(
                        label.text(),
                        expected_description,
                    )

                    self.assertTrue(
                        label.text().strip()
                    )

                    self.assertTrue(
                        label.wordWrap()
                    )

    def test_set_configuration(self):
        """Проверява зареждането на цялостна конфигурация."""

        expected = PreprocessingConfiguration(
            numeric_imputation_strategy="constant",
            categorical_imputation_strategy="most_frequent",
            numeric_fill_value=-12.5,
            categorical_fill_value="Неизвестен",
            drop_empty_columns=True,
            drop_constant_features=True,
            scaling_method="robust",
            encoding_method="frequency",
        )

        self.widget.set_configuration(
            expected
        )

        self.assertEqual(
            self.widget.get_configuration(),
            expected,
        )

        self.assertEqual(
            self.widget.preprocessing_configuration,
            expected,
        )

        self.assertTrue(
            self.widget.is_valid
        )

        self.assertEqual(
            self.widget.numeric_fill_input.text(),
            "-12.5",
        )

        self.assertEqual(
            self.widget.categorical_fill_input.text(),
            "Неизвестен",
        )

        self.assertTrue(
            self.widget.numeric_fill_input.isEnabled()
        )

        self.assertTrue(
            self.widget.drop_empty_checkbox.isChecked()
        )

        self.assertTrue(
            self.widget.drop_constant_checkbox.isChecked()
        )

        self.assertEqual(
            self.widget.scaling_description.text(),
            SCALING_DESCRIPTIONS["robust"],
        )

        self.assertEqual(
            self.widget.encoding_description.text(),
            ENCODING_DESCRIPTIONS["frequency"],
        )

    def test_set_invalid_configuration(self):
        """Проверява отхвърлянето на неподдържани обекти."""

        original_configuration = (
            self.widget.get_configuration()
        )

        invalid_configurations = (
            None,
            {},
            [],
            "standard",
            123,
            True,
        )

        for configuration in invalid_configurations:

            with self.subTest(
                configuration=configuration
            ):

                with self.assertRaises(
                    PreprocessingError
                ):

                    self.widget.set_configuration(
                        configuration
                    )

                # Невалидният обект не трябва да
                # променя текущите настройки.
                self.assertEqual(
                    self.widget.get_configuration(),
                    original_configuration,
                )

    def test_configuration_changed_signal(self):
        """Проверява сигнала при промяна на настройки."""

        received_configurations = []

        def on_configuration_changed():
            received_configurations.append(
                self.widget.preprocessing_configuration
            )

        self.widget.configuration_changed.connect(
            on_configuration_changed
        )

        self._select_combo_value(
            self.widget.scaling_combo,
            "minmax",
        )

        self.assertEqual(
            len(received_configurations),
            1,
        )

        self.assertEqual(
            received_configurations[-1].scaling_method,
            "minmax",
        )

        self._select_combo_value(
            self.widget.encoding_combo,
            "frequency",
        )

        self.assertEqual(
            len(received_configurations),
            2,
        )

        self.assertEqual(
            received_configurations[-1].encoding_method,
            "frequency",
        )

        self.widget.drop_empty_checkbox.setChecked(
            True
        )

        self.assertEqual(
            len(received_configurations),
            3,
        )

        self.assertTrue(
            received_configurations[-1].drop_empty_columns
        )

        # Сигнал се изпраща и при невалидна конфигурация.
        self.widget.categorical_fill_input.setText(
            ""
        )

        self.assertEqual(
            len(received_configurations),
            4,
        )

        self.assertIsNone(
            received_configurations[-1]
        )

        # След поправяне отново получаваме валиден обект.
        self.widget.categorical_fill_input.setText(
            "Неизвестен"
        )

        self.assertEqual(
            len(received_configurations),
            5,
        )

        self.assertIsInstance(
            received_configurations[-1],
            PreprocessingConfiguration,
        )

    def test_set_configuration_emits_single_signal(self):
        """Проверява един общ сигнал при зареждане."""

        received_configurations = []

        def on_configuration_changed():
            received_configurations.append(
                self.widget.preprocessing_configuration
            )

        self.widget.configuration_changed.connect(
            on_configuration_changed
        )

        expected = PreprocessingConfiguration(
            numeric_imputation_strategy="constant",
            categorical_imputation_strategy="most_frequent",
            numeric_fill_value=15.5,
            categorical_fill_value="Резервна стойност",
            drop_empty_columns=True,
            drop_constant_features=True,
            scaling_method="maxabs",
            encoding_method="ordinal",
        )

        self.widget.set_configuration(
            expected
        )

        self.assertEqual(
            len(received_configurations),
            1,
        )

        self.assertEqual(
            received_configurations[0],
            expected,
        )

        # Проверяваме дали сигналите на контролите
        # са възстановени след зареждането.
        self._select_combo_value(
            self.widget.scaling_combo,
            "none",
        )

        self.assertEqual(
            len(received_configurations),
            2,
        )

        self.assertEqual(
            received_configurations[-1].scaling_method,
            "none",
        )

    def test_reset_to_defaults(self):
        """Проверява бутона за възстановяване на настройките."""

        custom_configuration = PreprocessingConfiguration(
            numeric_imputation_strategy="constant",
            categorical_imputation_strategy="most_frequent",
            numeric_fill_value=100.0,
            categorical_fill_value="Резервна категория",
            drop_empty_columns=True,
            drop_constant_features=True,
            scaling_method="robust",
            encoding_method="frequency",
        )

        self.widget.set_configuration(
            custom_configuration
        )

        # Създаваме невалидна конфигурация,
        # за да проверим дали reset поправя и грешките.
        self.widget.numeric_fill_input.setText(
            "invalid"
        )

        self.widget.categorical_fill_input.setText(
            ""
        )

        self.assertFalse(
            self.widget.is_valid
        )

        received_configurations = []

        def on_configuration_changed():
            received_configurations.append(
                self.widget.preprocessing_configuration
            )

        self.widget.configuration_changed.connect(
            on_configuration_changed
        )

        # Симулираме натискането на бутона.
        self.widget.reset_button.click()

        expected = PreprocessingConfiguration()

        self.assertEqual(
            self.widget.get_configuration(),
            expected,
        )

        self.assertEqual(
            self.widget.preprocessing_configuration,
            expected,
        )

        self.assertTrue(
            self.widget.is_valid
        )

        self.assertFalse(
            self.widget.numeric_fill_input.isEnabled()
        )

        self.assertTrue(
            self.widget.categorical_fill_input.isEnabled()
        )

        self.assertFalse(
            self.widget.drop_empty_checkbox.isChecked()
        )

        self.assertFalse(
            self.widget.drop_constant_checkbox.isChecked()
        )

        self.assertTrue(
            self.widget.validation_label.isHidden()
        )

        self.assertEqual(
            self.widget.validation_label.text(),
            "",
        )

        self.assertEqual(
            self.widget.numeric_fill_input.text(),
            "0.0",
        )

        self.assertEqual(
            self.widget.categorical_fill_input.text(),
            expected.categorical_fill_value,
        )

        self.assertEqual(
            len(received_configurations),
            1,
        )

        self.assertEqual(
            received_configurations[0],
            expected,
        )


if __name__ == "__main__":
    unittest.main()