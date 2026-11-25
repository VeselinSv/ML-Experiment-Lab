"""Автоматизирани тестове за графичния Preprocessing отчет."""

import os
import unittest

from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt

from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
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
    COLUMN_TYPE_NAMES,
    build_preprocessing_report,
)

from ml_experiment_lab.training import (
    train_experiment_model,
)

from ml_experiment_lab.ui.widgets.preprocessing_report_widget import (
    SCALING_PARAMETER_NAMES,
    PreprocessingReportWidget,
)


class PreprocessingReportWidgetTests(unittest.TestCase):
    """Проверява визуализирането на реални Preprocessing отчети."""

    @classmethod
    def setUpClass(cls):
        """Създава QApplication и обучава тестови модели."""

        cls.app = QApplication.instance()

        if cls.app is None:
            cls.app = QApplication([])

        cls.dataset = LoadedDataset(
            path=Path("report_widget_test.csv"),
            columns=(
                "Value",
                "City",
                "Country",
                "Registration",
                "Notes",
                "Target",
            ),
            rows=tuple(
                (
                    "" if index % 7 == 0 else str(index * 10),
                    "Варна" if index <= 16 else "София",
                    "България",
                    f"2024-01-{index:02d} 10:30:00",
                    "",
                    "" if index == 24 else str(index * 100),
                )
                for index in range(1, 25)
            ),
        )

        cls.reports = {}

        # Създаваме реален отчет за всеки метод на кодиране.
        for encoding_method in (
            "onehot",
            "ordinal",
            "frequency",
        ):

            configuration = PreprocessingConfiguration(
                numeric_imputation_strategy="constant",
                categorical_imputation_strategy="most_frequent",
                numeric_fill_value=-7.5,
                categorical_fill_value="Неизвестен",
                drop_empty_columns=True,
                drop_constant_features=True,
                scaling_method="robust",
                encoding_method=encoding_method,
            )

            experiment = train_experiment_model(
                dataset=cls.dataset,
                target_column="Target",
                feature_columns=(
                    "Value",
                    "City",
                    "Country",
                    "Registration",
                    "Notes",
                ),
                task_type="regression",
                algorithm="linear_regression",
                test_size=0.2,
                random_state=42,
                preprocessing_configuration=configuration,
            )

            cls.reports[encoding_method] = (
                build_preprocessing_report(experiment)
            )

    def setUp(self):
        """Създава нов графичен компонент за всеки тест."""

        self.widget = PreprocessingReportWidget()

    def tearDown(self):
        """Освобождава графичния компонент."""

        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()

    @staticmethod
    def _table_rows(table):
        """Връща текстовото съдържание на таблица."""

        return [
            tuple(
                table.item(row, column).text()
                for column in range(table.columnCount())
            )
            for row in range(table.rowCount())
        ]

    @staticmethod
    def _format_number(value):
        """Форматира числова стойност за сравнение."""

        return format(float(value), ".12g")

    def _load_report(self, encoding_method="frequency"):
        """Зарежда избран тестов отчет."""

        report = self.reports[encoding_method]

        self.widget.set_report(report)

        return report

    def _select_column(self, column_name):
        """Избира входна колона в раздела Параметри."""

        index = self.widget._column_combo.findData(
            column_name
        )

        self.assertGreaterEqual(
            index,
            0,
            f"Липсва колона: {column_name}",
        )

        self.widget._column_combo.setCurrentIndex(
            index
        )

    def _assert_tables_are_empty(self):
        """Проверява дали всички таблици са празни."""

        tables = (
            self.widget._summary_table,
            self.widget._columns_table,
            self.widget._parameters_table,
            self.widget._features_table,
        )

        for table in tables:
            self.assertEqual(table.rowCount(), 0)

    def test_initial_empty_state(self):
        """Проверява началното състояние без отчет."""

        self.assertIsNone(self.widget.report)

        self.assertFalse(self.widget.has_report)

        self.assertIs(
            self.widget._stack.currentWidget(),
            self.widget._empty_page,
        )

        self.assertIn(
            "Все още няма отчет",
            self.widget._empty_label.text(),
        )

        self.assertEqual(
            self.widget._column_combo.count(),
            0,
        )

        self.assertFalse(
            self.widget._column_combo.isEnabled()
        )

        self._assert_tables_are_empty()

    def test_set_report(self):
        """Проверява зареждането на валиден отчет."""

        report = self._load_report()

        self.assertIs(self.widget.report, report)

        self.assertTrue(self.widget.has_report)

        self.assertIs(
            self.widget._stack.currentWidget(),
            self.widget._tabs,
        )

        self.assertEqual(
            self.widget._tabs.currentIndex(),
            0,
        )

        self.assertGreater(
            self.widget._summary_table.rowCount(),
            0,
        )

        self.assertEqual(
            self.widget._columns_table.rowCount(),
            report.input_feature_count,
        )

    def test_report_tabs(self):
        """Проверява наличността и имената на разделите."""

        expected_tabs = [
            "Обобщение",
            "Колони",
            "Параметри",
            "Характеристики",
            "Предупреждения",
        ]

        actual_tabs = [
            self.widget._tabs.tabText(index)
            for index in range(self.widget._tabs.count())
        ]

        self.assertEqual(
            actual_tabs,
            expected_tabs,
        )

    def test_summary_values(self):
        """Проверява основните показатели и настройките."""

        report = self._load_report()

        summary = dict(
            self._table_rows(
                self.widget._summary_table
            )
        )

        expected_values = {
            "Целева колона": "Target",
            "Оригинални наблюдения": "24",
            "Изключени наблюдения": "1",
            "Валидни наблюдения": "23",
            "Тренировъчни наблюдения": "18",
            "Тестови наблюдения": "5",
            "Заявен размер на тестовата извадка": "20.00%",
            "Действителен размер на тестовата извадка": (
                f"{5 / 23 * 100:.2f}%"
            ),
            "Random State": "42",
            "Оригинални входни колони": "5",
            "Числови колони": "1",
            "Категориални колони": "2",
            "Датни колони": "1",
            "Изцяло празни колони": "1",
            "Колони със смесени типове": "0",
            "Генерирани характеристики": "12",
            "Крайни характеристики": str(
                report.final_feature_count
            ),
            "Премахнати празни колони": "1",
            "Премахнати постоянни характеристики": str(
                report.removed_constant_count
            ),
            "Липсващи входни стойности": str(
                report.total_missing_count
            ),
            "Невалидни входни стойности": str(
                report.total_invalid_count
            ),
            "Непознати категории": str(
                report.total_unknown_count
            ),
            "Числово попълване": (
                NUMERIC_IMPUTATION_STRATEGIES["constant"]
            ),
            "Категориално попълване": (
                CATEGORICAL_IMPUTATION_STRATEGIES["most_frequent"]
            ),
            "Настроена числова константа": "-7.5",
            "Настроена категориална константа": "Неизвестен",
            "Мащабиране": SCALING_METHODS["robust"],
            "Категориално кодиране": ENCODING_METHODS["frequency"],
            "Премахване на празни колони": "Да",
            "Премахване на постоянни характеристики": "Да",
            "Диагностични предупреждения": str(
                len(report.warnings)
            ),
        }

        self.assertEqual(
            summary,
            expected_values,
        )

    def test_columns_table_values(self):
        """Проверява показателите за всяка входна колона."""

        report = self._load_report()

        actual_rows = self._table_rows(
            self.widget._columns_table
        )

        expected_rows = []

        for column in report.columns:

            expected_rows.append(
                (
                    column.column_name,
                    COLUMN_TYPE_NAMES[column.column_type],
                    "Да" if column.is_mixed else "Не",
                    "Да" if column.is_empty else "Не",
                    str(column.training_missing_count),
                    str(column.test_missing_count),
                    str(column.training_invalid_count),
                    str(column.test_invalid_count),
                    str(column.training_unknown_count),
                    str(column.test_unknown_count),
                    str(column.generated_feature_count),
                    str(column.retained_feature_count),
                    str(column.removed_constant_count),
                    "Да" if column.removed_as_empty else "Не",
                )
            )

        self.assertEqual(
            actual_rows,
            expected_rows,
        )

    def test_column_selector(self):
        """Проверява избора и реда на входните колони."""

        report = self._load_report()

        combo = self.widget._column_combo

        expected_columns = [
            column.column_name
            for column in report.columns
        ]

        actual_columns = [
            combo.itemData(index)
            for index in range(combo.count())
        ]

        self.assertEqual(
            actual_columns,
            expected_columns,
        )

        self.assertTrue(combo.isEnabled())

        for column_name in expected_columns:

            with self.subTest(column=column_name):

                self._select_column(column_name)

                self.assertEqual(
                    combo.currentData(),
                    column_name,
                )

                self.assertIn(
                    f"Колона: {column_name}",
                    self.widget._column_details_label.text(),
                )

    def test_numeric_parameters(self):
        """Проверява числовото попълване и RobustScaler."""

        report = self._load_report()

        self._select_column("Value")

        rows = self._table_rows(
            self.widget._parameters_table
        )

        self.assertIn(
            (
                "Попълване",
                "Value",
                "Приложена стратегия",
                NUMERIC_IMPUTATION_STRATEGIES["constant"],
            ),
            rows,
        )

        self.assertIn(
            (
                "Попълване",
                "Value",
                "Стойност за попълване",
                "-7.5",
            ),
            rows,
        )

        self.assertIn(
            (
                "Мащабиране",
                "Value",
                "Приложен метод",
                SCALING_METHODS["robust"],
            ),
            rows,
        )

        column = report.get_column_report("Value")

        self.assertTrue(column.features)

        feature = column.features[0]

        parameter_names = {
            name
            for name, _ in feature.scaling_parameters
        }

        self.assertEqual(
            parameter_names,
            {"center", "scale"},
        )

        for name, value in feature.scaling_parameters:

            with self.subTest(parameter=name):

                self.assertIn(
                    (
                        "Мащабиране",
                        feature.display_name,
                        SCALING_PARAMETER_NAMES[name],
                        self._format_number(value),
                    ),
                    rows,
                )

    def test_onehot_encoding_parameters(self):
        """Проверява научените категории при One-Hot Encoding."""

        report = self._load_report("onehot")

        self._select_column("City")

        rows = self._table_rows(
            self.widget._parameters_table
        )

        column = report.get_column_report("City")

        self.assertEqual(
            column.encoding_method,
            "onehot",
        )

        self.assertEqual(
            len(column.categories),
            2,
        )

        feature_names = {
            feature.technical_name: feature.display_name
            for feature in column.features
        }

        for category in column.categories:

            with self.subTest(category=category.category):

                self.assertIn(
                    (
                        "Кодиране",
                        category.category,
                        "Бинарна характеристика",
                        feature_names[category.feature_name],
                    ),
                    rows,
                )

        self.assertIn(
            (
                "Кодиране",
                "Непознати категории",
                "Резервна стойност",
                "Нули във всички бинарни характеристики",
            ),
            rows,
        )

    def test_ordinal_encoding_parameters(self):
        """Проверява числовите кодове на категориите."""

        report = self._load_report("ordinal")

        self._select_column("City")

        rows = self._table_rows(
            self.widget._parameters_table
        )

        column = report.get_column_report("City")

        self.assertEqual(
            len(column.categories),
            2,
        )

        for category in column.categories:

            with self.subTest(category=category.category):

                self.assertIn(
                    (
                        "Кодиране",
                        category.category,
                        "Числов код",
                        self._format_number(
                            category.encoded_value
                        ),
                    ),
                    rows,
                )

        self.assertIn(
            (
                "Кодиране",
                "Непознати категории",
                "Резервна стойност",
                "-1",
            ),
            rows,
        )

    def test_frequency_encoding_parameters(self):
        """Проверява научените относителни честоти."""

        report = self._load_report("frequency")

        self._select_column("City")

        rows = self._table_rows(
            self.widget._parameters_table
        )

        column = report.get_column_report("City")

        self.assertEqual(
            len(column.categories),
            2,
        )

        self.assertAlmostEqual(
            sum(
                category.encoded_value
                for category in column.categories
            ),
            1.0,
        )

        for category in column.categories:

            with self.subTest(category=category.category):

                self.assertIn(
                    (
                        "Кодиране",
                        category.category,
                        "Относителна честота",
                        self._format_number(
                            category.encoded_value
                        ),
                    ),
                    rows,
                )

        self.assertIn(
            (
                "Попълване",
                "City",
                "Стойност за попълване",
                column.categorical_imputation_value,
            ),
            rows,
        )

        self.assertIn(
            (
                "Кодиране",
                "Непознати категории",
                "Резервна стойност",
                "0",
            ),
            rows,
        )

    def test_datetime_parameters(self):
        """Проверява деветте генерирани датни характеристики."""

        report = self._load_report()

        self._select_column("Registration")

        column = report.get_column_report(
            "Registration"
        )

        self.assertEqual(
            column.column_type,
            "datetime",
        )

        self.assertEqual(
            len(column.features),
            9,
        )

        rows = self._table_rows(
            self.widget._parameters_table
        )

        for feature in column.features:

            with self.subTest(
                feature=feature.display_name
            ):

                self.assertIn(
                    (
                        "Попълване",
                        feature.display_name,
                        "Стойност за попълване",
                        self._format_number(
                            feature.imputation_value
                        ),
                    ),
                    rows,
                )

                for name, value in feature.scaling_parameters:

                    self.assertIn(
                        (
                            "Мащабиране",
                            feature.display_name,
                            SCALING_PARAMETER_NAMES[name],
                            self._format_number(value),
                        ),
                        rows,
                    )

    def test_removed_empty_column(self):
        """Проверява визуализирането на празна колона."""

        report = self._load_report()

        # Първо зареждаме параметри на друга колона.
        self._select_column("Value")

        self.assertGreater(
            self.widget._parameters_table.rowCount(),
            0,
        )

        # След това избираме премахнатата колона.
        self._select_column("Notes")

        column = report.get_column_report("Notes")

        self.assertTrue(column.is_empty)

        self.assertTrue(column.removed_as_empty)

        self.assertEqual(
            self.widget._parameters_table.rowCount(),
            0,
        )

        self.assertIn(
            "изцяло празна в тренировъчните данни",
            self.widget._column_details_label.text(),
        )

        self.assertIn(
            (
                "Notes",
                "—",
                "Notes",
                "Премахната",
                "Изцяло празна входна колона в Training",
            ),
            self._table_rows(
                self.widget._features_table
            ),
        )

    def test_feature_tracking(self):
        """Проверява проследяването на всички характеристики."""

        report = self._load_report()

        actual_rows = self._table_rows(
            self.widget._features_table
        )

        expected_rows = []

        for feature in report.generated_features:

            expected_rows.append(
                (
                    feature.source_column,
                    feature.technical_name,
                    feature.display_name,
                    (
                        "Премахната"
                        if feature.removed_as_constant
                        else "Запазена"
                    ),
                    (
                        "Постоянна характеристика в Training"
                        if feature.removed_as_constant
                        else "—"
                    ),
                )
            )

        for column_name in report.removed_empty_columns:

            expected_rows.append(
                (
                    column_name,
                    "—",
                    column_name,
                    "Премахната",
                    "Изцяло празна входна колона в Training",
                )
            )

        self.assertEqual(
            actual_rows,
            expected_rows,
        )

        retained_count = sum(
            row[3] == "Запазена"
            for row in actual_rows
        )

        self.assertEqual(
            retained_count,
            report.final_feature_count,
        )

        self.assertEqual(
            len(actual_rows),
            (
                report.generated_feature_count
                + report.removed_empty_count
            ),
        )

        self.assertIn(
            "Country",
            report.removed_constant_features,
        )

        self.assertIn(
            (
                "Крайни характеристики: "
                f"{report.final_feature_count}"
            ),
            self.widget._features_summary_label.text(),
        )

    def test_warnings(self):
        """Проверява диагностичните предупреждения."""

        report = self._load_report()

        self.assertTrue(report.warnings)

        expected_text = "\n\n".join(
            f"{index}. {warning}"
            for index, warning in enumerate(
                report.warnings,
                start=1,
            )
        )

        self.assertEqual(
            self.widget._warnings_text.toPlainText(),
            expected_text,
        )

        self.assertEqual(
            self.widget._warnings_count_label.text(),
            f"Брой предупреждения: {len(report.warnings)}",
        )

        # Проверяваме и състоянието без предупреждения.
        report_without_warnings = replace(
            report,
            warnings=(),
        )

        self.widget.set_report(
            report_without_warnings
        )

        self.assertEqual(
            self.widget._warnings_count_label.text(),
            "Брой предупреждения: 0",
        )

        self.assertIn(
            "Няма диагностични предупреждения",
            self.widget._warnings_text.toPlainText(),
        )

    def test_read_only_tables(self):
        """Проверява защитата от редактиране."""

        self._load_report()

        self._select_column("Value")

        tables = (
            self.widget._summary_table,
            self.widget._columns_table,
            self.widget._parameters_table,
            self.widget._features_table,
        )

        for table in tables:

            self.assertEqual(
                table.editTriggers(),
                QAbstractItemView.EditTrigger.NoEditTriggers,
            )

            self.assertGreater(
                table.rowCount(),
                0,
            )

            for row in range(table.rowCount()):

                for column in range(table.columnCount()):

                    item = table.item(row, column)

                    self.assertIsNotNone(item)

                    self.assertFalse(
                        bool(
                            item.flags()
                            & Qt.ItemFlag.ItemIsEditable
                        )
                    )

        self.assertTrue(
            self.widget._warnings_text.isReadOnly()
        )

    def test_report_lifecycle(self):
        """Проверява подмяна, валидиране и изчистване на отчет."""

        first_report = self._load_report("onehot")

        self._select_column("City")

        self.widget._tabs.setCurrentIndex(2)

        first_parameters = self._table_rows(
            self.widget._parameters_table
        )

        # Зареждаме нов отчет.
        second_report = self._load_report("frequency")

        self.assertIs(
            self.widget.report,
            second_report,
        )

        self.assertIsNot(
            self.widget.report,
            first_report,
        )

        self.assertEqual(
            self.widget._tabs.currentIndex(),
            0,
        )

        self.assertEqual(
            self.widget._column_combo.currentData(),
            second_report.columns[0].column_name,
        )

        self._select_column("City")

        second_parameters = self._table_rows(
            self.widget._parameters_table
        )

        self.assertNotEqual(
            first_parameters,
            second_parameters,
        )

        self.assertFalse(
            any(
                row[2] == "Бинарна характеристика"
                for row in second_parameters
            )
        )

        # Невалиден обект не трябва да замества отчета.
        for invalid_report in (None, {}, "invalid", 123):

            with self.subTest(report=invalid_report):

                with self.assertRaises(TypeError):

                    self.widget.set_report(
                        invalid_report
                    )

                self.assertIs(
                    self.widget.report,
                    second_report,
                )

        # Изчистваме отчета.
        self.widget.clear_report()

        self.assertIsNone(self.widget.report)

        self.assertFalse(self.widget.has_report)

        self.assertIs(
            self.widget._stack.currentWidget(),
            self.widget._empty_page,
        )

        self.assertEqual(
            self.widget._column_combo.count(),
            0,
        )

        self.assertFalse(
            self.widget._column_combo.isEnabled()
        )

        self._assert_tables_are_empty()

        self.assertEqual(
            self.widget._column_details_label.text(),
            "",
        )

        self.assertEqual(
            self.widget._features_summary_label.text(),
            "",
        )

        self.assertEqual(
            self.widget._warnings_count_label.text(),
            "",
        )

        self.assertEqual(
            self.widget._warnings_text.toPlainText(),
            "",
        )

        # Проверяваме повторно зареждане след изчистване.
        self.widget.set_report(first_report)

        self.assertTrue(self.widget.has_report)

        self.assertIs(
            self.widget.report,
            first_report,
        )

        self.assertGreater(
            self.widget._summary_table.rowCount(),
            0,
        )


if __name__ == "__main__":
    unittest.main()