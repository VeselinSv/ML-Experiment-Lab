from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openpyxl import Workbook, load_workbook  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QPlainTextEdit,
)

from ml_experiment_lab.ui.pages.datasets_page import (  # noqa: E402
    DatasetsPage,
)


class DatasetsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.page = DatasetsPage()

    def tearDown(self) -> None:
        self.page.close()
        self.page.deleteLater()
        self.app.processEvents()
        self.temp_dir.cleanup()

    def create_csv(
        self,
        filename: str,
        contents: str,
    ) -> Path:
        path = self.temp_path / filename
        path.write_text(contents, encoding="utf-8")
        return path

    def create_xlsx(
        self,
        filename: str,
        sheet_name: str,
        rows: list[list[str]],
    ) -> Path:
        path = self.temp_path / filename
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = sheet_name

        for row in rows:
            worksheet.append(row)

        workbook.save(path)
        workbook.close()
        return path

    def test_import_populates_data_and_quality_tables(self) -> None:
        path = self.create_csv(
            "students.csv",
            "name,age,score\nAlice,17,9.5\nBob,18,\n",
        )

        self.assertTrue(self.page.add_dataset(path))
        self.assertEqual(self.page.dataset_count, 1)
        self.assertEqual(self.page._table_model.rowCount(), 2)
        self.assertEqual(self.page._profile_table.rowCount(), 3)
        self.assertGreater(self.page._issues_table.rowCount(), 0)
        self.assertIn(
            "Липсващи клетки",
            self.page._quality_summary_label.text(),
        )

    def test_selected_file_path_shows_full_path(self) -> None:
        path = self.create_csv(
            "a_very_long_dataset_filename_for_testing.csv",
            "value\n1\n2\n",
        )

        self.assertTrue(self.page.add_dataset(path))

        self.assertEqual(
            self.page._selected_file_path.toPlainText(),
            str(path.resolve()),
        )
        self.assertEqual(
            self.page._selected_file_path.lineWrapMode(),
            QPlainTextEdit.LineWrapMode.NoWrap,
        )
        self.assertEqual(
            self.page._selected_file_path.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )

    def test_recommendation_cells_are_scrollable(self) -> None:
        path = self.create_csv(
            "missing_values.csv",
            "name,score\nAlice,9.5\nBob,\n",
        )

        self.assertTrue(self.page.add_dataset(path))
        self.assertGreater(self.page._issues_table.rowCount(), 0)

        recommendation_found = False

        for row in range(self.page._issues_table.rowCount()):
            widget = self.page._issues_table.cellWidget(row, 3)

            if widget is not None:
                self.assertIsInstance(widget, QPlainTextEdit)
                self.assertTrue(widget.isReadOnly())
                self.assertEqual(
                    widget.verticalScrollBarPolicy(),
                    Qt.ScrollBarPolicy.ScrollBarAsNeeded,
                )
                recommendation_found = True
                break

        self.assertTrue(
            recommendation_found,
            "Очаква се поне една препоръка със скролер.",
        )

    def test_finding_details_are_scrollable(self) -> None:
        path = self.create_csv(
            "missing_values.csv",
            "name,score\nAlice,9.5\nBob,\n",
        )

        self.assertTrue(self.page.add_dataset(path))

        details_widget = self.page._issues_table.cellWidget(0, 1)

        self.assertIsInstance(details_widget, QPlainTextEdit)
        self.assertTrue(details_widget.isReadOnly())
        self.assertEqual(
            details_widget.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )

    def test_search_disabled_until_column_is_selected(self) -> None:
        path = self.create_csv(
            "cities.csv",
            "name,city\nAlice,Sofia\nBob,Plovdiv\n",
        )

        self.assertTrue(self.page.add_dataset(path))
        self.assertFalse(self.page._search_input.isEnabled())
        self.assertIsNone(self.page._selected_search_column)
        self.assertEqual(self.page._proxy_model.rowCount(), 2)

    def test_search_only_checks_selected_column(self) -> None:
        path = self.create_csv(
            "cities.csv",
            "name,city\nAlice,Sofia\nBob,Plovdiv\n",
        )

        self.assertTrue(self.page.add_dataset(path))

        self.page._on_table_header_clicked(1)
        self.page._search_input.setText("Alice")
        self.assertEqual(self.page._proxy_model.rowCount(), 0)

        self.page._on_table_header_clicked(0)
        self.assertEqual(self.page._search_input.text(), "")

        self.page._search_input.setText("alice")
        self.assertEqual(self.page._proxy_model.rowCount(), 1)

    def test_duplicate_file_is_not_added_twice(self) -> None:
        path = self.create_csv(
            "sample.csv",
            "value\n1\n2\n",
        )

        self.assertTrue(self.page.add_dataset(path))
        self.assertTrue(self.page.add_dataset(path))
        self.assertEqual(self.page.dataset_count, 1)

    def test_invalid_file_error_is_in_bulgarian(self) -> None:
        path = self.temp_path / "missing.csv"

        self.assertFalse(self.page.add_dataset(path))
        self.assertIn(
            "Неуспешно зареждане:",
            self.page._status_label.text(),
        )
        self.assertIn(
            "Файлът не съществува",
            self.page._status_label.text(),
        )

    def test_export_includes_rows_hidden_by_search(self) -> None:
        path = self.create_csv(
            "students.csv",
            "name,age\nAlice,17\nBob,18\n",
        )

        self.assertTrue(self.page.add_dataset(path))
        self.page._on_table_header_clicked(0)
        self.page._search_input.setText("Alice")
        self.assertEqual(self.page._proxy_model.rowCount(), 1)

        exported_path = self.temp_path / "exported.csv"

        self.assertTrue(
            self.page.export_current_dataset(exported_path)
        )

        exported = exported_path.read_text(encoding="utf-8-sig")

        self.assertIn("Alice,17", exported)
        self.assertIn("Bob,18", exported)

    def test_csv_dataset_can_be_exported_as_xlsx(self) -> None:
        source_path = self.create_csv(
            "students.csv",
            "name,age\nAlice,17\nBob,18\n",
        )
        self.assertTrue(self.page.add_dataset(source_path))

        exported_path = self.temp_path / "students_export.xlsx"
        self.assertTrue(
            self.page.export_current_dataset(exported_path)
        )
        self.assertTrue(exported_path.exists())

        workbook = load_workbook(
            exported_path,
            read_only=True,
            data_only=True,
        )
        try:
            worksheet = workbook["Данни"]
            exported_rows = list(
                worksheet.iter_rows(values_only=True)
            )
        finally:
            workbook.close()

        self.assertEqual(
            exported_rows,
            [
                ("name", "age"),
                ("Alice", "17"),
                ("Bob", "18"),
            ],
        )

    def test_xlsx_dataset_can_be_exported_as_csv(self) -> None:
        source_path = self.create_xlsx(
            "students.xlsx",
            "Ученици",
            [
                ["name", "age"],
                ["Alice", "17"],
                ["Bob", "18"],
            ],
        )
        self.assertTrue(
            self.page.add_dataset(
                source_path,
                sheet_name="Ученици",
            )
        )

        exported_path = self.temp_path / "students_export.csv"
        self.assertTrue(
            self.page.export_current_dataset(exported_path)
        )

        with exported_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            exported_rows = list(csv.reader(file))

        self.assertEqual(
            exported_rows,
            [
                ["name", "age"],
                ["Alice", "17"],
                ["Bob", "18"],
            ],
        )

    def test_export_dialog_uses_selected_excel_format(self) -> None:
        source_path = self.create_csv(
            "students.csv",
            "name,age\nAlice,17\nBob,18\n",
        )
        self.assertTrue(self.page.add_dataset(source_path))

        selected_path = self.temp_path / "chosen_export"
        with patch(
            "ml_experiment_lab.ui.pages.datasets_page.QFileDialog.getSaveFileName",
            return_value=(
                str(selected_path),
                "Excel файлове (*.xlsx)",
            ),
        ):
            self.page._choose_export_location()

        exported_path = self.temp_path / "chosen_export.xlsx"
        self.assertTrue(exported_path.exists())
        self.assertIn(
            "Excel файл",
            self.page._status_label.text(),
        )

    def test_export_dialog_uses_selected_csv_format_for_xlsx(self) -> None:
        source_path = self.create_xlsx(
            "students.xlsx",
            "Ученици",
            [
                ["name", "age"],
                ["Alice", "17"],
            ],
        )
        self.assertTrue(
            self.page.add_dataset(
                source_path,
                sheet_name="Ученици",
            )
        )

        selected_path = self.temp_path / "chosen_csv_export"
        with patch(
            "ml_experiment_lab.ui.pages.datasets_page.QFileDialog.getSaveFileName",
            return_value=(
                str(selected_path),
                "CSV файлове (*.csv)",
            ),
        ):
            self.page._choose_export_location()

        exported_path = self.temp_path / "chosen_csv_export.csv"
        self.assertTrue(exported_path.exists())
        self.assertIn(
            "CSV файл",
            self.page._status_label.text(),
        )

    def test_remove_does_not_delete_original_file(self) -> None:
        path = self.create_csv(
            "sample.csv",
            "value\n1\n2\n",
        )

        self.assertTrue(self.page.add_dataset(path))
        self.assertTrue(
            self.page.remove_current_dataset(confirm=False)
        )

        self.assertEqual(self.page.dataset_count, 0)
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()