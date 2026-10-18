
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openpyxl import Workbook  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ml_experiment_lab.datasets import (  # noqa: E402
    DatasetLoadError,
    get_excel_sheet_names,
    load_dataset,
)
from ml_experiment_lab.ui.pages.datasets_page import (  # noqa: E402
    DatasetsPage,
)


class ExcelImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def create_workbook(self, filename: str) -> Path:
        """Create an Excel workbook with two sample worksheets."""
        path = self.temp_path / filename
        workbook = Workbook()

        sales = workbook.active
        sales.title = "Продажби"
        sales.append(["Продукт", "Количество", "Цена"])
        sales.append(["Книга", 3, 12.50])
        sales.append(["Тетрадка", 5, 2.40])
        sales.append(["Молив", 10, None])

        expenses = workbook.create_sheet("Разходи")
        expenses.append(["Категория", "Сума"])
        expenses.append(["Транспорт", 25])
        expenses.append(["Храна", 40])

        workbook.save(path)
        workbook.close()

        return path

    def test_get_excel_sheet_names(self) -> None:
        path = self.create_workbook("sample.xlsx")

        sheet_names = get_excel_sheet_names(path)

        self.assertEqual(
            sheet_names,
            ("Продажби", "Разходи"),
        )

    def test_load_selected_worksheet(self) -> None:
        path = self.create_workbook("sample.xlsx")

        dataset = load_dataset(
            path,
            sheet_name="Продажби",
        )

        self.assertEqual(dataset.path, path.resolve())
        self.assertEqual(dataset.sheet_name, "Продажби")
        self.assertEqual(
            dataset.columns,
            ("Продукт", "Количество", "Цена"),
        )
        self.assertEqual(dataset.row_count, 3)
        self.assertEqual(dataset.column_count, 3)
        self.assertEqual(
            dataset.rows[0],
            ("Книга", "3", "12.5"),
        )
        self.assertEqual(
            dataset.rows[2],
            ("Молив", "10", ""),
        )

    def test_missing_worksheet_raises_error(self) -> None:
        path = self.create_workbook("sample.xlsx")

        with self.assertRaisesRegex(
            DatasetLoadError,
            "Worksheet not found",
        ):
            load_dataset(
                path,
                sheet_name="Несъществуващ лист",
            )

    def test_page_can_load_two_sheets_from_same_file(self) -> None:
        path = self.create_workbook("sample.xlsx")
        page = DatasetsPage()

        try:
            sales_loaded = page.add_dataset(
                path,
                sheet_name="Продажби",
            )
            expenses_loaded = page.add_dataset(
                path,
                sheet_name="Разходи",
            )

            self.assertTrue(sales_loaded)
            self.assertTrue(expenses_loaded)
            self.assertEqual(page.dataset_count, 2)
            self.assertEqual(
                page.current_dataset.sheet_name,
                "Разходи",
            )
            self.assertEqual(
                page.current_dataset.columns,
                ("Категория", "Сума"),
            )
        finally:
            page.close()
            page.deleteLater()
            self.app.processEvents()

    def test_reloading_same_sheet_does_not_duplicate_it(self) -> None:
        path = self.create_workbook("sample.xlsx")
        page = DatasetsPage()

        try:
            self.assertTrue(
                page.add_dataset(path, sheet_name="Продажби")
            )
            self.assertTrue(
                page.add_dataset(path, sheet_name="Продажби")
            )

            self.assertEqual(page.dataset_count, 1)
        finally:
            page.close()
            page.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
