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

from ml_experiment_lab.ui.pages.datasets_page import DatasetsPage  # noqa: E402
from ml_experiment_lab.ui.pages.experiments_page import (  # noqa: E402
    ExperimentsPage,
)


class ExperimentsPageDatasetSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.datasets_page = DatasetsPage()
        self.experiments_page = ExperimentsPage(
            self.datasets_page
        )

    def tearDown(self) -> None:
        self.experiments_page.close()
        self.experiments_page.deleteLater()

        self.datasets_page.close()
        self.datasets_page.deleteLater()

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

    def test_selector_is_disabled_without_datasets(self) -> None:
        self.assertFalse(
            self.experiments_page._dataset_combo.isEnabled()
        )

        self.assertIsNone(
            self.experiments_page.selected_dataset
        )

        self.assertIn(
            "Заредете CSV или Excel файл",
            self.experiments_page._status_label.text(),
        )

    def test_loaded_dataset_appears_and_is_selected(self) -> None:
        path = self.create_csv(
            "students.csv",
            "name,age\nAlice,17\nBob,18\n",
        )

        self.assertTrue(
            self.datasets_page.add_dataset(path)
        )

        self.assertTrue(
            self.experiments_page._dataset_combo.isEnabled()
        )

        self.assertEqual(
            self.experiments_page._dataset_combo.count(),
            2,
        )

        self.assertEqual(
            self.experiments_page._dataset_combo.currentText(),
            "students.csv",
        )

        self.assertEqual(
            self.experiments_page.selected_dataset.path,
            path.resolve(),
        )

        details = self.experiments_page._dataset_details_label.text()

        self.assertIn("Редове: 2", details)
        self.assertIn("Колони: 2", details)

    def test_selector_updates_when_dataset_is_added(self) -> None:
        first_path = self.create_csv(
            "first.csv",
            "value\n1\n2\n",
        )

        second_path = self.create_csv(
            "second.csv",
            "value\n3\n4\n",
        )

        self.assertTrue(
            self.datasets_page.add_dataset(first_path)
        )

        first_key = (
            self.experiments_page._dataset_combo.currentData()
        )

        self.assertTrue(
            self.datasets_page.add_dataset(second_path)
        )

        self.assertEqual(
            self.experiments_page._dataset_combo.count(),
            3,
        )

        # Добавянето на нов набор не променя текущия избор.
        self.assertEqual(
            self.experiments_page._dataset_combo.currentData(),
            first_key,
        )

        self.assertEqual(
            self.experiments_page.selected_dataset.path,
            first_path.resolve(),
        )

    def test_selector_updates_when_selected_dataset_is_removed(self) -> None:
        first_path = self.create_csv(
            "first.csv",
            "value\n1\n2\n",
        )

        second_path = self.create_csv(
            "second.csv",
            "value\n3\n4\n",
        )

        self.assertTrue(
            self.datasets_page.add_dataset(first_path)
        )

        self.assertTrue(
            self.datasets_page.add_dataset(second_path)
        )

        second_key = next(
            key
            for key, dataset in self.datasets_page.loaded_datasets.items()
            if dataset.path == second_path.resolve()
        )

        second_index = (
            self.experiments_page._dataset_combo.findData(second_key)
        )

        self.experiments_page._dataset_combo.setCurrentIndex(
            second_index
        )

        self.datasets_page._select_dataset_by_key(second_key)

        self.assertTrue(
            self.datasets_page.remove_current_dataset(
                confirm=False
            )
        )

        self.assertEqual(
            self.experiments_page._dataset_combo.count(),
            2,
        )

        self.assertNotEqual(
            self.experiments_page._dataset_combo.currentData(),
            second_key,
        )

        self.assertIsNotNone(
            self.experiments_page.selected_dataset
        )

        self.assertEqual(
            self.experiments_page.selected_dataset.path,
            first_path.resolve(),
        )

    def test_excel_sheet_name_is_displayed(self) -> None:
        path = self.temp_path / "sales.xlsx"

        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Продажби"

        worksheet.append(["product", "price"])
        worksheet.append(["Book", "15"])

        workbook.save(path)
        workbook.close()

        self.assertTrue(
            self.datasets_page.add_dataset(
                path,
                sheet_name="Продажби",
            )
        )

        self.assertIn(
            "Продажби",
            self.experiments_page._dataset_combo.currentText(),
        )

        self.assertIn(
            "Работен лист: Продажби",
            self.experiments_page._dataset_details_label.text(),
        )


if __name__ == "__main__":
    unittest.main()