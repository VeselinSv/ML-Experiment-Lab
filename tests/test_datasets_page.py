
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from ml_experiment_lab.ui.pages.datasets_page import (  # noqa: E402
    DatasetsPage,
)


class DatasetsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = (
            QApplication.instance()
            or QApplication([])
        )

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
        csv_path = self.temp_path / filename
        csv_path.write_text(contents, encoding="utf-8")
        return csv_path

    def test_import_populates_preview_and_profile(self) -> None:
        csv_path = self.create_csv(
            "students.csv",
            "name,age\nAlice,17\nBob,18\n",
        )

        result = self.page.add_dataset(csv_path)

        self.assertTrue(result)
        self.assertEqual(self.page.dataset_count, 1)
        self.assertIsNotNone(self.page.current_dataset)

        self.assertEqual(
            self.page._table_model.rowCount(),
            2,
        )
        self.assertEqual(
            self.page._table_model.columnCount(),
            2,
        )
        self.assertEqual(
            self.page._proxy_model.rowCount(),
            2,
        )
        self.assertEqual(
            self.page._profile_table.rowCount(),
            2,
        )

    def test_search_filters_all_columns(self) -> None:
        csv_path = self.create_csv(
            "students.csv",
            "name,city\nAlice,Sofia\nBob,Plovdiv\n",
        )

        self.assertTrue(self.page.add_dataset(csv_path))

        # Searching a value from the second column must work too.
        self.page._search_input.setText("plovdiv")

        self.assertEqual(
            self.page._proxy_model.rowCount(),
            1,
        )

        visible_index = self.page._proxy_model.index(0, 0)

        self.assertEqual(
            self.page._proxy_model.data(visible_index),
            "Bob",
        )

    def test_same_file_is_not_added_twice(self) -> None:
        csv_path = self.create_csv(
            "sample.csv",
            "value\n1\n2\n",
        )

        self.assertTrue(self.page.add_dataset(csv_path))
        self.assertTrue(self.page.add_dataset(csv_path))

        self.assertEqual(self.page.dataset_count, 1)

    def test_missing_file_is_not_registered(self) -> None:
        missing_path = self.temp_path / "missing.csv"

        result = self.page.add_dataset(missing_path)

        self.assertFalse(result)
        self.assertEqual(self.page.dataset_count, 0)
        self.assertIsNone(self.page.current_dataset)
        self.assertIn(
            "Could not load CSV",
            self.page._status_label.text(),
        )

    def test_export_includes_rows_hidden_by_search(self) -> None:
        csv_path = self.create_csv(
            "students.csv",
            "name,age\nAlice,17\nBob,18\n",
        )

        self.assertTrue(self.page.add_dataset(csv_path))

        self.page._search_input.setText("Alice")
        self.assertEqual(self.page._proxy_model.rowCount(), 1)

        export_path = self.temp_path / "exported.csv"

        result = self.page.export_current_dataset(export_path)

        self.assertTrue(result)
        self.assertTrue(export_path.exists())

        exported_text = export_path.read_text(
            encoding="utf-8-sig"
        )

        self.assertIn("Alice,17", exported_text)
        self.assertIn("Bob,18", exported_text)


if __name__ == "__main__":
    unittest.main()
