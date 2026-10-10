
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Qt трябва да работи и в среда без видим работен плот.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from ml_experiment_lab.ui.widgets.dataset_import_widget import (  # noqa: E402
    DatasetImportWidget,
)


class DatasetImportWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.widget = DatasetImportWidget()

    def tearDown(self) -> None:
        self.widget.close()
        self.widget.deleteLater()
        self.temp_dir.cleanup()
        self.app.processEvents()

    def test_load_file_displays_dataset(self) -> None:
        csv_path = self.temp_path / "students.csv"
        csv_path.write_text(
            "name,age\nAlice,17\nBob,18\n",
            encoding="utf-8",
        )

        received_datasets = []
        self.widget.dataset_loaded.connect(
            received_datasets.append
        )

        result = self.widget.load_file(csv_path)

        self.assertTrue(result)
        self.assertIsNotNone(self.widget.dataset)
        self.assertEqual(self.widget.dataset.row_count, 2)
        self.assertEqual(self.widget.dataset.column_count, 2)

        self.assertEqual(self.widget._table.rowCount(), 2)
        self.assertEqual(self.widget._table.columnCount(), 2)
        self.assertEqual(
            self.widget._table.horizontalHeaderItem(0).text(),
            "name",
        )
        self.assertEqual(
            self.widget._table.item(0, 0).text(),
            "Alice",
        )
        self.assertEqual(len(received_datasets), 1)
        self.assertIn(
            "successfully",
            self.widget._status_label.text(),
        )

    def test_missing_file_returns_false(self) -> None:
        missing_path = self.temp_path / "missing.csv"

        result = self.widget.load_file(missing_path)

        self.assertFalse(result)
        self.assertIsNone(self.widget.dataset)
        self.assertIn(
            "Could not load CSV",
            self.widget._status_label.text(),
        )

    def test_only_first_100_rows_are_displayed(self) -> None:
        csv_path = self.temp_path / "large.csv"

        records = ["value"]
        records.extend(str(number) for number in range(105))
        csv_path.write_text(
            "\n".join(records) + "\n",
            encoding="utf-8",
        )

        result = self.widget.load_file(csv_path)

        self.assertTrue(result)
        self.assertEqual(self.widget.dataset.row_count, 105)
        self.assertEqual(self.widget._table.rowCount(), 100)
        self.assertIn(
            "first 100 rows",
            self.widget._summary_label.text(),
        )

    def test_failed_load_preserves_previous_dataset(self) -> None:
        csv_path = self.temp_path / "valid.csv"
        csv_path.write_text(
            "name\nAlice\n",
            encoding="utf-8",
        )

        self.assertTrue(self.widget.load_file(csv_path))
        original_dataset = self.widget.dataset

        result = self.widget.load_file(
            self.temp_path / "missing.csv"
        )

        self.assertFalse(result)
        self.assertIs(self.widget.dataset, original_dataset)
        self.assertIn(
            "previous dataset remains displayed",
            self.widget._status_label.text(),
        )


if __name__ == "__main__":
    unittest.main()
