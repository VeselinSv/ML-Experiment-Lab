
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_experiment_lab.datasets import (  # noqa: E402
    DatasetLoadError,
    load_csv,
)


class LoadCSVTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_loads_semicolon_delimited_csv(self) -> None:
        csv_path = self.temp_path / "sample.csv"
        csv_path.write_text(
            "name;age\nAlice;17\nBob;18\n",
            encoding="utf-8",
        )

        dataset = load_csv(csv_path)

        self.assertEqual(dataset.columns, ("name", "age"))
        self.assertEqual(dataset.row_count, 2)
        self.assertEqual(dataset.column_count, 2)
        self.assertEqual(
            dataset.preview(1),
            (("Alice", "17"),),
        )
        self.assertEqual(dataset.path, csv_path.resolve())

    def test_loads_utf8_bom_and_comma_delimiter(self) -> None:
        csv_path = self.temp_path / "bom.csv"
        csv_path.write_text(
            "\ufeffid,score\n1,9.5\n",
            encoding="utf-8",
        )

        dataset = load_csv(csv_path)

        self.assertEqual(dataset.columns, ("id", "score"))
        self.assertEqual(dataset.rows, (("1", "9.5"),))

    def test_preview_rejects_negative_limit(self) -> None:
        csv_path = self.temp_path / "sample.csv"
        csv_path.write_text("name\nAlice\n", encoding="utf-8")
        dataset = load_csv(csv_path)

        with self.assertRaises(ValueError):
            dataset.preview(-1)

    def test_missing_file_raises_helpful_error(self) -> None:
        with self.assertRaisesRegex(
            DatasetLoadError, "does not exist"
        ):
            load_csv(self.temp_path / "missing.csv")

    def test_wrong_extension_is_rejected(self) -> None:
        text_path = self.temp_path / "sample.txt"
        text_path.write_text(
            "name\nAlice\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(
            DatasetLoadError, r"\.csv"
        ):
            load_csv(text_path)

    def test_empty_file_is_rejected(self) -> None:
        csv_path = self.temp_path / "empty.csv"
        csv_path.write_text("", encoding="utf-8")

        with self.assertRaisesRegex(
            DatasetLoadError, "empty"
        ):
            load_csv(csv_path)

    def test_inconsistent_rows_are_rejected(self) -> None:
        csv_path = self.temp_path / "invalid.csv"
        csv_path.write_text(
            "name,age\nAlice,17\nBob\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            DatasetLoadError, "Invalid record"
        ):
            load_csv(csv_path)

    def test_duplicate_column_names_are_rejected(self) -> None:
        csv_path = self.temp_path / "duplicate.csv"
        csv_path.write_text(
            "name,name\nAlice,17\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            DatasetLoadError, "unique"
        ):
            load_csv(csv_path)


if __name__ == "__main__":
    unittest.main()
