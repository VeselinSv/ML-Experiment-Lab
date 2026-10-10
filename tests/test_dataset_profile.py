
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_experiment_lab.datasets import (  # noqa: E402
    load_csv,
    profile_dataset,
)


class DatasetProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_profiles_numeric_categorical_and_missing_values(self) -> None:
        csv_path = self.temp_path / "students.csv"
        csv_path.write_text(
            "age,score,group\n"
            "17,9.5,A\n"
            "18,,B\n"
            "18,,B\n"
            "19,8.0,A\n",
            encoding="utf-8",
        )

        dataset = load_csv(csv_path)
        profile = profile_dataset(dataset)

        columns = {
            column.name: column
            for column in profile.columns
        }

        self.assertEqual(profile.row_count, 4)
        self.assertEqual(profile.column_count, 3)
        self.assertEqual(profile.missing_cells, 2)
        self.assertEqual(profile.duplicate_rows, 1)

        self.assertEqual(
            columns["age"].inferred_type,
            "integer",
        )
        self.assertEqual(columns["age"].minimum, 17.0)
        self.assertEqual(columns["age"].maximum, 19.0)
        self.assertEqual(columns["age"].mean, 18.0)

        self.assertEqual(
            columns["score"].inferred_type,
            "decimal",
        )
        self.assertEqual(columns["score"].missing_count, 2)
        self.assertEqual(columns["score"].minimum, 8.0)
        self.assertEqual(columns["score"].maximum, 9.5)

        self.assertEqual(
            columns["group"].inferred_type,
            "category",
        )
        self.assertEqual(columns["group"].unique_count, 2)

    def test_empty_column_is_reported(self) -> None:
        csv_path = self.temp_path / "empty_column.csv"
        csv_path.write_text(
            "name,score\nAlice,\nBob,\n",
            encoding="utf-8",
        )

        profile = profile_dataset(load_csv(csv_path))
        columns = {
            column.name: column
            for column in profile.columns
        }

        self.assertEqual(
            columns["score"].inferred_type,
            "empty",
        )
        self.assertEqual(columns["score"].missing_count, 2)
        self.assertEqual(columns["score"].unique_count, 0)
        self.assertIsNone(columns["score"].mean)

    def test_boolean_column_is_recognized(self) -> None:
        csv_path = self.temp_path / "boolean.csv"
        csv_path.write_text(
            "approved\ntrue\nfalse\ntrue\n",
            encoding="utf-8",
        )

        profile = profile_dataset(load_csv(csv_path))

        self.assertEqual(
            profile.columns[0].inferred_type,
            "boolean",
        )
        self.assertEqual(
            profile.columns[0].unique_count,
            2,
        )

    def test_date_column_is_recognized(self) -> None:
        csv_path = self.temp_path / "dates.csv"
        csv_path.write_text(
            "created_at\n"
            "2026-01-01\n"
            "2026-01-02\n",
            encoding="utf-8",
        )

        profile = profile_dataset(load_csv(csv_path))

        self.assertEqual(
            profile.columns[0].inferred_type,
            "date",
        )


if __name__ == "__main__":
    unittest.main()
