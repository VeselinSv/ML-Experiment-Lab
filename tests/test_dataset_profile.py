
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

    def _load(self, name: str, contents: str):
        path = self.temp_path / name
        path.write_text(contents, encoding="utf-8")
        return profile_dataset(load_csv(path))

    def test_missing_percentage_and_numeric_statistics(self) -> None:
        profile = self._load(
            "students.csv",
            "age,score\n17,9.5\n18,\n19,8.0\n",
        )
        columns = {
            item.name: item for item in profile.columns
        }

        self.assertEqual(profile.total_cells, 6)
        self.assertEqual(profile.missing_cells, 1)
        self.assertAlmostEqual(
            profile.missing_percentage,
            100 / 6,
        )
        self.assertEqual(
            columns["age"].inferred_type,
            "integer",
        )
        self.assertEqual(columns["age"].minimum, 17.0)
        self.assertEqual(columns["age"].maximum, 19.0)
        self.assertEqual(columns["age"].mean, 18.0)
        self.assertEqual(columns["score"].missing_count, 1)
        self.assertAlmostEqual(
            columns["score"].missing_percentage,
            100 / 3,
        )
        self.assertIn(
            "missing_values",
            {issue.code for issue in profile.issues},
        )

    def test_fully_empty_column_is_reported(self) -> None:
        profile = self._load(
            "empty_column.csv",
            "name,score\nAlice,\nBob,\n",
        )
        columns = {
            item.name: item for item in profile.columns
        }

        self.assertEqual(
            columns["score"].inferred_type,
            "empty",
        )
        self.assertEqual(columns["score"].missing_count, 2)
        self.assertEqual(
            columns["score"].missing_percentage,
            100.0,
        )
        self.assertEqual(profile.empty_columns, ("score",))
        self.assertIn(
            "empty_column",
            {issue.code for issue in profile.issues},
        )

    def test_constant_column_is_reported(self) -> None:
        profile = self._load(
            "constant.csv",
            "approved\nyes\nyes\nyes\n",
        )
        column = profile.columns[0]

        self.assertEqual(column.inferred_type, "boolean")
        self.assertTrue(column.constant)
        self.assertEqual(
            profile.constant_columns,
            ("approved",),
        )
        self.assertIn(
            "constant_column",
            {issue.code for issue in profile.issues},
        )

    def test_mixed_numeric_and_text_values_are_reported(self) -> None:
        profile = self._load(
            "mixed.csv",
            "value\n4\nunknown\n6\n8\n",
        )
        column = profile.columns[0]

        self.assertEqual(column.inferred_type, "mixed")
        self.assertEqual(column.mixed_numeric_count, 3)
        self.assertEqual(column.mixed_other_count, 1)
        self.assertIn(
            "mixed_values",
            {issue.code for issue in profile.issues},
        )

    def test_duplicate_rows_are_reported(self) -> None:
        profile = self._load(
            "duplicates.csv",
            "value,label\n1,A\n1,A\n1,A\n2,B\n",
        )

        self.assertEqual(profile.duplicate_rows, 2)

        issue = next(
            item
            for item in profile.issues
            if item.code == "duplicate_rows"
        )
        self.assertEqual(issue.affected_count, 2)

    def test_iqr_outlier_is_flagged_as_potential_not_error(self) -> None:
        profile = self._load(
            "outliers.csv",
            "value\n10\n11\n12\n12\n100\n",
        )
        column = profile.columns[0]

        issue = next(
            item
            for item in profile.issues
            if item.code == "potential_outliers"
        )

        self.assertEqual(column.outlier_count, 1)
        self.assertEqual(issue.severity, "info")
        self.assertIn(
            "Do not remove them automatically",
            issue.recommendation,
        )

    def test_header_only_dataset_is_reported(self) -> None:
        profile = self._load(
            "empty.csv",
            "name,age\n",
        )

        self.assertEqual(profile.row_count, 0)
        self.assertEqual(profile.total_cells, 0)
        self.assertIn(
            "empty_dataset",
            {issue.code for issue in profile.issues},
        )

    def test_date_column_is_recognized(self) -> None:
        profile = self._load(
            "dates.csv",
            "created_at\n2026-01-01\n2026-01-02\n",
        )

        self.assertEqual(
            profile.columns[0].inferred_type,
            "date",
        )


if __name__ == "__main__":
    unittest.main()
