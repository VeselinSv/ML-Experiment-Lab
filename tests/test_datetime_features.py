"""Тестове за разпознаване и обработка на дати."""

import math
import unittest
from datetime import datetime

from ml_experiment_lab.datetime_features import (
    DATETIME_FEATURE_NAMES,
    extract_datetime_features,
    is_datetime_value,
    parse_datetime_value,
)


class DatetimeFeaturesTests(unittest.TestCase):
    """Проверява обработката на датни характеристики."""

    def test_iso_date(self):
        """Проверява международния формат за дата."""

        result = parse_datetime_value(
            "2026-11-25"
        )

        self.assertEqual(
            result,
            datetime(2026, 11, 25),
        )

    def test_bulgarian_date(self):
        """Проверява българския формат за дата."""

        self.assertEqual(
            parse_datetime_value("25.11.2026"),
            datetime(2026, 11, 25),
        )

        self.assertEqual(
            parse_datetime_value("5.1.2026"),
            datetime(2026, 1, 5),
        )

    def test_datetime_with_time(self):
        """Проверява дати, съдържащи час."""

        expected = datetime(
            2026, 11, 25, 14, 30, 45
        )

        values = (
            "2026-11-25 14:30:45",
            "2026-11-25T14:30:45",
            "25.11.2026 14:30:45",
        )

        for value in values:
            with self.subTest(value=value):
                self.assertEqual(
                    parse_datetime_value(value),
                    expected,
                )

        self.assertEqual(
            parse_datetime_value(
                "25.11.2026 14:30"
            ),
            datetime(2026, 11, 25, 14, 30),
        )

    def test_fractional_seconds(self):
        """Проверява дата с дробни секунди."""

        result = parse_datetime_value(
            "2026-11-25 14:30:45.123"
        )

        self.assertEqual(
            result,
            datetime(
                2026,
                11,
                25,
                14,
                30,
                45,
                123000,
            ),
        )

    def test_invalid_dates(self):
        """Проверява отхвърлянето на невалидни дати."""

        invalid_values = (
            "31.04.2026",
            "32.01.2026",
            "2026-13-01",
            "2026-00-01",
            "2026-11-00",
            "2026-11-25 25:30:00",
            "2026-11-25 14:60:00",
            "2026-11-25 14:30:60",
        )

        for value in invalid_values:
            with self.subTest(value=value):
                self.assertIsNone(
                    parse_datetime_value(value)
                )

    def test_leap_year(self):
        """Проверява обработката на високосни години."""

        self.assertEqual(
            parse_datetime_value("29.02.2024"),
            datetime(2024, 2, 29),
        )

        self.assertIsNone(
            parse_datetime_value("29.02.2025")
        )

        self.assertTrue(
            is_datetime_value("2000-02-29")
        )

        self.assertFalse(
            is_datetime_value("1900-02-29")
        )

    def test_non_datetime_values(self):
        """Проверява отхвърлянето на други типове."""

        invalid_values = (
            "25,50",
            "25.50",
            "2026",
            "1000",
            "1e3",
            "София",
            "Неизвестен",
            "01/02/2026",
            "14:30:45",
            "2026-11-25T14:30:45Z",
            "2026-11-25T14:30:45+02:00",
        )

        for value in invalid_values:
            with self.subTest(value=value):
                self.assertFalse(
                    is_datetime_value(value)
                )

    def test_extract_datetime_features(self):
        """Проверява извличането на характеристиките."""

        result = extract_datetime_features(
            "25.11.2026 14:30:45"
        )

        expected = (
            2026.0,
            11.0,
            25.0,
            2.0,
            4.0,
            329.0,
            14.0,
            30.0,
            45.0,
        )

        self.assertEqual(
            result,
            expected,
        )

        self.assertEqual(
            len(result),
            len(DATETIME_FEATURE_NAMES),
        )

    def test_date_without_time(self):
        """Проверява дата без зададен час."""

        result = extract_datetime_features(
            "2026-01-01"
        )

        expected = (
            2026.0,
            1.0,
            1.0,
            3.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
        )

        self.assertEqual(
            result,
            expected,
        )

    def test_missing_and_invalid_values(self):
        """Проверява NaN при липсваща или невалидна дата."""

        values = (
            "",
            "   ",
            "Неизвестен",
            "31.02.2026",
        )

        for value in values:
            with self.subTest(value=value):
                result = extract_datetime_features(
                    value
                )

                self.assertEqual(
                    len(result),
                    len(DATETIME_FEATURE_NAMES),
                )

                self.assertTrue(
                    all(
                        math.isnan(feature)
                        for feature in result
                    )
                )


if __name__ == "__main__":
    unittest.main()