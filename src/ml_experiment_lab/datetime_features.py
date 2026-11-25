"""Разпознаване и обработка на датни характеристики."""

from __future__ import annotations

import math
import re
from datetime import datetime


DATETIME_FEATURE_NAMES = (
    "year",
    "month",
    "day",
    "day_of_week",
    "quarter",
    "day_of_year",
    "hour",
    "minute",
    "second",
)


# Международен формат:
# 2026-11-25
# 2026-11-25 14:30
# 2026-11-25T14:30:45
ISO_DATETIME_PATTERN = re.compile(
    r"(?P<year>[0-9]{4})-"
    r"(?P<month>[0-9]{2})-"
    r"(?P<day>[0-9]{2})"
    r"(?:[ T]"
    r"(?P<hour>[0-9]{2}):"
    r"(?P<minute>[0-9]{2})"
    r"(?::(?P<second>[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,6}))?"
    r")?"
    r")?"
)


# Български формат:
# 25.11.2026
# 5.1.2026
# 25.11.2026 14:30:45
BULGARIAN_DATETIME_PATTERN = re.compile(
    r"(?P<day>[0-9]{1,2})\."
    r"(?P<month>[0-9]{1,2})\."
    r"(?P<year>[0-9]{4})"
    r"(?: "
    r"(?P<hour>[0-9]{2}):"
    r"(?P<minute>[0-9]{2})"
    r"(?::(?P<second>[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,6}))?"
    r")?"
    r")?"
)


def parse_datetime_value(
    value: str,
) -> datetime | None:
    """Преобразува текстова стойност в дата или дата и час.

    Поддържа международния ISO формат и българския
    формат с ден, месец и година.

    Връща None при липсваща или невалидна стойност.
    """

    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value:
        return None

    match = ISO_DATETIME_PATTERN.fullmatch(value)

    if match is None:
        match = BULGARIAN_DATETIME_PATTERN.fullmatch(
            value
        )

    if match is None:
        return None

    parts = match.groupdict()

    fraction = parts.get("fraction") or ""

    microsecond = int(
        fraction.ljust(6, "0")
    )

    try:
        return datetime(
            year=int(parts["year"]),
            month=int(parts["month"]),
            day=int(parts["day"]),
            hour=int(parts["hour"] or 0),
            minute=int(parts["minute"] or 0),
            second=int(parts["second"] or 0),
            microsecond=microsecond,
        )

    except ValueError:
        return None


def is_datetime_value(value: str) -> bool:
    """Проверява дали стойността е валидна дата."""

    return parse_datetime_value(value) is not None


def extract_datetime_features(
    value: str,
) -> tuple[float, ...]:
    """Извлича числови характеристики от дата.

    При невалидна или липсваща дата връща NaN
    за всяка датна характеристика.
    """

    parsed_value = parse_datetime_value(value)

    if parsed_value is None:
        return (
            math.nan,
        ) * len(DATETIME_FEATURE_NAMES)

    quarter = (
        (parsed_value.month - 1) // 3
    ) + 1

    day_of_year = (
        parsed_value.timetuple().tm_yday
    )

    return (
        float(parsed_value.year),
        float(parsed_value.month),
        float(parsed_value.day),
        float(parsed_value.weekday()),
        float(quarter),
        float(day_of_year),
        float(parsed_value.hour),
        float(parsed_value.minute),
        float(parsed_value.second),
    )