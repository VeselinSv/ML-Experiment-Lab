
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite, sqrt

from .loader import LoadedDataset


MISSING_VALUES = {
    "",
    "na",
    "n/a",
    "null",
    "none",
    "nan",
    "<na>",
    "missing",
}

BOOLEAN_VALUES = {
    "true",
    "false",
    "yes",
    "no",
    "y",
    "n",
}

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",
)


@dataclass(frozen=True)
class ColumnProfile:
    """Statistics and inferred type for one dataset column."""

    name: str
    inferred_type: str
    missing_count: int
    unique_count: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None


@dataclass(frozen=True)
class DatasetProfile:
    """Summary and column profiles for a complete dataset."""

    row_count: int
    column_count: int
    missing_cells: int
    duplicate_rows: int
    columns: tuple[ColumnProfile, ...]


def is_missing(value: str) -> bool:
    """Return True if a cell represents a missing value."""
    return value.strip().casefold() in MISSING_VALUES


def _parse_number(value: str) -> float | None:
    """Parse a finite number; return None for non-numeric text."""
    try:
        number = float(value.strip())
    except (ValueError, AttributeError):
        return None

    if not isfinite(number):
        return None

    return number


def _is_date_column(values: list[str]) -> bool:
    """Check whether all supplied values match supported date formats."""
    if not values:
        return False

    for value in values:
        parsed = False

        for date_format in DATE_FORMATS:
            try:
                datetime.strptime(value.strip(), date_format)
                parsed = True
                break
            except ValueError:
                continue

        if not parsed:
            return False

    return True


def _infer_type(values: list[str]) -> str:
    """Infer a practical starting type for a column."""
    if not values:
        return "empty"

    normalized = [value.strip().casefold() for value in values]

    if all(value in BOOLEAN_VALUES for value in normalized):
        return "boolean"

    try:
        for value in values:
            int(value.strip())
        return "integer"
    except ValueError:
        pass

    numbers = [_parse_number(value) for value in values]

    if all(number is not None for number in numbers):
        return "decimal"

    if _is_date_column(values):
        return "date"

    unique_count = len(set(values))
    unique_ratio = unique_count / len(values)

    # A small number of recurring text values often indicates
    # a categorical feature, such as city, colour, or product type.
    category_limit = min(30, max(2, int(sqrt(len(values)) * 2)))

    if unique_count <= category_limit and unique_ratio <= 0.5:
        return "category"

    return "text"


def _profile_column(
    name: str,
    values: list[str],
) -> ColumnProfile:
    """Calculate statistics for one column."""
    missing_count = sum(is_missing(value) for value in values)

    present_values = [
        value.strip()
        for value in values
        if not is_missing(value)
    ]

    inferred_type = _infer_type(present_values)
    unique_count = len(set(present_values))

    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None

    if inferred_type in {"integer", "decimal"}:
        numbers = [
            _parse_number(value)
            for value in present_values
        ]

        # _infer_type has already verified that the values are numeric.
        valid_numbers = [
            number for number in numbers
            if number is not None
        ]

        if valid_numbers:
            minimum = min(valid_numbers)
            maximum = max(valid_numbers)
            mean = sum(valid_numbers) / len(valid_numbers)

    return ColumnProfile(
        name=name,
        inferred_type=inferred_type,
        missing_count=missing_count,
        unique_count=unique_count,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
    )


def profile_dataset(dataset: LoadedDataset) -> DatasetProfile:
    """Inspect a loaded dataset and return its data-quality profile."""
    row_count = dataset.row_count
    column_count = dataset.column_count

    column_profiles: list[ColumnProfile] = []

    for column_index, column_name in enumerate(dataset.columns):
        values = [
            row[column_index]
            for row in dataset.rows
        ]

        column_profiles.append(
            _profile_column(column_name, values)
        )

    missing_cells = sum(
        column.missing_count
        for column in column_profiles
    )

    # Count repeated rows after their first occurrence.
    # For example, if the same row occurs three times,
    # two of those occurrences count as duplicates.
    seen_rows: set[tuple[str, ...]] = set()
    duplicate_rows = 0

    for row in dataset.rows:
        if row in seen_rows:
            duplicate_rows += 1
        else:
            seen_rows.add(row)

    return DatasetProfile(
        row_count=row_count,
        column_count=column_count,
        missing_cells=missing_cells,
        duplicate_rows=duplicate_rows,
        columns=tuple(column_profiles),
    )
