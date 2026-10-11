
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
import re

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

BOOLEAN_VALUES = {"true", "false", "yes", "no", "y", "n"}

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",
)


@dataclass(frozen=True)
class DataQualityIssue:
    """One detected data-quality issue and its recommendation."""

    code: str
    severity: str
    title: str
    description: str
    recommendation: str
    column_name: str | None = None
    affected_count: int = 0


@dataclass(frozen=True)
class ColumnProfile:
    """Statistics and inferred type for a single column."""

    name: str
    inferred_type: str
    missing_count: int
    unique_count: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    missing_percentage: float = 0.0
    non_missing_count: int = 0
    constant: bool = False
    outlier_count: int = 0
    mixed_numeric_count: int = 0
    mixed_other_count: int = 0


@dataclass(frozen=True)
class DatasetProfile:
    """Overall data-quality metrics and findings."""

    row_count: int
    column_count: int
    missing_cells: int
    duplicate_rows: int
    columns: tuple[ColumnProfile, ...]
    total_cells: int = 0
    missing_percentage: float = 0.0
    empty_columns: tuple[str, ...] = ()
    constant_columns: tuple[str, ...] = ()
    issues: tuple[DataQualityIssue, ...] = ()


def is_missing(value: str) -> bool:
    """Return True when a cell represents a missing value."""
    return value.strip().casefold() in MISSING_VALUES


def _parse_number(value: str) -> float | None:
    """Convert a string to a finite float when possible."""
    try:
        number = float(value.strip())
    except (ValueError, AttributeError):
        return None

    if not isfinite(number):
        return None

    return number


def _is_integer(value: str) -> bool:
    """Check whether a string represents an integer."""
    return re.fullmatch(r"[+-]?\d+", value.strip()) is not None


def _is_date_column(values: list[str]) -> bool:
    """Return True if every value matches a supported date format."""
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

    if all(_is_integer(value) for value in values):
        return "integer"

    parsed_numbers = [_parse_number(value) for value in values]
    numeric_count = sum(
        number is not None for number in parsed_numbers
    )

    if numeric_count == len(values):
        return "decimal"

    if _is_date_column(values):
        return "date"

    if (
        0 < numeric_count < len(values)
        and numeric_count >= 2
        and numeric_count / len(values) >= 0.5
    ):
        return "mixed"

    unique_count = len(set(value.strip() for value in values))
    unique_ratio = unique_count / len(values)
    category_limit = min(
        30,
        max(2, int((len(values) ** 0.5) * 2)),
    )

    if unique_count <= category_limit and unique_ratio <= 0.5:
        return "category"

    return "text"


def _quantile(
    sorted_values: list[float],
    fraction: float,
) -> float:
    """Calculate a quantile using linear interpolation."""
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower

    return (
        sorted_values[lower] * (1 - weight)
        + sorted_values[upper] * weight
    )


def _count_iqr_outliers(numbers: list[float]) -> int:
    """Count potential outliers using the 1.5 × IQR rule."""
    if len(numbers) < 4:
        return 0

    ordered = sorted(numbers)
    q1 = _quantile(ordered, 0.25)
    q3 = _quantile(ordered, 0.75)
    iqr = q3 - q1

    if iqr <= 0:
        return 0

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return sum(
        value < lower_bound or value > upper_bound
        for value in ordered
    )


def _profile_column(
    name: str,
    values: list[str],
) -> ColumnProfile:
    """Calculate statistics and quality metrics for one column."""
    missing_count = sum(is_missing(value) for value in values)

    present_values = [
        value.strip()
        for value in values
        if not is_missing(value)
    ]

    inferred_type = _infer_type(present_values)
    unique_count = len(set(present_values))
    total_value_count = len(values)

    missing_percentage = (
        missing_count / total_value_count * 100
        if total_value_count
        else 0.0
    )

    minimum = None
    maximum = None
    mean = None

    outlier_count = 0
    mixed_numeric_count = 0
    mixed_other_count = 0

    if inferred_type in {"integer", "decimal"}:
        numbers = [
            _parse_number(value)
            for value in present_values
        ]

        valid_numbers = [
            number
            for number in numbers
            if number is not None
        ]

        if valid_numbers:
            minimum = min(valid_numbers)
            maximum = max(valid_numbers)
            mean = sum(valid_numbers) / len(valid_numbers)
            outlier_count = _count_iqr_outliers(valid_numbers)

    elif inferred_type == "mixed":
        parsed = [
            _parse_number(value)
            for value in present_values
        ]

        numeric_values = [
            number
            for number in parsed
            if number is not None
        ]

        mixed_numeric_count = len(numeric_values)
        mixed_other_count = (
            len(present_values) - mixed_numeric_count
        )

    return ColumnProfile(
        name=name,
        inferred_type=inferred_type,
        missing_count=missing_count,
        unique_count=unique_count,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        missing_percentage=missing_percentage,
        non_missing_count=len(present_values),
        constant=bool(present_values) and unique_count == 1,
        outlier_count=outlier_count,
        mixed_numeric_count=mixed_numeric_count,
        mixed_other_count=mixed_other_count,
    )


def _build_issues(
    row_count: int,
    columns: tuple[ColumnProfile, ...],
    duplicate_rows: int,
) -> tuple[DataQualityIssue, ...]:
    """Create actionable, human-readable data-quality findings."""
    issues: list[DataQualityIssue] = []

    if row_count == 0:
        issues.append(
            DataQualityIssue(
                code="empty_dataset",
                severity="error",
                title="Dataset contains no data rows",
                description=(
                    "The file contains column headers but no data rows."
                ),
                recommendation=(
                    "Check that you selected the correct CSV file "
                    "and that it contains records."
                ),
            )
        )

    for column in columns:
        name = column.name

        if row_count > 0 and column.non_missing_count == 0:
            issues.append(
                DataQualityIssue(
                    code="empty_column",
                    severity="warning",
                    title="Column contains no values",
                    description=(
                        f"All {row_count} values in column "
                        f"'{name}' are missing."
                    ),
                    recommendation=(
                        "Check the data source. If this column is not "
                        "needed, consider removing it after review."
                    ),
                    column_name=name,
                    affected_count=row_count,
                )
            )

        elif column.missing_count > 0:
            severe = column.missing_percentage >= 20.0

            issues.append(
                DataQualityIssue(
                    code="missing_values",
                    severity="warning" if severe else "info",
                    title=(
                        "High number of missing values"
                        if severe
                        else "Missing values"
                    ),
                    description=(
                        f"Column '{name}' has {column.missing_count} "
                        f"missing values out of {row_count} "
                        f"({column.missing_percentage:.1f}%)."
                    ),
                    recommendation=(
                        "Investigate why values are missing. Choose an "
                        "appropriate imputation or row/column removal "
                        "strategy before model training."
                    ),
                    column_name=name,
                    affected_count=column.missing_count,
                )
            )

        if column.constant:
            issues.append(
                DataQualityIssue(
                    code="constant_column",
                    severity="info",
                    title="Constant-valued column",
                    description=(
                        f"All non-missing values in '{name}' are identical."
                    ),
                    recommendation=(
                        "A constant feature usually provides no information "
                        "for distinguishing observations. Check whether "
                        "it is needed."
                    ),
                    column_name=name,
                    affected_count=column.non_missing_count,
                )
            )

        if column.inferred_type == "mixed":
            issues.append(
                DataQualityIssue(
                    code="mixed_values",
                    severity="warning",
                    title="Potentially mixed data types",
                    description=(
                        f"Column '{name}' contains "
                        f"{column.mixed_numeric_count} numeric and "
                        f"{column.mixed_other_count} non-numeric values."
                    ),
                    recommendation=(
                        "Review the non-numeric values. They may be errors, "
                        "special markers, or valid categories."
                    ),
                    column_name=name,
                    affected_count=column.mixed_other_count,
                )
            )

        if column.outlier_count > 0:
            issues.append(
                DataQualityIssue(
                    code="potential_outliers",
                    severity="info",
                    title="Potential outliers",
                    description=(
                        f"Numeric column '{name}' contains "
                        f"{column.outlier_count} potential outlier(s) "
                        "according to the 1.5 × IQR rule."
                    ),
                    recommendation=(
                        "Review these values in context. Do not remove "
                        "them automatically; they may be valid observations."
                    ),
                    column_name=name,
                    affected_count=column.outlier_count,
                )
            )

    if duplicate_rows > 0:
        issues.append(
            DataQualityIssue(
                code="duplicate_rows",
                severity="warning",
                title="Duplicate rows",
                description=(
                    f"Found {duplicate_rows} repeated row occurrence(s), "
                    "excluding the first occurrence of each row."
                ),
                recommendation=(
                    "Review repeated rows and confirm whether they are "
                    "duplicates or valid separate observations before "
                    "removing them."
                ),
                affected_count=duplicate_rows,
            )
        )

    severity_order = {
        "error": 0,
        "warning": 1,
        "info": 2,
    }

    issues.sort(
        key=lambda issue: (
            severity_order.get(issue.severity, 3),
            issue.column_name or "",
            issue.code,
        )
    )

    return tuple(issues)


def profile_dataset(dataset: LoadedDataset) -> DatasetProfile:
    """Inspect a dataset and return data-quality metrics and findings."""
    columns: list[ColumnProfile] = []

    for column_index, column_name in enumerate(dataset.columns):
        values = [
            row[column_index]
            for row in dataset.rows
        ]

        columns.append(_profile_column(column_name, values))

    column_profiles = tuple(columns)

    missing_cells = sum(
        column.missing_count for column in column_profiles
    )

    total_cells = dataset.row_count * dataset.column_count

    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells
        else 0.0
    )

    # Count repeated rows after the first occurrence.
    seen_rows: set[tuple[str, ...]] = set()
    duplicate_rows = 0

    for row in dataset.rows:
        if row in seen_rows:
            duplicate_rows += 1
        else:
            seen_rows.add(row)

    empty_columns = tuple(
        column.name
        for column in column_profiles
        if dataset.row_count > 0
        and column.non_missing_count == 0
    )

    constant_columns = tuple(
        column.name
        for column in column_profiles
        if column.constant
    )

    issues = _build_issues(
        dataset.row_count,
        column_profiles,
        duplicate_rows,
    )

    return DatasetProfile(
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        missing_cells=missing_cells,
        duplicate_rows=duplicate_rows,
        columns=column_profiles,
        total_cells=total_cells,
        missing_percentage=missing_percentage,
        empty_columns=empty_columns,
        constant_columns=constant_columns,
        issues=issues,
    )
