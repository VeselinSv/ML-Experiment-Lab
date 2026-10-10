
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


class DatasetLoadError(ValueError):
    """Грешка при прочитане или проверка на CSV файл."""


@dataclass(frozen=True)
class LoadedDataset:
    """Данни, прочетени от CSV файл."""

    path: Path
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @property
    def row_count(self) -> int:
        """Брой редове с данни, без заглавния ред."""
        return len(self.rows)

    @property
    def column_count(self) -> int:
        """Брой колони."""
        return len(self.columns)

    def preview(
        self, limit: int = 5
    ) -> tuple[tuple[str, ...], ...]:
        """Връща до limit реда за предварителен преглед."""
        if limit < 0:
            raise ValueError("limit must be zero or greater")

        return self.rows[:limit]


def load_csv(file_path: str | Path) -> LoadedDataset:
    """Прочита CSV файл и проверява структурата му."""

    path = Path(file_path).expanduser()

    if not path.exists():
        raise DatasetLoadError(
            f"File does not exist: {path}"
        )

    if not path.is_file():
        raise DatasetLoadError(
            f"The selected path is not a file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise DatasetLoadError(
            "The selected file must have a .csv extension."
        )

    try:
        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            sample = file.read(4096)

            if not sample.strip():
                raise DatasetLoadError(
                    "The CSV file is empty."
                )

            file.seek(0)

            try:
                dialect = csv.Sniffer().sniff(
                    sample, delimiters=",;\t|"
                )
            except csv.Error:
                dialect = csv.excel

            reader = csv.reader(file, dialect)
            raw_headers = next(reader, None)

            if raw_headers is None:
                raise DatasetLoadError(
                    "The CSV file is empty."
                )

            columns = tuple(
                header.strip() for header in raw_headers
            )

            if not columns or all(not col for col in columns):
                raise DatasetLoadError(
                    "The first row must contain column names."
                )

            if any(not col for col in columns):
                raise DatasetLoadError(
                    "Every column must have a name."
                )

            if len(set(columns)) != len(columns):
                raise DatasetLoadError(
                    "Column names must be unique."
                )

            rows: list[tuple[str, ...]] = []

            for record_number, raw_row in enumerate(
                reader, start=2
            ):
                # Пропускаме празните редове.
                if not raw_row or all(
                    not value.strip() for value in raw_row
                ):
                    continue

                if len(raw_row) != len(columns):
                    raise DatasetLoadError(
                        f"Invalid record {record_number}: "
                        f"expected {len(columns)} values, "
                        f"but found {len(raw_row)}."
                    )

                rows.append(tuple(raw_row))

    except UnicodeDecodeError as exc:
        raise DatasetLoadError(
            "The file could not be read as UTF-8 text. "
            "Save it as UTF-8 and try again."
        ) from exc

    except csv.Error as exc:
        raise DatasetLoadError(
            f"The CSV format is invalid: {exc}"
        ) from exc

    except OSError as exc:
        raise DatasetLoadError(
            f"Could not open the CSV file: {exc}"
        ) from exc

    return LoadedDataset(
        path=path.resolve(),
        columns=columns,
        rows=tuple(rows),
    )
