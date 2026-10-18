
"""Компонент за импортиране и преглед на CSV и Excel файлове."""

from __future__ import annotations

from pathlib import Path
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ml_experiment_lab.datasets import (
    DatasetLoadError,
    LoadedDataset,
    get_excel_sheet_names,
    load_dataset,
)


class DatasetImportWidget(QWidget):
    """Избор на CSV или Excel файл и преглед на данните."""

    PREVIEW_LIMIT = 100

    dataset_loaded = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._dataset: LoadedDataset | None = None

        self._open_button = QPushButton("Избор на файл...")
        self._open_button.clicked.connect(self._choose_file)

        self._file_label = QLabel("Не е избран файл.")
        self._file_label.setWordWrap(True)

        self._summary_label = QLabel("Няма зареден набор от данни.")
        self._status_label = QLabel(
            "Изберете CSV или Excel файл, за да прегледате данните."
        )
        self._status_label.setWordWrap(True)

        self._table = QTableWidget()
        self._table.setColumnCount(0)
        self._table.setRowCount(0)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._table.setMinimumHeight(220)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._open_button)
        button_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(button_layout)
        layout.addWidget(self._file_label)
        layout.addWidget(self._summary_label)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._status_label)

    @property
    def dataset(self) -> LoadedDataset | None:
        """Връща текущия зареден набор от данни."""
        return self._dataset

    @staticmethod
    def _format_load_error(error: DatasetLoadError) -> str:
        """Превежда често срещаните грешки при импорт."""
        message = str(error)

        if message.startswith("File does not exist:"):
            path = message.split(":", 1)[1].strip()
            return f"Файлът не съществува: {path}"

        if message.startswith("The selected path is not a file:"):
            path = message.split(":", 1)[1].strip()
            return f"Избраният път не сочи към файл: {path}"

        if "must have a .csv extension" in message:
            return "Изберете CSV файл с разширение .csv."

        if "must have an .xlsx extension" in message:
            return "Изберете Excel файл с разширение .xlsx."

        if "Supported file types" in message:
            return "Поддържат се файлове с разширение .csv и .xlsx."

        if message == "The CSV file is empty.":
            return "CSV файлът е празен."

        if "selected worksheet is empty" in message:
            return "Избраният работен лист е празен."

        if "Worksheet not found:" in message:
            return "Избраният работен лист не е намерен."

        if "Could not open the Excel workbook:" in message:
            return (
                "Excel файлът не може да бъде отворен. "
                "Проверете дали е валиден и не е повреден."
            )

        if "Excel import requires openpyxl" in message:
            return (
                "За импортиране на Excel е необходима библиотеката "
                "openpyxl. Инсталирайте зависимостите на проекта."
            )

        if "first row must contain column names" in message:
            return "Първият ред трябва да съдържа имената на колоните."

        if "Every column must have a name" in message:
            return "Всяка колона трябва да има име."

        if "Column names must be unique" in message:
            return "Имената на колоните трябва да са уникални."

        if message.startswith("Invalid record "):
            match = re.search(r"Invalid record (\d+)", message)
            if match:
                return (
                    f"Ред {match.group(1)} има невалидна структура."
                )

        if "could not be read as UTF-8" in message:
            return (
                "Файлът не може да бъде прочетен като UTF-8. "
                "Запазете го в тази кодировка и опитайте отново."
            )

        if message.startswith("The CSV format is invalid:"):
            return (
                "Структурата на CSV файла е невалидна. "
                "Проверете разделителите и кавичките."
            )

        return (
            "Файлът не може да бъде зареден. "
            "Проверете формата и съдържанието му."
        )

    def _choose_file(self) -> None:
        """Отваря диалог за избор на CSV или Excel файл."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Избор на файл с данни",
            "",
            (
                "Поддържани файлове (*.csv *.xlsx);;"
                "CSV файлове (*.csv);;"
                "Excel файлове (*.xlsx);;"
                "Всички файлове (*)"
            ),
        )

        if not file_path:
            return

        path = Path(file_path)
        sheet_name: str | None = None

        if path.suffix.casefold() == ".xlsx":
            try:
                sheet_names = get_excel_sheet_names(path)
            except DatasetLoadError as exc:
                self._status_label.setText(
                    "Неуспешно зареждане: "
                    f"{self._format_load_error(exc)}"
                )
                return

            if not sheet_names:
                self._status_label.setText(
                    "Excel файлът не съдържа работни листове."
                )
                return

            if len(sheet_names) == 1:
                sheet_name = sheet_names[0]
            else:
                sheet_name, accepted = QInputDialog.getItem(
                    self,
                    "Избор на работен лист",
                    (
                        f"Excel файлът „{path.name}“ съдържа няколко "
                        "листа. Изберете кой да бъде зареден:"
                    ),
                    list(sheet_names),
                    0,
                    False,
                )

                if not accepted:
                    return

        self.load_file(path, sheet_name=sheet_name)

    def load_file(
        self,
        file_path: str | Path,
        *,
        sheet_name: str | None = None,
    ) -> bool:
        """Зарежда CSV или Excel файл и показва предварителен преглед.

        При неуспешен импорт предишният набор остава наличен.
        """
        try:
            dataset = load_dataset(
                file_path,
                sheet_name=sheet_name,
            )
        except DatasetLoadError as exc:
            message = (
                "Неуспешно зареждане: "
                f"{self._format_load_error(exc)}"
            )

            if self._dataset is not None:
                message += (
                    " Предишният набор от данни остава показан."
                )

            self._status_label.setText(message)
            return False

        self._dataset = dataset
        self._display_dataset(dataset)
        self.dataset_loaded.emit(dataset)
        return True

    def _display_dataset(self, dataset: LoadedDataset) -> None:
        """Показва информация и първите редове от набора."""
        preview_rows = dataset.preview(self.PREVIEW_LIMIT)

        file_text = f"Зареден файл: {dataset.path}"

        if dataset.sheet_name is not None:
            file_text += f" | Работен лист: {dataset.sheet_name}"

        self._file_label.setText(file_text)
        self._file_label.setToolTip(str(dataset.path))

        self._summary_label.setText(
            f"Редове: {dataset.row_count} · "
            f"Колони: {dataset.column_count}"
        )

        if dataset.row_count > len(preview_rows):
            self._summary_label.setText(
                f"Редове: {dataset.row_count} · "
                f"Колони: {dataset.column_count} · "
                f"Показани са първите {len(preview_rows)} реда"
            )

        self._table.clear()
        self._table.setColumnCount(dataset.column_count)
        self._table.setHorizontalHeaderLabels(list(dataset.columns))
        self._table.setRowCount(len(preview_rows))

        for row_index, row in enumerate(preview_rows):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setFlags(
                    item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                self._table.setItem(
                    row_index,
                    column_index,
                    item,
                )

        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setStretchLastSection(True)

        if dataset.row_count == 0:
            self._status_label.setText(
                "Файлът е зареден, но съдържа само имена на колони "
                "и няма редове с данни."
            )
        elif dataset.row_count > len(preview_rows):
            self._status_label.setText(
                "Файлът е зареден успешно. "
                f"Показани са само първите {len(preview_rows)} реда."
            )
        else:
            self._status_label.setText(
                "Файлът е зареден успешно."
            )
