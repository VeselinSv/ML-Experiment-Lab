
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
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
    load_csv,
)


class DatasetImportWidget(QWidget):
    """Widget for selecting a CSV file and previewing its data."""

    PREVIEW_LIMIT = 100

    # Изпраща събитието, когато нов набор от данни е зареден.
    dataset_loaded = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._dataset: LoadedDataset | None = None

        self._open_button = QPushButton("Open CSV file...")
        self._open_button.clicked.connect(self._choose_file)

        self._file_label = QLabel("No file selected.")
        self._summary_label = QLabel("No dataset loaded.")
        self._status_label = QLabel(
            "Choose a CSV file to preview its contents."
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
        """Return the currently loaded dataset."""
        return self._dataset

    def _choose_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a CSV file",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        if not file_path:
            # Потребителят е затворил диалога без избор.
            return

        self.load_file(file_path)

    def load_file(self, file_path: str | Path) -> bool:
        """Load a CSV file and display a preview.

        Returns True on success and False if the file cannot be loaded.
        If loading fails, a previously loaded dataset remains available.
        """
        try:
            dataset = load_csv(file_path)
        except DatasetLoadError as exc:
            message = f"Could not load CSV: {exc}"

            if self._dataset is not None:
                message += (
                    " The previous dataset remains displayed."
                )

            self._status_label.setText(message)
            return False

        # Променяме текущия набор едва след успешното прочитане.
        self._dataset = dataset
        self._display_dataset(dataset)

        self.dataset_loaded.emit(dataset)
        return True

    def _display_dataset(self, dataset: LoadedDataset) -> None:
        preview_rows = dataset.preview(self.PREVIEW_LIMIT)

        self._file_label.setText(
            f"Loaded file: {dataset.path}"
        )

        self._summary_label.setText(
            f"{dataset.row_count} rows · "
            f"{dataset.column_count} columns"
        )

        if dataset.row_count > len(preview_rows):
            self._summary_label.setText(
                f"{dataset.row_count} rows · "
                f"{dataset.column_count} columns · "
                f"showing the first {len(preview_rows)} rows"
            )

        self._table.clear()
        self._table.setColumnCount(dataset.column_count)
        self._table.setHorizontalHeaderLabels(
            list(dataset.columns)
        )
        self._table.setRowCount(len(preview_rows))

        for row_index, row in enumerate(preview_rows):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem(value)

                # Таблицата е само за преглед на данните.
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
                "CSV loaded successfully. "
                "The file contains column names but no data rows."
            )
        elif dataset.row_count > len(preview_rows):
            self._status_label.setText(
                f"CSV loaded successfully. "
                f"The table displays only the first "
                f"{len(preview_rows)} rows."
            )
        else:
            self._status_label.setText(
                "CSV loaded successfully."
            )
