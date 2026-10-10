
from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ml_experiment_lab.datasets import (
    DatasetLoadError,
    DatasetProfile,
    LoadedDataset,
    load_csv,
    profile_dataset,
)


class DatasetTableModel(QAbstractTableModel):
    """Read-only Qt table model backed by a LoadedDataset."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: LoadedDataset | None = None

    def set_dataset(self, dataset: LoadedDataset | None) -> None:
        """Replace the displayed dataset and refresh the view."""
        self.beginResetModel()
        self._dataset = dataset
        self.endResetModel()

    def rowCount(
        self,
        parent: QModelIndex = QModelIndex(),
    ) -> int:
        if parent.isValid() or self._dataset is None:
            return 0

        return self._dataset.row_count

    def columnCount(
        self,
        parent: QModelIndex = QModelIndex(),
    ) -> int:
        if parent.isValid() or self._dataset is None:
            return 0

        return self._dataset.column_count

    def data(
        self,
        index: QModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if not index.isValid() or self._dataset is None:
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return self._dataset.rows[index.row()][index.column()]

        if role == Qt.ItemDataRole.ToolTipRole:
            return self._dataset.rows[index.row()][index.column()]

        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if self._dataset is None:
            return None

        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < self._dataset.column_count:
                return self._dataset.columns[section]

            return None

        return str(section + 1)


class AllColumnsFilterProxyModel(QSortFilterProxyModel):
    """Filter dataset rows by searching every column."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._search_text = ""

        self.setDynamicSortFilter(True)
        self.setFilterCaseSensitivity(
            Qt.CaseSensitivity.CaseInsensitive
        )

    def set_search_text(self, text: str) -> None:
        """Set the text used to filter rows."""
        normalized_text = text.strip()

        if normalized_text == self._search_text:
            return

        self._search_text = normalized_text
        self.invalidateFilter()

    def filterAcceptsRow(
        self,
        source_row: int,
        source_parent: QModelIndex,
    ) -> bool:
        if not self._search_text:
            return True

        source_model = self.sourceModel()

        if source_model is None:
            return True

        needle = self._search_text.casefold()

        for column in range(
            source_model.columnCount(source_parent)
        ):
            index = source_model.index(
                source_row,
                column,
                source_parent,
            )

            value = source_model.data(
                index,
                Qt.ItemDataRole.DisplayRole,
            )

            if value is not None and needle in str(value).casefold():
                return True

        return False


class DatasetsPage(QWidget):
    """Dataset workspace: import, inspect, profile, and export CSV files."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # The datasets are held in memory for this application session.
        # The resolved file path is used as the unique key.
        self._datasets: dict[str, LoadedDataset] = {}
        self._current_path: str | None = None

        self._build_ui()

    @property
    def dataset_count(self) -> int:
        """Number of unique datasets loaded in this session."""
        return len(self._datasets)

    @property
    def current_dataset(self) -> LoadedDataset | None:
        """Currently selected dataset, if one exists."""
        if self._current_path is None:
            return None

        return self._datasets.get(self._current_path)

    def _build_ui(self) -> None:
        page_layout = QVBoxLayout(self)

        # Page heading and primary actions.
        heading_layout = QHBoxLayout()
        title_layout = QVBoxLayout()

        title = QLabel("Datasets")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Import CSV files, inspect their contents, "
            "and check data quality."
        )
        subtitle.setWordWrap(True)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        self._import_button = QPushButton("Import CSV files")
        self._export_button = QPushButton("Export selected")
        self._remove_button = QPushButton("Remove selected")

        self._export_button.setEnabled(False)
        self._remove_button.setEnabled(False)

        heading_layout.addLayout(title_layout, 1)
        heading_layout.addWidget(self._import_button)
        heading_layout.addWidget(self._export_button)
        heading_layout.addWidget(self._remove_button)

        page_layout.addLayout(heading_layout)

        # High-level statistics for the selected dataset.
        overview_group = QGroupBox("Dataset overview")
        overview_layout = QGridLayout(overview_group)

        self._dataset_count_value = QLabel("0")
        self._rows_value = QLabel("—")
        self._columns_value = QLabel("—")
        self._missing_value = QLabel("—")
        self._duplicates_value = QLabel("—")

        stat_items = [
            ("Loaded datasets", self._dataset_count_value),
            ("Rows", self._rows_value),
            ("Columns", self._columns_value),
            ("Missing cells", self._missing_value),
            ("Duplicate rows", self._duplicates_value),
        ]

        for column_index, (caption, value_label) in enumerate(
            stat_items
        ):
            caption_label = QLabel(caption)
            overview_layout.addWidget(
                caption_label, 0, column_index
            )
            overview_layout.addWidget(
                value_label, 1, column_index
            )

        page_layout.addWidget(overview_group)

        # Left side: imported dataset list.
        self._dataset_list = QListWidget()
        self._dataset_list.setMinimumWidth(220)
        self._dataset_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Loaded datasets"))
        left_layout.addWidget(self._dataset_list)

        # Right side: data preview and profile tabs.
        self._tabs = QTabWidget()

        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search data"))

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText(
            "Search across all columns..."
        )
        self._search_input.setClearButtonEnabled(True)

        search_layout.addWidget(self._search_input, 1)
        preview_layout.addLayout(search_layout)

        self._table_view = QTableView()
        self._table_view.setAlternatingRowColors(True)
        self._table_view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table_view.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table_view.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self._table_model = DatasetTableModel(self)
        self._proxy_model = AllColumnsFilterProxyModel(self)
        self._proxy_model.setSourceModel(self._table_model)

        self._table_view.setModel(self._proxy_model)
        self._table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self._table_view.horizontalHeader().setStretchLastSection(
            True
        )

        preview_layout.addWidget(self._table_view, 1)

        self._tabs.addTab(preview_tab, "Data preview")

        # Column-by-column statistics.
        profile_tab = QWidget()
        profile_layout = QVBoxLayout(profile_tab)

        self._profile_table = QTableWidget()
        self._profile_table.setColumnCount(7)
        self._profile_table.setHorizontalHeaderLabels(
            [
                "Column",
                "Inferred type",
                "Missing",
                "Unique",
                "Minimum",
                "Maximum",
                "Mean",
            ]
        )
        self._profile_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._profile_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._profile_table.setAlternatingRowColors(True)
        self._profile_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._profile_table.horizontalHeader().setStretchLastSection(
            True
        )

        self._quality_label = QLabel(
            "Import a dataset to see its data-quality summary."
        )
        self._quality_label.setWordWrap(True)

        profile_layout.addWidget(self._profile_table, 1)
        profile_layout.addWidget(self._quality_label)

        self._tabs.addTab(profile_tab, "Column profile")

        # Resizable workspace layout.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        page_layout.addWidget(splitter, 1)

        self._status_label = QLabel(
            "Import one or more CSV files to begin."
        )
        self._status_label.setWordWrap(True)
        page_layout.addWidget(self._status_label)

        # Signals connect user actions to page behaviour.
        self._import_button.clicked.connect(self._choose_files)
        self._export_button.clicked.connect(
            self._choose_export_location
        )
        self._remove_button.clicked.connect(
            self.remove_current_dataset
        )
        self._dataset_list.currentItemChanged.connect(
            self._on_dataset_selected
        )
        self._search_input.textChanged.connect(
            self._proxy_model.set_search_text
        )

    def _choose_files(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV files",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        if not file_paths:
            return

        datasets_before = self.dataset_count
        failed_files: list[str] = []

        for file_path in file_paths:
            if not self.add_dataset(
                file_path,
                update_status=False,
            ):
                failed_files.append(Path(file_path).name)

        new_datasets = self.dataset_count - datasets_before

        message = (
            f"Import complete: {new_datasets} new dataset(s) added."
        )

        if failed_files:
            message += (
                " Could not load: "
                + ", ".join(failed_files)
                + ". Check that these files are valid UTF-8 CSV files."
            )

        self._set_status(message)

    def add_dataset(
        self,
        file_path: str | Path,
        *,
        update_status: bool = True,
    ) -> bool:
        """Load and register a CSV file.

        This public method also makes it possible to test the import
        behaviour without opening a file dialog.
        """
        try:
            dataset = load_csv(file_path)
        except DatasetLoadError as exc:
            if update_status:
                self._set_status(f"Could not load CSV: {exc}")

            return False

        dataset_key = str(dataset.path)

        if dataset_key in self._datasets:
            self._select_dataset_by_key(dataset_key)

            if update_status:
                self._set_status(
                    f"{dataset.path.name} is already loaded."
                )

            return True

        self._datasets[dataset_key] = dataset

        item_text = (
            f"{dataset.path.name} — "
            f"{dataset.row_count} rows × "
            f"{dataset.column_count} columns"
        )

        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, dataset_key)
        item.setToolTip(str(dataset.path))

        self._dataset_list.addItem(item)
        self._dataset_list.setCurrentItem(item)

        if update_status:
            self._set_status(
                f"Loaded {dataset.path.name} successfully."
            )

        return True

    def _select_dataset_by_key(self, dataset_key: str) -> None:
        """Select an already-loaded dataset in the list."""
        for row in range(self._dataset_list.count()):
            item = self._dataset_list.item(row)

            if (
                item.data(Qt.ItemDataRole.UserRole)
                == dataset_key
            ):
                self._dataset_list.setCurrentItem(item)
                return

    def _on_dataset_selected(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        del previous  # Not needed for updating the current selection.

        if current is None:
            self._clear_current_dataset()
            return

        dataset_key = current.data(Qt.ItemDataRole.UserRole)
        dataset = self._datasets.get(dataset_key)

        if dataset is None:
            self._clear_current_dataset()
            return

        self._current_path = dataset_key

        self._table_model.set_dataset(dataset)
        self._proxy_model.invalidateFilter()

        profile = profile_dataset(dataset)
        self._update_overview(dataset, profile)
        self._update_column_profile(profile)

        self._export_button.setEnabled(True)
        self._remove_button.setEnabled(True)

    def _update_overview(
        self,
        dataset: LoadedDataset | None,
        profile: DatasetProfile | None,
    ) -> None:
        self._dataset_count_value.setText(
            str(self.dataset_count)
        )

        if dataset is None or profile is None:
            self._rows_value.setText("—")
            self._columns_value.setText("—")
            self._missing_value.setText("—")
            self._duplicates_value.setText("—")
            return

        self._rows_value.setText(str(profile.row_count))
        self._columns_value.setText(str(profile.column_count))
        self._missing_value.setText(str(profile.missing_cells))
        self._duplicates_value.setText(
            str(profile.duplicate_rows)
        )

    @staticmethod
    def _format_number(value: float | None) -> str:
        if value is None:
            return "—"

        return f"{value:.6g}"

    def _update_column_profile(
        self,
        profile: DatasetProfile,
    ) -> None:
        self._profile_table.setRowCount(
            len(profile.columns)
        )

        for row_index, column in enumerate(profile.columns):
            values = [
                column.name,
                column.inferred_type,
                str(column.missing_count),
                str(column.unique_count),
                self._format_number(column.minimum),
                self._format_number(column.maximum),
                self._format_number(column.mean),
            ]

            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(
                    item.flags()
                    & ~Qt.ItemFlag.ItemIsEditable
                )

                self._profile_table.setItem(
                    row_index,
                    column_index,
                    item,
                )

        if profile.missing_cells == 0:
            missing_summary = "No missing cells detected."
        else:
            missing_summary = (
                f"{profile.missing_cells} missing cell(s) detected."
            )

        if profile.duplicate_rows == 0:
            duplicate_summary = "No duplicate rows detected."
        else:
            duplicate_summary = (
                f"{profile.duplicate_rows} repeated row occurrence(s) "
                "detected."
            )

        self._quality_label.setText(
            f"Data quality: {missing_summary} "
            f"{duplicate_summary}"
        )

    def _clear_current_dataset(self) -> None:
        self._current_path = None
        self._table_model.set_dataset(None)
        self._proxy_model.invalidateFilter()
        self._profile_table.setRowCount(0)
        self._search_input.clear()

        self._export_button.setEnabled(False)
        self._remove_button.setEnabled(False)

        self._quality_label.setText(
            "Import a dataset to see its data-quality summary."
        )

        self._update_overview(None, None)

    def _choose_export_location(self) -> None:
        dataset = self.current_dataset

        if dataset is None:
            self._set_status("Select a dataset before exporting.")
            return

        suggested_path = dataset.path.with_name(
            f"{dataset.path.stem}_export.csv"
        )

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export selected dataset",
            str(suggested_path),
            "CSV files (*.csv)",
        )

        if not file_path:
            return

        self.export_current_dataset(file_path)

    def export_current_dataset(
        self,
        file_path: str | Path,
    ) -> bool:
        """Export the complete selected dataset, not just filtered rows."""
        dataset = self.current_dataset

        if dataset is None:
            self._set_status(
                "Select a dataset before exporting."
            )
            return False

        target_path = Path(file_path).expanduser()

        if target_path.suffix.casefold() != ".csv":
            target_path = target_path.with_suffix(".csv")

        try:
            with target_path.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(dataset.columns)
                writer.writerows(dataset.rows)

        except OSError as exc:
            self._set_status(f"Could not export CSV: {exc}")
            return False

        self._set_status(
            f"Exported {dataset.path.name} to {target_path.resolve()}."
        )
        return True

    def remove_current_dataset(
        self,
        checked: bool = False,
        *,
        confirm: bool = True,
    ) -> bool:
        """Remove the selected dataset from the current session."""
        del checked  # Supports connection to QPushButton.clicked.

        item = self._dataset_list.currentItem()

        if item is None:
            self._set_status("Select a dataset to remove.")
            return False

        dataset_key = item.data(Qt.ItemDataRole.UserRole)
        dataset = self._datasets.get(dataset_key)

        if dataset is None:
            return False

        if confirm:
            answer = QMessageBox.question(
                self,
                "Remove dataset",
                (
                    f"Remove {dataset.path.name} from this session? "
                    "The original file will not be deleted."
                ),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if answer != QMessageBox.StandardButton.Yes:
                return False

        row = self._dataset_list.currentRow()
        self._datasets.pop(dataset_key, None)
        self._dataset_list.takeItem(row)

        if self._dataset_list.count() > 0:
            self._dataset_list.setCurrentRow(
                min(row, self._dataset_list.count() - 1)
            )
        else:
            self._clear_current_dataset()

        self._update_overview(
            self.current_dataset,
            (
                profile_dataset(self.current_dataset)
                if self.current_dataset is not None
                else None
            ),
        )

        self._set_status(
            f"Removed {dataset.path.name} from the current session."
        )
        return True

    def _set_status(self, message: str) -> None:
        self._status_label.setText(message)
