
from __future__ import annotations

import csv
from pathlib import Path
import re

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
    QPlainTextEdit,
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
    """Read-only table model for a LoadedDataset."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: LoadedDataset | None = None

    def set_dataset(self, dataset: LoadedDataset | None) -> None:
        self.beginResetModel()
        self._dataset = dataset
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid() or self._dataset is None:
            return 0
        return self._dataset.row_count

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
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

        if role in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.ToolTipRole,
        ):
            return self._dataset.rows[index.row()][index.column()]

        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if role != Qt.ItemDataRole.DisplayRole or self._dataset is None:
            return None

        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < self._dataset.column_count:
                return self._dataset.columns[section]
            return None

        return str(section + 1)


class ColumnFilterProxyModel(QSortFilterProxyModel):
    """Filter rows only by the column selected by the user."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._search_text = ""
        self._search_column: int | None = None

        self.setDynamicSortFilter(True)
        self.setFilterCaseSensitivity(
            Qt.CaseSensitivity.CaseInsensitive
        )

    @property
    def search_column(self) -> int | None:
        return self._search_column

    def set_search_column(self, column: int | None) -> None:
        if column == self._search_column:
            return

        self._search_column = column
        self.invalidateFilter()

    def set_search_text(self, text: str) -> None:
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
        if not self._search_text or self._search_column is None:
            return True

        source_model = self.sourceModel()

        if source_model is None:
            return True

        if self._search_column >= source_model.columnCount(source_parent):
            return True

        index = source_model.index(
            source_row,
            self._search_column,
            source_parent,
        )
        value = source_model.data(
            index,
            Qt.ItemDataRole.DisplayRole,
        )

        if value is None:
            return False

        return self._search_text.casefold() in str(value).casefold()


class DatasetsPage(QWidget):
    """Workspace for importing, exploring, and assessing CSV datasets."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._datasets: dict[str, LoadedDataset] = {}
        self._current_path: str | None = None
        self._selected_search_column: int | None = None

        self._build_ui()

    @property
    def dataset_count(self) -> int:
        """Number of unique datasets loaded in the current session."""
        return len(self._datasets)

    @property
    def current_dataset(self) -> LoadedDataset | None:
        if self._current_path is None:
            return None
        return self._datasets.get(self._current_path)

    def _build_ui(self) -> None:
        page_layout = QVBoxLayout(self)

        # -----------------------------------------------------
        # Page heading and primary actions
        # -----------------------------------------------------
        heading_layout = QHBoxLayout()
        title_layout = QVBoxLayout()

        title = QLabel("Datasets")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Import CSV files, explore their contents, and assess data quality."
        )
        subtitle.setWordWrap(True)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        self._import_button = QPushButton("Import CSV")
        self._export_button = QPushButton("Export CSV")
        self._remove_button = QPushButton("Remove Dataset")

        self._export_button.setEnabled(False)
        self._remove_button.setEnabled(False)

        heading_layout.addLayout(title_layout, 1)
        heading_layout.addWidget(self._import_button)
        heading_layout.addWidget(self._export_button)
        heading_layout.addWidget(self._remove_button)

        page_layout.addLayout(heading_layout)

        # -----------------------------------------------------
        # Dataset overview
        # -----------------------------------------------------
        overview_group = QGroupBox("Dataset Overview")
        overview_layout = QGridLayout(overview_group)

        self._dataset_count_value = QLabel("0")
        self._rows_value = QLabel("—")
        self._columns_value = QLabel("—")
        self._missing_value = QLabel("—")
        self._duplicates_value = QLabel("—")

        overview_items = [
            ("Loaded Datasets", self._dataset_count_value),
            ("Rows", self._rows_value),
            ("Columns", self._columns_value),
            ("Missing Cells", self._missing_value),
            ("Duplicate Rows", self._duplicates_value),
        ]

        for column, (caption, value_label) in enumerate(overview_items):
            overview_layout.addWidget(QLabel(caption), 0, column)
            overview_layout.addWidget(value_label, 1, column)

        page_layout.addWidget(overview_group)

        # -----------------------------------------------------
        # Left panel: dataset list and full file path
        # -----------------------------------------------------
        self._dataset_list = QListWidget()
        self._dataset_list.setMinimumWidth(210)
        self._dataset_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._dataset_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("Loaded Datasets"))
        left_layout.addWidget(self._dataset_list, 1)

        left_layout.addWidget(QLabel("Selected File Path"))

        # A read-only, single-line text view. If the file path is long,
        # the user can scroll horizontally to see the full path and name.
        self._selected_file_path = QPlainTextEdit()
        self._selected_file_path.setObjectName("selectedFilePath")
        self._selected_file_path.setReadOnly(True)
        self._selected_file_path.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.NoWrap
        )
        self._selected_file_path.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._selected_file_path.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._selected_file_path.setFixedHeight(58)
        self._selected_file_path.setPlaceholderText(
            "Select a dataset to view its full file path."
        )
        self._selected_file_path.setToolTip(
            "The full path of the selected CSV file. "
            "Use the horizontal scrollbar to inspect long paths."
        )

        left_layout.addWidget(self._selected_file_path)

        # -----------------------------------------------------
        # Right panel: data preview and quality tabs
        # -----------------------------------------------------
        self._tabs = QTabWidget()

        # Data preview tab.
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)

        search_heading = QHBoxLayout()
        search_heading.addWidget(QLabel("Search in Column"))

        self._search_column_label = QLabel("No column selected")
        search_heading.addWidget(self._search_column_label, 1)
        preview_layout.addLayout(search_heading)

        search_layout = QHBoxLayout()

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText(
            "Select a column in the table first..."
        )
        self._search_input.setClearButtonEnabled(True)
        self._search_input.setEnabled(False)
        self._search_input.setToolTip(
            "Click a column header or a cell in the column you want to search."
        )

        search_layout.addWidget(self._search_input, 1)
        preview_layout.addLayout(search_layout)

        self._search_result_label = QLabel(
            "Select a dataset, then select a column to enable search."
        )
        self._search_result_label.setWordWrap(True)
        preview_layout.addWidget(self._search_result_label)

        self._table_view = QTableView()
        self._table_view.setAlternatingRowColors(True)
        self._table_view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectColumns
        )
        self._table_view.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table_view.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self._table_model = DatasetTableModel(self)
        self._proxy_model = ColumnFilterProxyModel(self)
        self._proxy_model.setSourceModel(self._table_model)
        self._table_view.setModel(self._proxy_model)

        table_header = self._table_view.horizontalHeader()
        table_header.setSectionsClickable(True)
        table_header.setHighlightSections(True)
        table_header.setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        table_header.setStretchLastSection(True)
        table_header.sectionClicked.connect(
            self._on_table_header_clicked
        )

        self._table_view.selectionModel().currentChanged.connect(
            self._on_table_current_changed
        )

        preview_layout.addWidget(self._table_view, 1)
        self._tabs.addTab(preview_tab, "Data Preview")

        # Data quality tab.
        quality_tab = QWidget()
        quality_layout = QVBoxLayout(quality_tab)

        self._quality_summary_label = QLabel(
            "Load a dataset to run data quality checks."
        )
        self._quality_summary_label.setWordWrap(True)
        quality_layout.addWidget(self._quality_summary_label)

        profile_group = QGroupBox("Column Statistics")
        profile_group_layout = QVBoxLayout(profile_group)

        self._profile_table = QTableWidget()
        self._profile_table.setColumnCount(8)
        self._profile_table.setHorizontalHeaderLabels(
            [
                "Column",
                "Detected Type",
                "Missing",
                "Missing (%)",
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
        self._profile_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._profile_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._profile_table.horizontalHeader().setStretchLastSection(True)

        profile_group_layout.addWidget(self._profile_table)
        quality_layout.addWidget(profile_group, 2)

        issues_group = QGroupBox("Findings & Recommendations")
        issues_group_layout = QVBoxLayout(issues_group)

        self._issues_table = QTableWidget()
        self._issues_table.setColumnCount(4)
        self._issues_table.setHorizontalHeaderLabels(
            [
                "Severity",
                "Finding & Details",
                "Column",
                "Recommended Action",
            ]
        )
        self._issues_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._issues_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._issues_table.setAlternatingRowColors(True)
        self._issues_table.setWordWrap(True)
        self._issues_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._issues_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        issues_header = self._issues_table.horizontalHeader()
        issues_header.setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        issues_header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        issues_header.setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )

        issues_group_layout.addWidget(self._issues_table)
        quality_layout.addWidget(issues_group, 3)

        self._tabs.addTab(quality_tab, "Data Quality")

        # -----------------------------------------------------
        # Overall layout and status
        # -----------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        page_layout.addWidget(splitter, 1)

        self._status_label = QLabel("Import a CSV file to get started.")
        self._status_label.setWordWrap(True)
        page_layout.addWidget(self._status_label)

        # -----------------------------------------------------
        # Signal connections
        # -----------------------------------------------------
        self._import_button.clicked.connect(self._choose_files)
        self._export_button.clicked.connect(self._choose_export_location)
        self._remove_button.clicked.connect(self.remove_current_dataset)
        self._dataset_list.currentItemChanged.connect(
            self._on_dataset_selected
        )
        self._search_input.textChanged.connect(
            self._on_search_text_changed
        )

    # ---------------------------------------------------------
    # Scrollable text helpers
    # ---------------------------------------------------------

    @staticmethod
    def _create_scrollable_text(
        text: str,
        *,
        height: int = 78,
    ) -> QPlainTextEdit:
        """Create a read-only text area with scrolling for long content."""
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(text)
        editor.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth
        )
        editor.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        editor.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        editor.setMinimumHeight(height)
        editor.setMaximumHeight(height)
        editor.setToolTip(text)
        return editor

    # ---------------------------------------------------------
    # Import and dataset registry
    # ---------------------------------------------------------

    @staticmethod
    def _format_load_error(error: DatasetLoadError) -> str:
        """Format CSV loader errors as readable English messages."""
        message = str(error)

        if message.startswith("File does not exist:"):
            path = message.split(":", 1)[1].strip()
            return f"File does not exist: {path}"

        if message.startswith("The selected path is not a file:"):
            path = message.split(":", 1)[1].strip()
            return f"The selected path is not a file: {path}"

        if "must have a .csv extension" in message:
            return "Please select a file with the .csv extension."

        if message == "The CSV file is empty.":
            return "The CSV file is empty."

        if "first row must contain column names" in message:
            return "The first row must contain column names."

        if "Every column must have a name" in message:
            return "Every column must have a name."

        if "Column names must be unique" in message:
            return "Column names must be unique."

        if message.startswith("Invalid record "):
            match = re.search(
                r"Invalid record (\d+): expected (\d+) values, but found (\d+)",
                message,
            )
            if match:
                record, expected, found = match.groups()
                return (
                    f"Invalid record {record}: expected {expected} values, "
                    f"but found {found}. Check the CSV file structure."
                )
            return (
                "Some rows have a different number of values than the header."
            )

        if "could not be read as UTF-8" in message:
            return (
                "The file could not be read as UTF-8. "
                "Save it as UTF-8 and try again."
            )

        if message.startswith("The CSV format is invalid:"):
            return (
                "The CSV structure is invalid. "
                "Check the delimiters and quotation marks."
            )

        if message.startswith("Could not open the CSV file:"):
            return "Could not open the CSV file. Check file permissions."

        return "Could not load the CSV file. Check the file and try again."

    def _choose_files(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV Files",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )

        if not file_paths:
            return

        initial_count = self.dataset_count
        failed_files: list[str] = []

        for file_path in file_paths:
            if not self.add_dataset(
                file_path,
                update_status=False,
            ):
                failed_files.append(Path(file_path).name)

        added_count = self.dataset_count - initial_count
        message = (
            f"Import finished. Added {added_count} new dataset(s)."
        )

        if failed_files:
            message += " Failed to load: " + ", ".join(failed_files) + "."

        self._set_status(message)

    def add_dataset(
        self,
        file_path: str | Path,
        *,
        update_status: bool = True,
    ) -> bool:
        """Load a CSV file and add it to the current session."""
        try:
            dataset = load_csv(file_path)
        except DatasetLoadError as exc:
            if update_status:
                self._set_status(
                    f"Could not load CSV: {self._format_load_error(exc)}"
                )
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

        # Keep the list compact; full name and path remain available
        # in the tooltip and the scrollable Selected File Path field.
        item = QListWidgetItem(
            f"{dataset.path.name}  ({dataset.row_count} rows)"
        )
        item.setData(Qt.ItemDataRole.UserRole, dataset_key)
        item.setToolTip(
            f"File: {dataset.path.name}\nFull path: {dataset.path}\n"
            f"Rows: {dataset.row_count}\nColumns: {dataset.column_count}"
        )

        self._dataset_list.addItem(item)
        self._dataset_list.setCurrentItem(item)

        if update_status:
            self._set_status(f"Loaded {dataset.path.name} successfully.")

        return True

    def _select_dataset_by_key(self, dataset_key: str) -> None:
        for row in range(self._dataset_list.count()):
            item = self._dataset_list.item(row)

            if item.data(Qt.ItemDataRole.UserRole) == dataset_key:
                self._dataset_list.setCurrentItem(item)
                return

    # ---------------------------------------------------------
    # Dataset and column selection
    # ---------------------------------------------------------

    def _on_dataset_selected(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        del previous

        if current is None:
            self._clear_current_dataset()
            return

        dataset_key = current.data(Qt.ItemDataRole.UserRole)
        dataset = self._datasets.get(dataset_key)

        if dataset is None:
            self._clear_current_dataset()
            return

        self._current_path = dataset_key
        self._reset_search_column()

        # Display the complete path, even when it is longer than the panel.
        self._selected_file_path.setPlainText(str(dataset.path))
        self._selected_file_path.setToolTip(str(dataset.path))

        self._table_view.clearSelection()
        self._table_model.set_dataset(dataset)
        self._proxy_model.invalidateFilter()

        profile = profile_dataset(dataset)
        self._update_overview(dataset, profile)
        self._update_column_profile(profile)
        self._update_quality_issues(profile)
        self._update_quality_summary(profile)

        self._export_button.setEnabled(True)
        self._remove_button.setEnabled(True)

    def _on_table_header_clicked(self, column: int) -> None:
        if self.current_dataset is None:
            return

        if not 0 <= column < self.current_dataset.column_count:
            return

        self._table_view.selectColumn(column)
        self._activate_search_column(column)

    def _on_table_current_changed(
        self,
        current: QModelIndex,
        previous: QModelIndex,
    ) -> None:
        del previous

        if current.isValid():
            self._activate_search_column(current.column())

    def _activate_search_column(self, column: int) -> None:
        dataset = self.current_dataset

        if dataset is None or not 0 <= column < dataset.column_count:
            return

        column_changed = column != self._selected_search_column
        self._selected_search_column = column
        column_name = dataset.columns[column]

        self._proxy_model.set_search_column(column)
        self._search_column_label.setText(
            f"Selected column: {column_name}"
        )
        self._search_input.setEnabled(True)
        self._search_input.setPlaceholderText(
            f"Search in {column_name}..."
        )
        self._search_input.setToolTip(
            f"Search is limited to the {column_name} column."
        )

        if column_changed:
            self._search_input.clear()

        self._update_search_result_label()

    def _reset_search_column(self) -> None:
        self._selected_search_column = None
        self._proxy_model.set_search_column(None)

        self._search_input.clear()
        self._search_input.setEnabled(False)
        self._search_input.setPlaceholderText(
            "Select a column in the table first..."
        )
        self._search_input.setToolTip(
            "Click a column header or a cell in the column you want to search."
        )

        self._search_column_label.setText("No column selected")
        self._update_search_result_label()

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def _on_search_text_changed(self, text: str) -> None:
        self._proxy_model.set_search_text(text)
        self._update_search_result_label()

    def _update_search_result_label(self) -> None:
        dataset = self.current_dataset

        if dataset is None:
            self._search_result_label.setText(
                "Select a dataset, then select a column to enable search."
            )
            return

        column = self._selected_search_column

        if column is None:
            self._search_result_label.setText(
                f"{dataset.row_count} rows available. Select a column to search."
            )
            return

        column_name = dataset.columns[column]
        query = self._search_input.text().strip()
        visible_count = self._proxy_model.rowCount()

        if not query:
            self._search_result_label.setText(
                f"Showing all {dataset.row_count} rows. "
                f"Search column: {column_name}."
            )
        else:
            self._search_result_label.setText(
                f"{visible_count} of {dataset.row_count} rows match "
                f"'{query}' in column '{column_name}'."
            )

    # ---------------------------------------------------------
    # Dataset overview and column statistics
    # ---------------------------------------------------------

    def _update_overview(
        self,
        dataset: LoadedDataset | None,
        profile: DatasetProfile | None,
    ) -> None:
        self._dataset_count_value.setText(str(self.dataset_count))

        if dataset is None or profile is None:
            self._rows_value.setText("—")
            self._columns_value.setText("—")
            self._missing_value.setText("—")
            self._duplicates_value.setText("—")
            return

        self._rows_value.setText(str(profile.row_count))
        self._columns_value.setText(str(profile.column_count))
        self._missing_value.setText(
            f"{profile.missing_cells} ({profile.missing_percentage:.1f}%)"
        )
        self._duplicates_value.setText(str(profile.duplicate_rows))

    @staticmethod
    def _format_number(value: float | None) -> str:
        if value is None:
            return "—"
        return f"{value:.6g}"

    def _update_column_profile(self, profile: DatasetProfile) -> None:
        self._profile_table.setRowCount(len(profile.columns))

        type_labels = {
            "integer": "Integer",
            "decimal": "Decimal",
            "boolean": "Boolean",
            "date": "Date",
            "category": "Category",
            "text": "Text",
            "mixed": "Mixed",
            "empty": "Empty",
        }

        for row_index, column in enumerate(profile.columns):
            values = [
                column.name,
                type_labels.get(column.inferred_type, "Unknown"),
                str(column.missing_count),
                f"{column.missing_percentage:.1f}%",
                str(column.unique_count),
                self._format_number(column.minimum),
                self._format_number(column.maximum),
                self._format_number(column.mean),
            ]

            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(
                    item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                item.setToolTip(value)
                self._profile_table.setItem(
                    row_index,
                    column_index,
                    item,
                )

    def _update_quality_issues(self, profile: DatasetProfile) -> None:
        self._issues_table.setRowCount(len(profile.issues))

        severity_labels = {
            "error": "Error",
            "warning": "Warning",
            "info": "Info",
        }

        for row_index, issue in enumerate(profile.issues):
            severity_item = QTableWidgetItem(
                severity_labels.get(issue.severity, "Info")
            )
            severity_item.setToolTip(severity_item.text())
            severity_item.setFlags(
                severity_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            self._issues_table.setItem(row_index, 0, severity_item)

            details_text = f"{issue.title}\n{issue.description}"
            details_widget = self._create_scrollable_text(
                details_text,
                height=78,
            )
            self._issues_table.setCellWidget(
                row_index,
                1,
                details_widget,
            )

            column_text = issue.column_name or "Dataset"
            column_item = QTableWidgetItem(column_text)
            column_item.setToolTip(column_text)
            column_item.setFlags(
                column_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            self._issues_table.setItem(row_index, 2, column_item)

            recommendation_widget = self._create_scrollable_text(
                issue.recommendation,
                height=78,
            )
            self._issues_table.setCellWidget(
                row_index,
                3,
                recommendation_widget,
            )

            # The scrollbars inside each text cell handle long content.
            # The table's vertical scrollbar handles the list of findings.
            self._issues_table.setRowHeight(row_index, 82)

    def _update_quality_summary(self, profile: DatasetProfile) -> None:
        error_count = sum(
            issue.severity == "error"
            for issue in profile.issues
        )
        warning_count = sum(
            issue.severity == "warning"
            for issue in profile.issues
        )
        info_count = sum(
            issue.severity == "info"
            for issue in profile.issues
        )

        summary = (
            f"Total cells: {profile.total_cells}. "
            f"Missing cells: {profile.missing_cells} "
            f"({profile.missing_percentage:.1f}%). "
            f"Duplicate rows: {profile.duplicate_rows}.\n"
            f"Check results: {error_count} errors, "
            f"{warning_count} warnings and "
            f"{info_count} informational findings."
        )

        if not profile.issues:
            summary += (
                "\nNo issues were found by the current checks. "
                "This does not guarantee that the data is suitable "
                "for every machine-learning task."
            )
        else:
            summary += (
                "\nReview the recommendations before using this dataset "
                "for training. Potential outliers are not necessarily errors."
            )

        self._quality_summary_label.setText(summary)

    def _clear_current_dataset(self) -> None:
        self._current_path = None
        self._reset_search_column()
        self._table_view.clearSelection()
        self._table_model.set_dataset(None)
        self._proxy_model.invalidateFilter()

        self._selected_file_path.clear()
        self._profile_table.setRowCount(0)
        self._issues_table.setRowCount(0)

        self._quality_summary_label.setText(
            "Load a dataset to run data quality checks."
        )
        self._export_button.setEnabled(False)
        self._remove_button.setEnabled(False)

        self._update_overview(None, None)

    # ---------------------------------------------------------
    # Export
    # ---------------------------------------------------------

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
            "Export Dataset",
            str(suggested_path),
            "CSV Files (*.csv)",
        )

        if file_path:
            self.export_current_dataset(file_path)

    def export_current_dataset(
        self,
        file_path: str | Path,
    ) -> bool:
        """Export all source rows, including rows hidden by search."""
        dataset = self.current_dataset

        if dataset is None:
            self._set_status("Select a dataset before exporting.")
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
            self._set_status(
                f"Could not export the CSV file: {exc}"
            )
            return False

        self._set_status(
            f"Dataset exported successfully to: {target_path.resolve()}"
        )
        return True

    # ---------------------------------------------------------
    # Remove dataset
    # ---------------------------------------------------------

    def remove_current_dataset(
        self,
        checked: bool = False,
        *,
        confirm: bool = True,
    ) -> bool:
        """Remove a dataset from this session without deleting its file."""
        del checked

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
                "Remove Dataset",
                f"Remove {dataset.path.name} from the current session? "
                "The original file will not be deleted.",
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

        current_dataset = self.current_dataset
        current_profile = (
            profile_dataset(current_dataset)
            if current_dataset is not None
            else None
        )

        self._update_overview(
            current_dataset,
            current_profile,
        )

        self._set_status(
            f"Removed {dataset.path.name} from the session. "
            "The original file was not deleted."
        )
        return True

    def _set_status(self, message: str) -> None:
        self._status_label.setText(message)
