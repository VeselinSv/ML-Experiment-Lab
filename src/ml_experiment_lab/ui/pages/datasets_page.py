
"""Страница за управление и анализ на набори от данни."""

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
    QInputDialog,
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
    get_excel_sheet_names,
    load_dataset,
    profile_dataset,
)


class DatasetTableModel(QAbstractTableModel):
    """Модел само за четене на таблица с данни."""

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
    """Филтрира редовете само по избраната от потребителя колона."""

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
        value = source_model.data(index, Qt.ItemDataRole.DisplayRole)

        if value is None:
            return False

        return self._search_text.casefold() in str(value).casefold()


class DatasetsPage(QWidget):
    """Работно пространство за управление на набори от данни."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._datasets: dict[str, LoadedDataset] = {}
        self._current_path: str | None = None
        self._selected_search_column: int | None = None
        self._build_ui()

    @property
    def dataset_count(self) -> int:
        """Брой уникални заредени набори в текущата сесия."""
        return len(self._datasets)

    @property
    def current_dataset(self) -> LoadedDataset | None:
        """Връща текущо избрания набор от данни."""
        if self._current_path is None:
            return None
        return self._datasets.get(self._current_path)

    @staticmethod
    def _dataset_key(dataset: LoadedDataset) -> str:
        """Осигурява отделен ключ за всеки Excel работен лист."""
        key = str(dataset.path)

        if dataset.sheet_name is not None:
            key += f"::лист::{dataset.sheet_name}"

        return key

    def _build_ui(self) -> None:
        page_layout = QVBoxLayout(self)

        # Заглавие и основни действия.
        heading_layout = QHBoxLayout()
        title_layout = QVBoxLayout()

        title = QLabel("Набори от данни")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Импортирайте CSV и Excel файлове, разглеждайте "
            "съдържанието им и проверявайте качеството на данните."
        )
        subtitle.setWordWrap(True)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        self._import_button = QPushButton("Импортиране на данни")
        self._export_button = QPushButton("Експортиране в CSV")
        self._remove_button = QPushButton("Премахване на набор")

        self._export_button.setEnabled(False)
        self._remove_button.setEnabled(False)

        heading_layout.addLayout(title_layout, 1)
        heading_layout.addWidget(self._import_button)
        heading_layout.addWidget(self._export_button)
        heading_layout.addWidget(self._remove_button)
        page_layout.addLayout(heading_layout)

        # Обобщена информация.
        overview_group = QGroupBox("Обща информация")
        overview_layout = QGridLayout(overview_group)

        self._dataset_count_value = QLabel("0")
        self._rows_value = QLabel("—")
        self._columns_value = QLabel("—")
        self._missing_value = QLabel("—")
        self._duplicates_value = QLabel("—")

        overview_items = [
            ("Заредени набори", self._dataset_count_value),
            ("Редове", self._rows_value),
            ("Колони", self._columns_value),
            ("Липсващи клетки", self._missing_value),
            ("Дублирани редове", self._duplicates_value),
        ]

        for column, (caption, value_label) in enumerate(overview_items):
            overview_layout.addWidget(QLabel(caption), 0, column)
            overview_layout.addWidget(value_label, 1, column)

        page_layout.addWidget(overview_group)

        # Ляв панел: списък и пълен път до файла.
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

        left_layout.addWidget(QLabel("Заредени набори от данни"))
        left_layout.addWidget(self._dataset_list, 1)
        left_layout.addWidget(QLabel("Пълен път до избрания файл"))

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
            "Изберете набор, за да видите пълния път до файла."
        )
        self._selected_file_path.setToolTip(
            "Пълният път до файла. Използвайте хоризонталния скролер "
            "за преглед на дълги пътища."
        )
        left_layout.addWidget(self._selected_file_path)

        # Десен панел: преглед на данните и качество.
        self._tabs = QTabWidget()

        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)

        search_heading = QHBoxLayout()
        search_heading.addWidget(QLabel("Търсене в колона"))
        self._search_column_label = QLabel("Няма избрана колона")
        search_heading.addWidget(self._search_column_label, 1)
        preview_layout.addLayout(search_heading)

        search_layout = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText(
            "Първо изберете колона от таблицата..."
        )
        self._search_input.setClearButtonEnabled(True)
        self._search_input.setEnabled(False)
        self._search_input.setToolTip(
            "Изберете заглавие или клетка от колоната, "
            "в която искате да търсите."
        )
        search_layout.addWidget(self._search_input, 1)
        preview_layout.addLayout(search_layout)

        self._search_result_label = QLabel(
            "Изберете набор от данни, а след това колона за търсене."
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
        self._tabs.addTab(preview_tab, "Преглед на данните")

        # Раздел за качеството на данните.
        quality_tab = QWidget()
        quality_layout = QVBoxLayout(quality_tab)

        self._quality_summary_label = QLabel(
            "Заредете набор от данни, за да проверите качеството му."
        )
        self._quality_summary_label.setWordWrap(True)
        quality_layout.addWidget(self._quality_summary_label)

        profile_group = QGroupBox("Статистики по колони")
        profile_group_layout = QVBoxLayout(profile_group)

        self._profile_table = QTableWidget()
        self._profile_table.setColumnCount(8)
        self._profile_table.setHorizontalHeaderLabels(
            [
                "Колона",
                "Разпознат тип",
                "Липсващи",
                "Липсващи (%)",
                "Уникални",
                "Минимум",
                "Максимум",
                "Средна стойност",
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

        issues_group = QGroupBox("Открити проблеми и препоръки")
        issues_group_layout = QVBoxLayout(issues_group)

        self._issues_table = QTableWidget()
        self._issues_table.setColumnCount(4)
        self._issues_table.setHorizontalHeaderLabels(
            [
                "Сериозност",
                "Проблем и подробности",
                "Колона",
                "Препоръчително действие",
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

        self._tabs.addTab(quality_tab, "Качество на данните")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        page_layout.addWidget(splitter, 1)

        self._status_label = QLabel(
            "Импортирайте CSV или Excel файл, за да започнете."
        )
        self._status_label.setWordWrap(True)
        page_layout.addWidget(self._status_label)

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
    # Скролируем текст
    # ---------------------------------------------------------

    @staticmethod
    def _create_scrollable_text(
        text: str,
        *,
        height: int = 78,
    ) -> QPlainTextEdit:
        """Създава поле само за четене със скролер."""
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
    # Зареждане на CSV и Excel
    # ---------------------------------------------------------

    @staticmethod
    def _format_load_error(error: DatasetLoadError) -> str:
        """Преобразува грешките при зареждане в български съобщения."""
        message = str(error)

        if message.startswith("File does not exist:"):
            path = message.split(":", 1)[1].strip()
            return f"Файлът не съществува: {path}"

        if message.startswith("The selected path is not a file:"):
            path = message.split(":", 1)[1].strip()
            return f"Избраният път не сочи към файл: {path}"

        if "must have a .csv extension" in message:
            return "Изберете файл с разширение .csv."

        if "must have an .xlsx extension" in message:
            return "Изберете файл с разширение .xlsx."

        if "Supported file types" in message:
            return "Поддържат се файлове с разширение .csv и .xlsx."

        if message == "The CSV file is empty.":
            return "CSV файлът е празен."

        if "selected worksheet is empty" in message:
            return "Избраният работен лист е празен."

        if "does not contain worksheets" in message:
            return "Excel файлът не съдържа работни листове."

        if "Worksheet not found:" in message:
            return "Избраният работен лист не е намерен."

        if "Could not open the Excel workbook:" in message:
            return (
                "Excel файлът не може да бъде отворен. "
                "Проверете дали файлът е валиден и не е повреден."
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
            match = re.search(
                r"Invalid record (\d+): expected (\d+) values, but found (\d+)",
                message,
            )
            if match:
                record, expected, found = match.groups()
                return (
                    f"Невалиден ред {record}: очакват се {expected} "
                    f"стойности, но са открити {found}. "
                    "Проверете структурата на файла."
                )

            match = re.search(r"Invalid record (\d+)", message)
            if match:
                return (
                    f"Невалиден ред {match.group(1)}: открити са данни "
                    "извън именуваните колони."
                )

            return "Някои редове имат невалидна структура."

        if "could not be read as UTF-8" in message:
            return (
                "Файлът не може да бъде прочетен като UTF-8. "
                "Запазете го в UTF-8 и опитайте отново."
            )

        if message.startswith("The CSV format is invalid:"):
            return (
                "Структурата на CSV файла е невалидна. "
                "Проверете разделителите и кавичките."
            )

        if message.startswith("Could not open the CSV file:"):
            return (
                "CSV файлът не може да бъде отворен. "
                "Проверете правата за достъп."
            )

        return (
            "Файлът не може да бъде зареден. "
            "Проверете формата и съдържанието му."
        )

    def _choose_files(self) -> None:
        """Отваря диалог за избор на CSV и Excel файлове."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Избор на набори от данни",
            "",
            (
                "Поддържани файлове (*.csv *.xlsx);;"
                "CSV файлове (*.csv);;"
                "Excel файлове (*.xlsx);;"
                "Всички файлове (*)"
            ),
        )

        if not file_paths:
            return

        initial_count = self.dataset_count
        failed_files: list[str] = []
        cancelled_count = 0

        for file_path in file_paths:
            path = Path(file_path)
            sheet_name: str | None = None

            if path.suffix.casefold() == ".xlsx":
                try:
                    sheet_names = get_excel_sheet_names(path)
                except DatasetLoadError as exc:
                    failed_files.append(
                        f"{path.name} ({self._format_load_error(exc)})"
                    )
                    continue

                if not sheet_names:
                    failed_files.append(
                        f"{path.name} (няма работни листове)"
                    )
                    continue

                if len(sheet_names) == 1:
                    sheet_name = sheet_names[0]
                else:
                    sheet_name, accepted = QInputDialog.getItem(
                        self,
                        "Избор на работен лист",
                        (
                            f"Excel файлът „{path.name}“ съдържа "
                            "няколко работни листа.\n"
                            "Изберете кой лист да бъде зареден:"
                        ),
                        list(sheet_names),
                        0,
                        False,
                    )

                    if not accepted:
                        cancelled_count += 1
                        continue

            if not self.add_dataset(
                path,
                sheet_name=sheet_name,
                update_status=False,
            ):
                failed_files.append(path.name)

        added_count = self.dataset_count - initial_count
        message = (
            f"Импортирането приключи. Добавени са "
            f"{added_count} нови набора от данни."
        )

        if cancelled_count:
            message += (
                f" Пропуснати по избор на потребителя: {cancelled_count}."
            )

        if failed_files:
            message += (
                " Неуспешно зареждане: "
                + "; ".join(failed_files)
                + "."
            )

        self._set_status(message)

    def add_dataset(
        self,
        file_path: str | Path,
        *,
        sheet_name: str | None = None,
        update_status: bool = True,
    ) -> bool:
        """Зарежда CSV файл или избран лист от Excel файл."""
        try:
            dataset = load_dataset(
                file_path,
                sheet_name=sheet_name,
            )
        except DatasetLoadError as exc:
            if update_status:
                self._set_status(
                    f"Неуспешно зареждане: "
                    f"{self._format_load_error(exc)}"
                )
            return False

        dataset_key = self._dataset_key(dataset)

        if dataset_key in self._datasets:
            self._select_dataset_by_key(dataset_key)

            if update_status:
                self._set_status(
                    "Този файл или работен лист вече е зареден."
                )
            return True

        self._datasets[dataset_key] = dataset

        sheet_suffix = (
            f" — лист: {dataset.sheet_name}"
            if dataset.sheet_name is not None
            else ""
        )

        item = QListWidgetItem(
            f"{dataset.path.name}{sheet_suffix} "
            f"({dataset.row_count} реда)"
        )
        item.setData(Qt.ItemDataRole.UserRole, dataset_key)

        tooltip = (
            f"Файл: {dataset.path.name}\n"
            f"Пълен път: {dataset.path}\n"
        )

        if dataset.sheet_name is not None:
            tooltip += f"Работен лист: {dataset.sheet_name}\n"

        tooltip += (
            f"Редове: {dataset.row_count}\n"
            f"Колони: {dataset.column_count}"
        )
        item.setToolTip(tooltip)

        self._dataset_list.addItem(item)
        self._dataset_list.setCurrentItem(item)

        if update_status:
            self._set_status(
                f"Файлът {dataset.path.name} е зареден успешно."
            )

        return True

    def _select_dataset_by_key(self, dataset_key: str) -> None:
        for row in range(self._dataset_list.count()):
            item = self._dataset_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == dataset_key:
                self._dataset_list.setCurrentItem(item)
                return

    # ---------------------------------------------------------
    # Избор на набор и колона
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

        self._selected_file_path.setPlainText(str(dataset.path))

        tooltip = str(dataset.path)
        if dataset.sheet_name is not None:
            tooltip += f"\nРаботен лист: {dataset.sheet_name}"
        self._selected_file_path.setToolTip(tooltip)

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
            f"Избрана колона: {column_name}"
        )
        self._search_input.setEnabled(True)
        self._search_input.setPlaceholderText(
            f"Търсене в „{column_name}“..."
        )
        self._search_input.setToolTip(
            f"Търсенето е ограничено до колоната „{column_name}“."
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
            "Първо изберете колона от таблицата..."
        )
        self._search_input.setToolTip(
            "Изберете заглавие или клетка от колоната, "
            "в която искате да търсите."
        )
        self._search_column_label.setText("Няма избрана колона")
        self._update_search_result_label()

    # ---------------------------------------------------------
    # Търсене
    # ---------------------------------------------------------

    def _on_search_text_changed(self, text: str) -> None:
        self._proxy_model.set_search_text(text)
        self._update_search_result_label()

    def _update_search_result_label(self) -> None:
        dataset = self.current_dataset

        if dataset is None:
            self._search_result_label.setText(
                "Изберете набор от данни и колона за търсене."
            )
            return

        column = self._selected_search_column

        if column is None:
            self._search_result_label.setText(
                f"Налични редове: {dataset.row_count}. "
                "Изберете колона, за да търсите."
            )
            return

        column_name = dataset.columns[column]
        query = self._search_input.text().strip()
        visible_count = self._proxy_model.rowCount()

        if not query:
            self._search_result_label.setText(
                f"Показани са всички {dataset.row_count} реда. "
                f"Колона за търсене: {column_name}."
            )
        else:
            self._search_result_label.setText(
                f"{visible_count} от {dataset.row_count} реда съвпадат "
                f"с „{query}“ в колоната „{column_name}“."
            )

    # ---------------------------------------------------------
    # Обобщение и статистики
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
            "integer": "Цяло число",
            "decimal": "Десетично число",
            "boolean": "Булева стойност",
            "date": "Дата",
            "category": "Категория",
            "text": "Текст",
            "mixed": "Смесени типове",
            "empty": "Празна колона",
        }

        for row_index, column in enumerate(profile.columns):
            values = [
                column.name,
                type_labels.get(column.inferred_type, "Неизвестен тип"),
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

    def _translated_issue(
        self,
        issue,
        profile: DatasetProfile,
    ) -> tuple[str, str, str]:
        """Връща преведени заглавие, описание и препоръка."""
        columns = {column.name: column for column in profile.columns}
        column = columns.get(issue.column_name or "")

        if issue.code == "empty_dataset":
            return (
                "Наборът не съдържа редове с данни",
                "Файлът съдържа имената на колоните, но няма записи.",
                "Проверете дали е избран правилният файл и дали съдържа данни.",
            )

        if issue.code == "empty_column":
            name = issue.column_name or "?"
            return (
                "Колоната няма стойности",
                f"Всичките {issue.affected_count} стойности в "
                f"колоната „{name}“ липсват.",
                "Проверете източника на данните. Ако колоната не е "
                "необходима, обмислете премахването ѝ след преглед.",
            )

        if issue.code == "missing_values":
            name = issue.column_name or "?"
            missing_count = (
                column.missing_count
                if column
                else issue.affected_count
            )
            percentage = (
                column.missing_percentage if column else 0.0
            )
            title = (
                "Голям брой липсващи стойности"
                if percentage >= 20
                else "Липсващи стойности"
            )
            return (
                title,
                f"Колоната „{name}“ съдържа {missing_count} "
                f"липсващи стойности от {profile.row_count} "
                f"({percentage:.1f}%).",
                "Проверете защо стойностите липсват. Преди обучение "
                "изберете подходящ начин за попълване или премахване "
                "на непълни редове или колони.",
            )

        if issue.code == "constant_column":
            name = issue.column_name or "?"
            return (
                "Колона с постоянна стойност",
                f"Всички налични стойности в колоната „{name}“ са еднакви.",
                "Постоянната характеристика обикновено не помага на модела "
                "да различава наблюденията. Проверете дали е необходима.",
            )

        if issue.code == "mixed_values":
            name = issue.column_name or "?"
            numeric_count = column.mixed_numeric_count if column else 0
            other_count = (
                column.mixed_other_count
                if column
                else issue.affected_count
            )
            return (
                "Възможно смесване на типове",
                f"Колоната „{name}“ съдържа {numeric_count} числови "
                f"и {other_count} нечислови стойности.",
                "Проверете нечисловите стойности. Те може да са грешки, "
                "специални маркери или валидни категории.",
            )

        if issue.code == "potential_outliers":
            name = issue.column_name or "?"
            return (
                "Потенциални крайни стойности",
                f"В числовата колона „{name}“ са открити "
                f"{issue.affected_count} потенциални крайни стойности "
                "според правилото 1,5 × IQR.",
                "Проверете стойностите спрямо контекста. Не ги "
                "премахвайте автоматично — може да са валидни наблюдения.",
            )

        if issue.code == "duplicate_rows":
            return (
                "Дублирани редове",
                f"Открити са {issue.affected_count} повторени реда "
                "след първото срещане на всеки ред.",
                "Проверете повторенията и потвърдете дали са дубликати "
                "или валидни отделни наблюдения, преди да ги премахнете.",
            )

        return (
            issue.title,
            issue.description,
            issue.recommendation,
        )

    def _update_quality_issues(self, profile: DatasetProfile) -> None:
        self._issues_table.setRowCount(len(profile.issues))

        severity_labels = {
            "error": "Грешка",
            "warning": "Предупреждение",
            "info": "Информация",
        }

        for row_index, issue in enumerate(profile.issues):
            severity_text = severity_labels.get(
                issue.severity,
                "Информация",
            )
            severity_item = QTableWidgetItem(severity_text)
            severity_item.setToolTip(severity_text)
            severity_item.setFlags(
                severity_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            self._issues_table.setItem(row_index, 0, severity_item)

            title, description, recommendation = self._translated_issue(
                issue, profile
            )
            details_widget = self._create_scrollable_text(
                f"{title}\n{description}",
                height=78,
            )
            self._issues_table.setCellWidget(
                row_index, 1, details_widget
            )

            column_text = issue.column_name or "Целият набор"
            column_item = QTableWidgetItem(column_text)
            column_item.setToolTip(column_text)
            column_item.setFlags(
                column_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            self._issues_table.setItem(row_index, 2, column_item)

            recommendation_widget = self._create_scrollable_text(
                recommendation,
                height=78,
            )
            self._issues_table.setCellWidget(
                row_index, 3, recommendation_widget
            )
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
            f"Общо клетки: {profile.total_cells}. "
            f"Липсващи клетки: {profile.missing_cells} "
            f"({profile.missing_percentage:.1f}%). "
            f"Дублирани редове: {profile.duplicate_rows}.\n"
            f"Резултати от проверката: {error_count} грешки, "
            f"{warning_count} предупреждения и "
            f"{info_count} информационни бележки."
        )

        if not profile.issues:
            summary += (
                "\nНе са открити проблеми при текущите проверки. "
                "Това не гарантира, че данните са подходящи за всяка "
                "задача по машинно обучение."
            )
        else:
            summary += (
                "\nПрегледайте препоръките, преди да използвате набора "
                "за обучение. Потенциалните крайни стойности не са "
                "непременно грешки."
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
            "Заредете набор от данни, за да проверите качеството му."
        )
        self._export_button.setEnabled(False)
        self._remove_button.setEnabled(False)
        self._update_overview(None, None)

    # ---------------------------------------------------------
    # Експортиране
    # ---------------------------------------------------------

    def _choose_export_location(self) -> None:
        dataset = self.current_dataset

        if dataset is None:
            self._set_status(
                "Изберете набор от данни преди експортиране."
            )
            return

        suggested_path = dataset.path.with_name(
            f"{dataset.path.stem}_export.csv"
        )

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Експортиране на набор от данни",
            str(suggested_path),
            "CSV файлове (*.csv)",
        )

        if file_path:
            self.export_current_dataset(file_path)

    def export_current_dataset(
        self,
        file_path: str | Path,
    ) -> bool:
        """Експортира всички редове, включително скритите от търсенето."""
        dataset = self.current_dataset

        if dataset is None:
            self._set_status(
                "Изберете набор от данни преди експортиране."
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
            self._set_status(
                f"CSV файлът не може да бъде експортиран: {exc}"
            )
            return False

        self._set_status(
            f"Наборът е експортиран успешно в: {target_path.resolve()}"
        )
        return True

    # ---------------------------------------------------------
    # Премахване на набор
    # ---------------------------------------------------------

    def remove_current_dataset(
        self,
        checked: bool = False,
        *,
        confirm: bool = True,
    ) -> bool:
        """Премахва набора от текущата сесия, без да изтрива файла."""
        del checked

        item = self._dataset_list.currentItem()

        if item is None:
            self._set_status(
                "Изберете набор от данни за премахване."
            )
            return False

        dataset_key = item.data(Qt.ItemDataRole.UserRole)
        dataset = self._datasets.get(dataset_key)

        if dataset is None:
            return False

        if confirm:
            answer = QMessageBox.question(
                self,
                "Премахване на набор от данни",
                f"Да се премахне ли {dataset.path.name} от текущата сесия? "
                "Оригиналният файл няма да бъде изтрит.",
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
            f"{dataset.path.name} е премахнат от текущата сесия. "
            "Оригиналният файл не е изтрит."
        )
        return True

    def _set_status(self, message: str) -> None:
        self._status_label.setText(message)
